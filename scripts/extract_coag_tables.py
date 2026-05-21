"""One-shot extractor for the lookup tables in `modal_aero_coag.F90:getcoags`.

Parses the Fortran ``data`` declarations and writes them out as numpy
``.npz`` arrays under ``mam4_jax/_coag_tables.npz``. Run once when the
upstream Fortran source changes; the resulting file is committed.

Tables extracted:
  * bm0     (10,)        — m0 intramodal FM correction
  * bm0ij   (10,10,10)   — m0 intermodal FM correction
  * bm3i    (10,10,10)   — m3 intermodal FM correction
  * bm2ii   (10,)        — m2 intramodal FM correction
  * bm2iitt (10,)        — m2 intramodal total correction
  * bm2ij   (10,10,10)   — m2 i→j correction
  * bm2ji   (10,10,10)   — m2 j-from-i total correction

The 3D tables are filled in Fortran with the innermost varying index
``ibeta = 1..10``; we store them with shape ``(i, j, k)`` matching the
call-site indexing ``bm0ij(n1, n2n, n2a)``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "mam4-original-src-code" / "e3sm_src" / "modal_aero_coag.F90"
OUT = REPO_ROOT / "mam4_jax" / "_coag_tables.npz"

NUM_RE = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
DATA_3D_RE = re.compile(
    r"^\s*data\s*\(\s*(\w+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*ibeta\s*\)"
)
DATA_1D_RE = re.compile(r"^\s*data\s+(\w+)\s*/")


def _read_numbers_until_slash(lines: list[str], start: int) -> tuple[list[float], int]:
    """Collect numeric literals from `lines[start]` onward until the
    closing `/` is seen. Returns (values, index_of_line_with_slash)."""
    vals: list[float] = []
    i = start
    while i < len(lines):
        line = lines[i]
        vals.extend(float(s) for s in NUM_RE.findall(line))
        if "/" in line.split("!", 1)[0] and i > start:
            return vals, i
        if "/" in line.split("!", 1)[0] and "/" in line.rsplit("/", 1)[0]:
            return vals, i
        if line.rstrip().endswith("/"):
            return vals, i
        if line.count("/") >= 2 and i == start:
            return vals, i
        i += 1
    raise RuntimeError(f"unterminated data block starting at line {start+1}")


def extract() -> dict[str, np.ndarray]:
    lines = SRC.read_text().splitlines()

    bm0     = np.zeros(10, dtype=np.float64)
    bm2ii   = np.zeros(10, dtype=np.float64)
    bm2iitt = np.zeros(10, dtype=np.float64)
    bm0ij   = np.zeros((10, 10, 10), dtype=np.float64)
    bm3i    = np.zeros((10, 10, 10), dtype=np.float64)
    bm2ij   = np.zeros((10, 10, 10), dtype=np.float64)
    bm2ji   = np.zeros((10, 10, 10), dtype=np.float64)

    onedim = {"bm0": bm0, "bm2ii": bm2ii, "bm2iitt": bm2iitt}
    threedim = {"bm0ij": bm0ij, "bm3i": bm3i, "bm2ij": bm2ij, "bm2ji": bm2ji}

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m3 = DATA_3D_RE.match(line)
        if m3:
            name, i_idx, j_idx = m3.group(1), int(m3.group(2)), int(m3.group(3))
            if name in threedim:
                vals, end = _read_numbers_until_slash(lines, i)
                # NUM_RE requires a decimal point, so the integer Fortran
                # indices `(1, 1, ibeta), ibeta = 1,10` are skipped — vals
                # holds exactly the 10 ibeta entries.
                if len(vals) != 10:
                    raise RuntimeError(
                        f"{name}({i_idx},{j_idx},*): got {len(vals)} floats"
                    )
                threedim[name][i_idx-1, j_idx-1, :] = vals
                i = end + 1
                continue
        m1 = DATA_1D_RE.match(line)
        if m1:
            name = m1.group(1)
            if name in onedim:
                vals, end = _read_numbers_until_slash(lines, i)
                if len(vals) != 10:
                    raise RuntimeError(f"{name}: got {len(vals)} floats")
                onedim[name][:] = vals
                i = end + 1
                continue
        i += 1

    return {
        "bm0": bm0, "bm2ii": bm2ii, "bm2iitt": bm2iitt,
        "bm0ij": bm0ij, "bm3i": bm3i,
        "bm2ij": bm2ij, "bm2ji": bm2ji,
    }


def main() -> None:
    tables = extract()
    # Sanity: every entry should have been filled (no all-zero rows in a
    # correction-factor table — they're all in [0, 1]).
    for name, arr in tables.items():
        if arr.size != arr[arr > 0].size + arr[arr == 0].size:
            raise AssertionError(f"{name}: unexpected NaN/Inf")
        if arr.ndim == 3:
            zero_rows = np.where(np.all(arr == 0, axis=2))
            if zero_rows[0].size:
                raise AssertionError(
                    f"{name}: empty rows at {list(zip(*zero_rows))}"
                )
        elif arr.ndim == 1 and np.all(arr == 0):
            raise AssertionError(f"{name}: all zero")
    np.savez(OUT, **tables)
    print(f"wrote {OUT}")
    for k, v in tables.items():
        print(f"  {k:8s}  shape={v.shape}  min={v.min():.6f}  max={v.max():.6f}")


if __name__ == "__main__":
    main()
