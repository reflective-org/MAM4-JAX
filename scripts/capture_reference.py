"""Capture Fortran reference outputs.

Three modes:

* ``--mode sweep`` (default): build the baseline (non-instrumented) executable
  and run the 12-point convergence sweep. Each (dt, nstep) writes a NetCDF
  to ``tests/reference/sweep/mam_dt<DT>_ndt<N>.nc``.

* ``--mode instrumented``: build the executable with the
  ``scripts/patches/`` overlay applied, run the executable for ``--nstep``
  timesteps, then convert the six ``mam4_dump_*.bin`` files into
  ``tests/reference/per_process/<process>_<phase>.npz``. Each ``.npz``
  bundles arrays ``istep``, ``q``, ``qqcw``, ``dgncur_a``, ``dgncur_awet``,
  ``qaerwat``, ``wetdens``. Schema: ``tests/reference/SCHEMA.md``.

* ``--mode polysvp``: build the standalone polysvp driver
  (``scripts/reference_drivers/polysvp_driver.F90``), run it across a
  170 K – 320 K temperature sweep (1501 points), parse the text output, and
  archive as ``tests/reference/polysvp/reference.npz`` with arrays
  ``T``, ``esat_water``, ``esat_ice`` (all float64).

* ``--mode qsat``: build the qsat driver
  (``scripts/reference_drivers/qsat_driver.F90``), sweep over a (T, p)
  grid (301 T × 5 p = 1505 points), and archive as
  ``tests/reference/qsat/reference.npz`` with arrays ``T``, ``p``,
  ``qs_water``, ``qs_ice``. Driver depends on ``gestbl()`` having been
  called to populate ``wv_saturation``'s module-level state — driver
  calls it with canonical box-model constants.

* ``--mode makoh``: applies ``scripts/patches/expose_internals.patch`` to
  make ``makoh_cubic`` and ``makoh_quartic`` public, builds the makoh
  driver (``scripts/reference_drivers/makoh_driver.F90``), runs it on a
  small batch of test polynomial coefficients, and archives complex
  roots to ``tests/reference/makoh/reference.npz`` with arrays
  ``cubic_inputs``, ``cubic_roots``, ``quartic_inputs``, ``quartic_roots``.

* ``--mode kohler``: applies the same overlay (which also exposes
  ``modal_aero_kohler``), builds the kohler driver
  (``scripts/reference_drivers/kohler_driver.F90``), sweeps a
  ``(rdry, hygro, s)`` grid (7 × 4 × 6 = 168 points covering all four
  branches), and archives to ``tests/reference/kohler/reference.npz``
  with arrays ``rdry_in``, ``hygro``, ``s``, ``rwet``.

* ``--mode instrumented-no-aitacc``: same as ``instrumented`` but also
  applies ``disable_aitacc_transfer.patch`` (calcsize is called with
  ``do_aitacc_transfer_in=.false.``). Writes the per-process dumps to
  ``tests/reference/per_process_no_aitacc/`` rather than the default
  ``per_process/`` so the two captures coexist. Defaults to ``--nstep 60``
  because calcsize is essentially trivial at ``nstep=1`` (per-mode
  evolution needs multiple steps to be meaningful).

* ``--mode instrumented-amicphys-off``: same as ``instrumented`` but
  writes the namelist with ``mdo_gasaerexch=0, mdo_rename=0,
  mdo_newnuc=0, mdo_coag=0`` (all four amicphys sub-processes disabled).
  Writes to ``tests/reference/per_process_amicphys_off/``. Defaults to
  ``--nstep 60``. Used to validate the M3.6 PR-A amicphys orchestration
  shell (state passthrough when no physics runs).

Usage:
    python scripts/capture_reference.py
    python scripts/capture_reference.py --mode instrumented [--nstep 1]
    python scripts/capture_reference.py --mode instrumented-no-aitacc [--nstep 60]
    python scripts/capture_reference.py --mode polysvp
    python scripts/capture_reference.py --mode qsat
    python scripts/capture_reference.py --mode makoh
    python scripts/capture_reference.py --mode kohler
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "mam4-original-src-code"
RUN_DIR = SRC_DIR / "run"
EXE = RUN_DIR / "mam_box_test.exe"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_reference.sh"

SWEEP_OUT_DIR = REPO_ROOT / "tests" / "reference" / "sweep"
PER_PROCESS_OUT_DIR = REPO_ROOT / "tests" / "reference" / "per_process"
PER_PROCESS_NO_AITACC_OUT_DIR = REPO_ROOT / "tests" / "reference" / "per_process_no_aitacc"
PER_PROCESS_AMICPHYS_OFF_OUT_DIR = REPO_ROOT / "tests" / "reference" / "per_process_amicphys_off"
POLYSVP_OUT_DIR = REPO_ROOT / "tests" / "reference" / "polysvp"
POLYSVP_EXE = RUN_DIR / "polysvp_driver.exe"
QSAT_OUT_DIR = REPO_ROOT / "tests" / "reference" / "qsat"
QSAT_EXE = RUN_DIR / "qsat_driver.exe"
MAKOH_OUT_DIR = REPO_ROOT / "tests" / "reference" / "makoh"
MAKOH_EXE = RUN_DIR / "makoh_driver.exe"
KOHLER_OUT_DIR = REPO_ROOT / "tests" / "reference" / "kohler"
KOHLER_EXE = RUN_DIR / "kohler_driver.exe"
INDICES_OUT_DIR = REPO_ROOT / "tests" / "reference" / "indices"

TOTAL_DURATION_S = 1800
NSTEP_SWEEP: tuple[int, ...] = (1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800)

DUMP_TAGS: tuple[str, ...] = (
    "calcsize_before", "calcsize_after",
    "wateruptake_before", "wateruptake_after",
    "amicphys_before", "amicphys_after",
)

# Tags with the per-(col, level, subarea) rename schema (different from
# DUMP_TAGS' outer pcnst-tracer layout). Captured by the rename_hook patch
# inside mam_amicphys_1subarea_clear. Absent from runs where mdo_rename=0.
RENAME_DUMP_TAGS: tuple[str, ...] = ("rename_before", "rename_after")


NAMELIST_TEMPLATE = dedent("""\
    &time_input
    mam_dt    = {dt},
    mam_nstep = {nstep},
    /
    &cntl_input
    mdo_gaschem    = 0,
    mdo_gasaerexch = {mdo_gasaerexch},
    mdo_rename     = {mdo_rename},
    mdo_newnuc     = {mdo_newnuc},
    mdo_coag       = {mdo_coag},
    /
    &met_input
    temp    = 273.,
    press   = 1.e5,
    RH_CLEA = 0.9,
    /
    &chem_input
    numc1=1.e8, numc2=1.e9, numc3=1.e5, numc4=2.e8,
    mfso41=0.3, mfpom1=0., mfsoa1=0.3, mfbc1=0., mfdst1=0., mfncl1=0.4,
    mfso42=0.3, mfsoa2=0.3, mfncl2=0.4,
    mfdst3=0., mfncl3=0.4, mfso43=0.3, mfbc3=0., mfpom3=0., mfsoa3=0.3,
    mfpom4=0., mfbc4=1.,
    qso2=1.e-4, qh2so4=1.e-13, qsoag=5.e-10,
    /
    &size_parameters
    dgnum1=0.1100e-6, dgnum2=0.0260e-6, dgnum3=2.000e-6, dgnum4=0.050e-6,
    sigmag1=1.800, sigmag2=1.600, sigmag3=1.800, sigmag4=1.600,
    /
""")


def ensure_built(instrumented: bool = False, polysvp: bool = False,
                 qsat: bool = False, makoh: bool = False,
                 kohler: bool = False, no_aitacc_transfer: bool = False) -> None:
    """Build the executable. Always rebuilds — the build flag determines
    whether the previous binary is the right flavour."""
    cmd = [str(BUILD_SCRIPT)]
    if instrumented:        cmd.append("--instrumented")
    if polysvp:             cmd.append("--polysvp")
    if qsat:                cmd.append("--qsat")
    if makoh:               cmd.append("--makoh")
    if kohler:              cmd.append("--kohler")
    if no_aitacc_transfer:  cmd.append("--no-aitacc-transfer")
    flavours = []
    if instrumented:        flavours.append("instrumented")
    if polysvp:             flavours.append("polysvp")
    if qsat:                flavours.append("qsat")
    if makoh:               flavours.append("makoh")
    if kohler:              flavours.append("kohler")
    if no_aitacc_transfer:  flavours.append("no-aitacc-transfer")
    flavours = flavours or ["baseline"]
    print(f"[capture_reference] building {'+'.join(flavours)} executable(s) ...")
    subprocess.run(cmd, check=True)


def write_namelist(dt: int, nstep: int, *,
                   mdo_gasaerexch: int = 1, mdo_rename: int = 1,
                   mdo_newnuc: int = 1, mdo_coag: int = 1) -> None:
    """Write the run/namelist file.

    The four ``mdo_*`` knobs default to the canonical all-on values from
    the upstream ``run_test.csh``. Set them to 0 individually for
    single-toggle captures, or all to 0 for an "amicphys-off" capture
    (every microphysical sub-process bypassed).
    """
    (RUN_DIR / "namelist").write_text(NAMELIST_TEMPLATE.format(
        dt=dt, nstep=nstep,
        mdo_gasaerexch=mdo_gasaerexch, mdo_rename=mdo_rename,
        mdo_newnuc=mdo_newnuc, mdo_coag=mdo_coag,
    ))


def run_one_baseline(nstep: int) -> Path:
    dt = TOTAL_DURATION_S // nstep
    write_namelist(dt, nstep)
    print(f"[capture_reference] sweep dt={dt:>4}s nstep={nstep:<5} ...", flush=True)
    subprocess.run(["./mam_box_test.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)
    dest = SWEEP_OUT_DIR / f"mam_dt{dt}_ndt{nstep}.nc"
    shutil.copy2(RUN_DIR / "mam_output.nc", dest)
    return dest


# ----- instrumented mode ----------------------------------------------------

# Binary record layout written by scripts/patches/mam4_dump_state.F90:
#   int32  : istep
#   int32  : ncol, pver, pcnst, ntot_amode      (4 values)
#   float64: q          (ncol*pver*pcnst)
#   float64: qqcw       (ncol*pver*pcnst)
#   float64: dgncur_a   (ncol*pver*ntot_amode)
#   float64: dgncur_awet(ncol*pver*ntot_amode)
#   float64: qaerwat    (ncol*pver*ntot_amode)
#   float64: wetdens    (ncol*pver*ntot_amode)


def _read_dump(path: Path) -> dict[str, np.ndarray]:
    """Parse one mam4_dump_<tag>.bin into per-array stacks across timesteps."""
    raw = path.read_bytes()
    pos = 0
    istep_list: list[int] = []
    q_list: list[np.ndarray] = []
    qqcw_list: list[np.ndarray] = []
    dgncur_a_list: list[np.ndarray] = []
    dgncur_awet_list: list[np.ndarray] = []
    qaerwat_list: list[np.ndarray] = []
    wetdens_list: list[np.ndarray] = []

    while pos < len(raw):
        istep = int(np.frombuffer(raw, dtype=np.int32, count=1, offset=pos)[0]); pos += 4
        hdr = np.frombuffer(raw, dtype=np.int32, count=4, offset=pos); pos += 16
        ncol, pver, pcnst, ntot_amode = (int(x) for x in hdr)

        n_tracer = ncol * pver * pcnst
        n_mode   = ncol * pver * ntot_amode

        def take(n: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal pos
            arr = np.frombuffer(raw, dtype=np.float64, count=n, offset=pos).reshape(shape).copy()
            pos += n * 8
            return arr

        q_list.append(take(n_tracer, (ncol, pver, pcnst)))
        qqcw_list.append(take(n_tracer, (ncol, pver, pcnst)))
        dgncur_a_list.append(take(n_mode, (ncol, pver, ntot_amode)))
        dgncur_awet_list.append(take(n_mode, (ncol, pver, ntot_amode)))
        qaerwat_list.append(take(n_mode, (ncol, pver, ntot_amode)))
        wetdens_list.append(take(n_mode, (ncol, pver, ntot_amode)))
        istep_list.append(istep)

    return {
        "istep":       np.asarray(istep_list, dtype=np.int32),
        "q":           np.stack(q_list),
        "qqcw":        np.stack(qqcw_list),
        "dgncur_a":    np.stack(dgncur_a_list),
        "dgncur_awet": np.stack(dgncur_awet_list),
        "qaerwat":     np.stack(qaerwat_list),
        "wetdens":     np.stack(wetdens_list),
    }


def _read_rename_dump(path: Path) -> dict[str, np.ndarray]:
    """Parse one mam4_dump_rename_{before,after}.bin into per-call stacks.

    Binary record layout written by mam4_dump_state::dump_rename_snapshot:
      int32 : istep, i, k, jsub                       (4 values)
      int32 : max_mode, max_aer                       (2 values)
      int32 : mtoo_renamexf(max_mode)
      f64   : qnum_cur(max_mode)
      f64   : qaer_cur(max_aer, max_mode)
      f64   : qaer_delsub_grow4rnam(max_aer, max_mode)
      f64   : qwtr_cur(max_mode)
      f64   : fac_m2v_aer(max_aer)
    """
    raw = path.read_bytes()
    pos = 0
    istep_list: list[int] = []
    i_list: list[int] = []
    k_list: list[int] = []
    jsub_list: list[int] = []
    mtoo_list: list[np.ndarray] = []
    qnum_list: list[np.ndarray] = []
    qaer_list: list[np.ndarray] = []
    qdel_list: list[np.ndarray] = []
    qwtr_list: list[np.ndarray] = []
    fac_list:  list[np.ndarray] = []

    while pos < len(raw):
        hdr1 = np.frombuffer(raw, dtype=np.int32, count=4, offset=pos); pos += 16
        istep, i, k, jsub = (int(x) for x in hdr1)
        hdr2 = np.frombuffer(raw, dtype=np.int32, count=2, offset=pos); pos += 8
        max_mode, max_aer = (int(x) for x in hdr2)

        mtoo = np.frombuffer(raw, dtype=np.int32, count=max_mode,
                             offset=pos).copy(); pos += max_mode * 4

        def take(n: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal pos
            arr = np.frombuffer(raw, dtype=np.float64, count=n,
                                offset=pos).reshape(shape, order="F").copy()
            pos += n * 8
            return arr

        qnum_list.append(take(max_mode, (max_mode,)))
        qaer_list.append(take(max_aer * max_mode, (max_aer, max_mode)))
        qdel_list.append(take(max_aer * max_mode, (max_aer, max_mode)))
        qwtr_list.append(take(max_mode, (max_mode,)))
        fac_list.append(take(max_aer, (max_aer,)))
        mtoo_list.append(mtoo)
        istep_list.append(istep); i_list.append(i); k_list.append(k); jsub_list.append(jsub)

    return {
        "istep":                  np.asarray(istep_list, dtype=np.int32),
        "i":                      np.asarray(i_list,     dtype=np.int32),
        "k":                      np.asarray(k_list,     dtype=np.int32),
        "jsub":                   np.asarray(jsub_list,  dtype=np.int32),
        "mtoo_renamexf":          np.stack(mtoo_list),
        "qnum_cur":               np.stack(qnum_list),
        "qaer_cur":               np.stack(qaer_list),
        "qaer_delsub_grow4rnam":  np.stack(qdel_list),
        "qwtr_cur":               np.stack(qwtr_list),
        "fac_m2v_aer":            np.stack(fac_list),
    }


def _read_indices(path: Path) -> dict[str, np.ndarray]:
    """Parse mam4_indices.txt into a dict of numpy arrays.

    The text format is human-readable section-marked output written by
    mam4_dump_state::dump_indices. 2D arrays appear as one line per mode
    with slots listed within the line, so Python sees them as
    (ntot_amode, maxd_aspectype) (mode-first), which is the transpose of
    the Fortran declaration (slot, mode).

    Integer index values are converted to 0-based here; Fortran writes
    them 1-based. The sentinel value 0 (empty slot) becomes -1 in 0-based
    form, matching the existing IndexTables sentinel.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("%"):
            current = s.lstrip("%").strip().split()[0]
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(s)

    def ints(name: str) -> np.ndarray:
        toks = [int(t) for ln in sections[name] for t in ln.split()]
        return np.asarray(toks, dtype=np.int32)

    def ints_2d(name: str, nrows: int, ncols: int) -> np.ndarray:
        # One row per line; modes are rows, slots are columns.
        rows = sections[name]
        assert len(rows) == nrows, f"{name}: expected {nrows} rows, got {len(rows)}"
        out = np.full((nrows, ncols), 0, dtype=np.int32)
        for r, line in enumerate(rows):
            vals = [int(t) for t in line.split()]
            out[r, :len(vals)] = vals
        return out

    def strings(name: str) -> list[str]:
        return list(sections[name])

    ntot_amode     = ints("ntot_amode").item()
    ntot_aspectype = ints("ntot_aspectype").item()
    maxd_aspectype = ints("maxd_aspectype").item()

    # Convert 1-based pcnst indices to 0-based; preserve 0 (empty) → -1.
    def to_0based(arr: np.ndarray) -> np.ndarray:
        return np.where(arr == 0, -1, arr - 1).astype(np.int32)

    return {
        "ntot_amode":     np.int32(ntot_amode),
        "ntot_aspectype": np.int32(ntot_aspectype),
        "maxd_aspectype": np.int32(maxd_aspectype),
        "numptr_amode":     to_0based(ints("numptr_amode")),
        "numptrcw_amode":   to_0based(ints("numptrcw_amode")),
        "nspec_amode":      ints("nspec_amode"),
        "lspectype_amode":  to_0based(ints_2d("lspectype_amode",
                                              ntot_amode, maxd_aspectype)),
        "lmassptr_amode":   to_0based(ints_2d("lmassptr_amode",
                                              ntot_amode, maxd_aspectype)),
        "lmassptrcw_amode": to_0based(ints_2d("lmassptrcw_amode",
                                              ntot_amode, maxd_aspectype)),
        "modename_amode":   np.asarray(strings("modename_amode")),
        "specname_amode":   np.asarray(strings("specname_amode")),
    }


def run_instrumented(nstep: int, no_aitacc_transfer: bool = False,
                     amicphys_off: bool = False) -> list[Path]:
    dt = TOTAL_DURATION_S // nstep
    if amicphys_off:
        out_dir = PER_PROCESS_AMICPHYS_OFF_OUT_DIR
        flavour = "instrumented-amicphys-off"
    elif no_aitacc_transfer:
        out_dir = PER_PROCESS_NO_AITACC_OUT_DIR
        flavour = "instrumented-no-aitacc"
    else:
        out_dir = PER_PROCESS_OUT_DIR
        flavour = "instrumented"
    out_dir.mkdir(parents=True, exist_ok=True)
    INDICES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe any prior dumps so we never mix runs.
    for stale in list(RUN_DIR.glob("mam4_dump_*.bin")) + [RUN_DIR / "mam4_indices.txt"]:
        if stale.exists():
            stale.unlink()

    if amicphys_off:
        write_namelist(dt, nstep,
                       mdo_gasaerexch=0, mdo_rename=0,
                       mdo_newnuc=0, mdo_coag=0)
    else:
        write_namelist(dt, nstep)
    print(f"[capture_reference] {flavour} dt={dt}s nstep={nstep} ...", flush=True)
    subprocess.run(["./mam_box_test.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)

    written: list[Path] = []

    # Index tables (written once at init, before the time loop).
    # Only the default (non-no-aitacc, non-amicphys-off) run writes the
    # canonical indices.
    if not no_aitacc_transfer and not amicphys_off:
        indices_txt = RUN_DIR / "mam4_indices.txt"
        if not indices_txt.is_file():
            raise RuntimeError(f"expected indices dump missing: {indices_txt}")
        indices_npz = INDICES_OUT_DIR / "reference.npz"
        np.savez(indices_npz, **_read_indices(indices_txt))
        written.append(indices_npz)

    # Per-process tracer snapshots (one record per istep, across the loop).
    for tag in DUMP_TAGS:
        bin_path = RUN_DIR / f"mam4_dump_{tag}.bin"
        if not bin_path.is_file():
            raise RuntimeError(f"expected dump missing: {bin_path}")
        arrays = _read_dump(bin_path)
        npz_path = out_dir / f"{tag}.npz"
        np.savez(npz_path, **arrays)
        written.append(npz_path)

    # Rename hook dumps (skipped when mdo_rename=0; the hook lives inside
    # do_rename_if_block30 so the .bin files are never created in that case).
    for tag in RENAME_DUMP_TAGS:
        bin_path = RUN_DIR / f"mam4_dump_{tag}.bin"
        if not bin_path.is_file():
            continue
        arrays = _read_rename_dump(bin_path)
        npz_path = out_dir / f"{tag}.npz"
        np.savez(npz_path, **arrays)
        written.append(npz_path)

    return written


# ----- polysvp mode ---------------------------------------------------------

def _read_text_table(path: Path, n_cols: int) -> np.ndarray:
    text = path.read_text()
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("%"):
            continue
        rows.append([float(x) for x in s.split()])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] != n_cols:
        raise RuntimeError(f"unexpected table shape at {path}: {arr.shape}")
    return arr


def run_polysvp() -> list[Path]:
    POLYSVP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[capture_reference] running polysvp driver ...", flush=True)
    subprocess.run(["./polysvp_driver.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)

    arr = _read_text_table(RUN_DIR / "polysvp_reference.txt", n_cols=3)
    out = POLYSVP_OUT_DIR / "reference.npz"
    np.savez(out, T=arr[:, 0], esat_water=arr[:, 1], esat_ice=arr[:, 2])
    return [out]


def run_qsat() -> list[Path]:
    QSAT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[capture_reference] running qsat driver ...", flush=True)
    subprocess.run(["./qsat_driver.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)

    arr = _read_text_table(RUN_DIR / "qsat_reference.txt", n_cols=4)
    out = QSAT_OUT_DIR / "reference.npz"
    np.savez(out, T=arr[:, 0], p=arr[:, 1], qs_water=arr[:, 2], qs_ice=arr[:, 3])
    return [out]


def _read_makoh(path: Path) -> dict[str, np.ndarray]:
    """Parse makoh_reference.txt into a dict of numpy arrays.

    The text format is a section-marked dump produced by makoh_driver.F90.
    Each section starts with '%' followed by all-numeric rows.
    """
    sections: dict[str, list[list[float]]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("%"):
            current = s.lstrip("%").strip().split()[0]
            sections[current] = []
            continue
        if current is not None:
            sections[current].append([float(t) for t in s.split()])

    def arr(name: str) -> np.ndarray:
        return np.asarray(sections[name], dtype=np.float64)

    # cubic_roots is laid out as ncub*3 rows of (real, imag); reassemble
    # into a complex (ncub, 3) array.
    def complex_roots(name: str, n_roots: int) -> np.ndarray:
        ri = arr(name)  # shape (n_cases * n_roots, 2)
        cmplx = ri[:, 0] + 1j * ri[:, 1]
        return cmplx.reshape(-1, n_roots)

    return {
        "cubic_inputs":   arr("cubic_inputs"),
        "cubic_roots":    complex_roots("cubic_roots", 3),
        "quartic_inputs": arr("quartic_inputs"),
        "quartic_roots":  complex_roots("quartic_roots", 4),
    }


def run_makoh() -> list[Path]:
    MAKOH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[capture_reference] running makoh driver ...", flush=True)
    subprocess.run(["./makoh_driver.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)

    arrays = _read_makoh(RUN_DIR / "makoh_reference.txt")
    out = MAKOH_OUT_DIR / "reference.npz"
    np.savez(out, **arrays)
    return [out]


def run_kohler() -> list[Path]:
    KOHLER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[capture_reference] running kohler driver ...", flush=True)
    subprocess.run(["./kohler_driver.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)

    arr = _read_text_table(RUN_DIR / "kohler_reference.txt", n_cols=4)
    out = KOHLER_OUT_DIR / "reference.npz"
    np.savez(out,
             rdry_in=arr[:, 0], hygro=arr[:, 1], s=arr[:, 2], rwet=arr[:, 3])
    return [out]


# ----- entry point ----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode",
        choices=("sweep", "instrumented", "instrumented-no-aitacc",
                 "instrumented-amicphys-off",
                 "polysvp", "qsat", "makoh", "kohler"),
        default="sweep",
    )
    ap.add_argument(
        "--nstep", type=int, default=None,
        help=(
            "instrumented mode: number of timesteps over 1800 s "
            "(default: 1 for `instrumented`, 60 for `instrumented-no-aitacc`)"
        ),
    )
    args = ap.parse_args()

    if args.mode == "sweep":
        ensure_built()
        SWEEP_OUT_DIR.mkdir(parents=True, exist_ok=True)
        written = [run_one_baseline(n) for n in NSTEP_SWEEP]
        out_root = SWEEP_OUT_DIR
    elif args.mode == "instrumented":
        ensure_built(instrumented=True)
        nstep = args.nstep if args.nstep is not None else 1
        if nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(nstep)
        out_root = PER_PROCESS_OUT_DIR
    elif args.mode == "instrumented-no-aitacc":
        ensure_built(instrumented=True, no_aitacc_transfer=True)
        nstep = args.nstep if args.nstep is not None else 60
        if nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(nstep, no_aitacc_transfer=True)
        out_root = PER_PROCESS_NO_AITACC_OUT_DIR
    elif args.mode == "instrumented-amicphys-off":
        # Uses the default instrumented build (no patches besides the
        # existing driver_instrumentation overlay) — the all-mdo-off
        # control flow is selected via the namelist, not a Fortran patch.
        ensure_built(instrumented=True)
        nstep = args.nstep if args.nstep is not None else 60
        if nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(nstep, amicphys_off=True)
        out_root = PER_PROCESS_AMICPHYS_OFF_OUT_DIR
    elif args.mode == "polysvp":
        ensure_built(polysvp=True)
        written = run_polysvp()
        out_root = POLYSVP_OUT_DIR
    elif args.mode == "qsat":
        ensure_built(qsat=True)
        written = run_qsat()
        out_root = QSAT_OUT_DIR
    elif args.mode == "makoh":
        ensure_built(makoh=True)
        written = run_makoh()
        out_root = MAKOH_OUT_DIR
    else:  # kohler
        ensure_built(kohler=True)
        written = run_kohler()
        out_root = KOHLER_OUT_DIR

    print(f"\n[capture_reference] {len(written)} file(s) written under {out_root}")
    for p in written:
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
