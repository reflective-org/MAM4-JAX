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

Usage:
    python scripts/capture_reference.py
    python scripts/capture_reference.py --mode instrumented [--nstep 1]
    python scripts/capture_reference.py --mode polysvp
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
POLYSVP_OUT_DIR = REPO_ROOT / "tests" / "reference" / "polysvp"
POLYSVP_EXE = RUN_DIR / "polysvp_driver.exe"

TOTAL_DURATION_S = 1800
NSTEP_SWEEP: tuple[int, ...] = (1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800)

DUMP_TAGS: tuple[str, ...] = (
    "calcsize_before", "calcsize_after",
    "wateruptake_before", "wateruptake_after",
    "amicphys_before", "amicphys_after",
)


NAMELIST_TEMPLATE = dedent("""\
    &time_input
    mam_dt    = {dt},
    mam_nstep = {nstep},
    /
    &cntl_input
    mdo_gaschem    = 0,
    mdo_gasaerexch = 1,
    mdo_rename     = 1,
    mdo_newnuc     = 1,
    mdo_coag       = 1,
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


def ensure_built(instrumented: bool = False, polysvp: bool = False) -> None:
    """Build the executable. Always rebuilds — the build flag determines
    whether the previous binary is the right flavour."""
    cmd = [str(BUILD_SCRIPT)]
    if instrumented:
        cmd.append("--instrumented")
    if polysvp:
        cmd.append("--polysvp")
    flavours = []
    if instrumented: flavours.append("instrumented")
    if polysvp:      flavours.append("polysvp")
    flavours = flavours or ["baseline"]
    print(f"[capture_reference] building {'+'.join(flavours)} executable(s) ...")
    subprocess.run(cmd, check=True)


def write_namelist(dt: int, nstep: int) -> None:
    (RUN_DIR / "namelist").write_text(NAMELIST_TEMPLATE.format(dt=dt, nstep=nstep))


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


def run_instrumented(nstep: int) -> list[Path]:
    dt = TOTAL_DURATION_S // nstep
    PER_PROCESS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe any prior dumps so we never mix runs.
    for stale in RUN_DIR.glob("mam4_dump_*.bin"):
        stale.unlink()

    write_namelist(dt, nstep)
    print(f"[capture_reference] instrumented dt={dt}s nstep={nstep} ...", flush=True)
    subprocess.run(["./mam_box_test.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)

    written: list[Path] = []
    for tag in DUMP_TAGS:
        bin_path = RUN_DIR / f"mam4_dump_{tag}.bin"
        if not bin_path.is_file():
            raise RuntimeError(f"expected dump missing: {bin_path}")
        arrays = _read_dump(bin_path)
        npz_path = PER_PROCESS_OUT_DIR / f"{tag}.npz"
        np.savez(npz_path, **arrays)
        written.append(npz_path)
    return written


# ----- polysvp mode ---------------------------------------------------------

def run_polysvp() -> list[Path]:
    POLYSVP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[capture_reference] running polysvp driver ...", flush=True)
    subprocess.run(["./polysvp_driver.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)

    text = (RUN_DIR / "polysvp_reference.txt").read_text()
    # Skip comment lines (start with '#') and parse the three-column table.
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] != 3:
        raise RuntimeError(f"unexpected polysvp_reference.txt shape: {arr.shape}")

    out = POLYSVP_OUT_DIR / "reference.npz"
    np.savez(out, T=arr[:, 0], esat_water=arr[:, 1], esat_ice=arr[:, 2])
    return [out]


# ----- entry point ----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("sweep", "instrumented", "polysvp"),
                    default="sweep")
    ap.add_argument("--nstep", type=int, default=1,
                    help="instrumented mode: number of timesteps over 1800 s (default 1)")
    args = ap.parse_args()

    if args.mode == "sweep":
        ensure_built()
        SWEEP_OUT_DIR.mkdir(parents=True, exist_ok=True)
        written = [run_one_baseline(n) for n in NSTEP_SWEEP]
        out_root = SWEEP_OUT_DIR
    elif args.mode == "instrumented":
        ensure_built(instrumented=True)
        if args.nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={args.nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(args.nstep)
        out_root = PER_PROCESS_OUT_DIR
    else:  # polysvp
        ensure_built(polysvp=True)
        written = run_polysvp()
        out_root = POLYSVP_OUT_DIR

    print(f"\n[capture_reference] {len(written)} file(s) written under {out_root}")
    for p in written:
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
