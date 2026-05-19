"""Capture Fortran reference outputs by running the 12-point convergence sweep.

For each (dt, nstep) in the canonical sweep, writes a namelist matching
the Fortran driver's expectations, runs the executable, and copies
mam_output.nc into ``tests/reference/sweep/mam_dt<DT>_ndt<N>.nc``.

The executable is rebuilt automatically (via ``scripts/build_reference.sh``)
if it is not present.

Usage:
    python scripts/capture_reference.py

Per-process instrumented capture (M2 phase 3) will be added behind a
``--mode instrumented`` flag in a follow-up commit.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "mam4-original-src-code"
RUN_DIR = SRC_DIR / "run"
EXE = RUN_DIR / "mam_box_test.exe"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_reference.sh"
OUT_DIR = REPO_ROOT / "tests" / "reference" / "sweep"

# Canonical 1800 s convergence sweep from the upstream run_test.csh.
TOTAL_DURATION_S = 1800
NSTEP_SWEEP: tuple[int, ...] = (1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800)

# Namelist matches the Fortran driver's expectations
# (test_drivers/driver.F90 + box_model_utils/rad_constituents.F90 :211).
# Values mirror run_test.csh and the rad_constituents MAM4-MOM defaults
# (dgnum/sigmag in metres).
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


def ensure_built() -> None:
    if EXE.is_file() and os.access(EXE, os.X_OK):
        return
    print(f"[capture_reference] {EXE.name} missing; building via {BUILD_SCRIPT.name} ...")
    subprocess.run([str(BUILD_SCRIPT)], check=True)


def run_one(nstep: int) -> Path:
    dt = TOTAL_DURATION_S // nstep
    (RUN_DIR / "namelist").write_text(NAMELIST_TEMPLATE.format(dt=dt, nstep=nstep))
    print(f"[capture_reference] dt={dt:>4}s nstep={nstep:<5} ...", flush=True)
    subprocess.run(["./mam_box_test.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)
    dest = OUT_DIR / f"mam_dt{dt}_ndt{nstep}.nc"
    shutil.copy2(RUN_DIR / "mam_output.nc", dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    ensure_built()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for n in NSTEP_SWEEP:
        written.append(run_one(n))

    print(f"\n[capture_reference] {len(written)} file(s) written under {OUT_DIR}")
    for p in written:
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
