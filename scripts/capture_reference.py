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
PER_PROCESS_RENAME_ONLY_OUT_DIR  = REPO_ROOT / "tests" / "reference" / "per_process_rename_only"
PER_PROCESS_GASAEREXCH_ONLY_OUT_DIR = (
    REPO_ROOT / "tests" / "reference" / "per_process_gasaerexch_only"
)
PER_PROCESS_GASAEREXCH_OUT_DIR = (
    REPO_ROOT / "tests" / "reference" / "per_process_gasaerexch"
)
PER_PROCESS_GASAEREXCH_AND_NEWNUC_OUT_DIR = (
    REPO_ROOT / "tests" / "reference" / "per_process_gasaerexch_and_newnuc"
)
PER_PROCESS_COAG_OUT_DIR = REPO_ROOT / "tests" / "reference" / "per_process_coag"
POLYSVP_OUT_DIR = REPO_ROOT / "tests" / "reference" / "polysvp"
POLYSVP_EXE = RUN_DIR / "polysvp_driver.exe"
QSAT_OUT_DIR = REPO_ROOT / "tests" / "reference" / "qsat"
QSAT_EXE = RUN_DIR / "qsat_driver.exe"
MAKOH_OUT_DIR = REPO_ROOT / "tests" / "reference" / "makoh"
MAKOH_EXE = RUN_DIR / "makoh_driver.exe"
KOHLER_OUT_DIR = REPO_ROOT / "tests" / "reference" / "kohler"
KOHLER_EXE = RUN_DIR / "kohler_driver.exe"
NEWNUC_HELPERS_OUT_DIR = REPO_ROOT / "tests" / "reference" / "newnuc_helpers"
NEWNUC_HELPERS_EXE = RUN_DIR / "newnuc_helpers_driver.exe"
MER07_VEH02_OUT_DIR = REPO_ROOT / "tests" / "reference" / "mer07_veh02"
MER07_VEH02_EXE = RUN_DIR / "mer07_veh02_driver.exe"
COAG_COEFFICIENTS_OUT_DIR = REPO_ROOT / "tests" / "reference" / "coag_coefficients"
COAG_COEFFICIENTS_EXE = RUN_DIR / "coag_coefficients_driver.exe"
INDICES_OUT_DIR = REPO_ROOT / "tests" / "reference" / "indices"

TOTAL_DURATION_S = 1800
NSTEP_SWEEP: tuple[int, ...] = (1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800)

DUMP_TAGS: tuple[str, ...] = (
    "calcsize_before", "calcsize_after",
    "wateruptake_before", "wateruptake_after",
    "amicphys_before", "amicphys_after",
    "amicphys_after_writeback",
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
                 kohler: bool = False, no_aitacc_transfer: bool = False,
                 skip_soaexch: bool = False,
                 skip_pcarbon_aging: bool = False,
                 newnuc_helpers: bool = False,
                 mer07_veh02: bool = False,
                 coag_coefficients: bool = False) -> None:
    """Build the executable. Always rebuilds — the build flag determines
    whether the previous binary is the right flavour."""
    cmd = [str(BUILD_SCRIPT)]
    if instrumented:        cmd.append("--instrumented")
    if polysvp:             cmd.append("--polysvp")
    if qsat:                cmd.append("--qsat")
    if makoh:               cmd.append("--makoh")
    if kohler:              cmd.append("--kohler")
    if newnuc_helpers:      cmd.append("--newnuc-helpers")
    if mer07_veh02:         cmd.append("--mer07-veh02")
    if coag_coefficients:   cmd.append("--coag-coefficients")
    if no_aitacc_transfer:  cmd.append("--no-aitacc-transfer")
    if skip_soaexch:        cmd.append("--skip-soaexch")
    if skip_pcarbon_aging:  cmd.append("--skip-pcarbon-aging")
    flavours = []
    if instrumented:        flavours.append("instrumented")
    if polysvp:             flavours.append("polysvp")
    if qsat:                flavours.append("qsat")
    if makoh:               flavours.append("makoh")
    if kohler:              flavours.append("kohler")
    if newnuc_helpers:      flavours.append("newnuc-helpers")
    if mer07_veh02:         flavours.append("mer07-veh02")
    if coag_coefficients:   flavours.append("coag-coefficients")
    if no_aitacc_transfer:  flavours.append("no-aitacc-transfer")
    if skip_soaexch:        flavours.append("skip-soaexch")
    if skip_pcarbon_aging:  flavours.append("skip-pcarbon-aging")
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


def _read_amicphys_init(path: Path) -> dict[str, np.ndarray]:
    """Parse mam4_amicphys_init.txt (amicphys-internal tables) into a dict.

    Written by the amicphys_init_dump.patch overlay from inside
    modal_aero_amicphys_init, where the module-private lmap/fcvt/name
    tables are in scope. Same '%'-section text layout as mam4_indices.txt.

    Integer index conversion: lmap_* values from Fortran are
    *gas_pcnst-relative* (i.e. inside chemistry's offset). The pcnst
    absolute index is `lmap_X + loffset`. We dump both: the raw value
    (`lmap_*`) and the loffset-adjusted, 0-based, -1-sentinel form
    (`pcnst_lmap_*`) so consumers can pick the level they need.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("%"):
            # The fcvt_num / fcvt_wtr lines pack the value on the same
            # line as the marker; handle that here.
            head = s.lstrip("%").strip()
            tokens = head.split()
            current = tokens[0]
            sections[current] = []
            # Inline value (skip the first token = section name).
            if len(tokens) > 1 and not tokens[1].startswith("("):
                sections[current].append(" ".join(tokens[1:]))
            continue
        if current is not None:
            sections[current].append(s)

    def ints(name: str) -> np.ndarray:
        toks = [int(t) for ln in sections[name] for t in ln.split()]
        return np.asarray(toks, dtype=np.int32)

    def floats(name: str) -> np.ndarray:
        toks = [float(t) for ln in sections[name] for t in ln.split()]
        return np.asarray(toks, dtype=np.float64)

    def scalar_int(name: str) -> int:
        return int(ints(name)[0])

    def scalar_float(name: str) -> float:
        return float(floats(name)[0])

    def ints_2d(name: str, nrows: int, ncols: int) -> np.ndarray:
        rows = sections[name]
        assert len(rows) == nrows, f"{name}: expected {nrows} rows, got {len(rows)}"
        out = np.zeros((nrows, ncols), dtype=np.int32)
        for r, line in enumerate(rows):
            vals = [int(t) for t in line.split()]
            assert len(vals) == ncols, f"{name}[{r}]: expected {ncols} cols, got {len(vals)}"
            out[r, :] = vals
        return out

    loffset  = scalar_int("loffset")
    ngas     = scalar_int("ngas")
    naer     = scalar_int("naer")
    max_gas  = scalar_int("max_gas")
    max_aer  = scalar_int("max_aer")
    ntot_amode = 4   # fixed for MAM4-MOM; could be cross-checked against the other dump

    lmap_gas    = ints("lmap_gas")
    lmap_num    = ints("lmap_num")
    lmap_numcw  = ints("lmap_numcw")
    lmap_aer    = ints_2d("lmap_aer",    ntot_amode, naer)
    lmap_aercw  = ints_2d("lmap_aercw",  ntot_amode, naer)
    fcvt_gas    = floats("fcvt_gas")
    fcvt_aer    = floats("fcvt_aer")
    fcvt_num    = scalar_float("fcvt_num")
    fcvt_wtr    = scalar_float("fcvt_wtr")
    mwdry       = scalar_float("mwdry")
    adv_mass    = floats("adv_mass")     # shape (gas_pcnst,)
    vmdry       = scalar_float("vmdry")
    mw_gas      = floats("mw_gas")       # shape (ngas,)
    vol_molar_gas   = floats("vol_molar_gas")
    accom_coef_gas  = floats("accom_coef_gas")
    npoa        = scalar_int("npoa")
    nsoa        = scalar_int("nsoa")
    iaer_pom    = scalar_int("iaer_pom")
    iaer_soa    = scalar_int("iaer_soa")
    npca        = scalar_int("npca")
    nufi        = scalar_int("nufi")
    mode_aging_optaa  = ints("mode_aging_optaa")
    lptr2_soa_a_amode = ints_2d("lptr2_soa_a_amode", ntot_amode, nsoa)
    mw_so4a_host      = scalar_float("mw_so4a_host")
    mw_nh4a_host      = scalar_float("mw_nh4a_host")
    dens_so4a_host    = scalar_float("dens_so4a_host")

    # Convert lmap_* from gas_pcnst-relative 1-based to pcnst-absolute 0-based.
    # Empty slots (Fortran 0) become -1 sentinel.
    def to_pcnst_0based(arr: np.ndarray) -> np.ndarray:
        return np.where(arr == 0, -1, arr - 1 + loffset).astype(np.int32)

    return {
        "amicphys_loffset":     np.int32(loffset),
        "amicphys_ngas":        np.int32(ngas),
        "amicphys_naer":        np.int32(naer),
        "amicphys_max_gas":     np.int32(max_gas),
        "amicphys_max_aer":     np.int32(max_aer),
        "lmap_gas":             lmap_gas,
        "lmap_num":             lmap_num,
        "lmap_numcw":           lmap_numcw,
        "lmap_aer":             lmap_aer,
        "lmap_aercw":           lmap_aercw,
        "pcnst_lmap_gas":       to_pcnst_0based(lmap_gas),
        "pcnst_lmap_num":       to_pcnst_0based(lmap_num),
        "pcnst_lmap_numcw":     to_pcnst_0based(lmap_numcw),
        "pcnst_lmap_aer":       to_pcnst_0based(lmap_aer),
        "pcnst_lmap_aercw":     to_pcnst_0based(lmap_aercw),
        "fcvt_gas":             fcvt_gas,
        "fcvt_aer":             fcvt_aer,
        "fcvt_num":             np.float64(fcvt_num),
        "fcvt_wtr":             np.float64(fcvt_wtr),
        "mwdry":                np.float64(mwdry),
        "adv_mass":             adv_mass,
        "vmdry":                np.float64(vmdry),
        "mw_gas":               mw_gas,
        "vol_molar_gas":        vol_molar_gas,
        "accom_coef_gas":       accom_coef_gas,
        "amicphys_npoa":        np.int32(npoa),
        "amicphys_nsoa":        np.int32(nsoa),
        "amicphys_iaer_pom":    np.int32(iaer_pom),
        "amicphys_iaer_soa":    np.int32(iaer_soa),
        "amicphys_npca":        np.int32(npca),
        "amicphys_nufi":        np.int32(nufi),
        "mode_aging_optaa":     mode_aging_optaa,
        "lptr2_soa_a_amode":    lptr2_soa_a_amode,
        "mw_so4a_host":         np.float64(mw_so4a_host),
        "mw_nh4a_host":         np.float64(mw_nh4a_host),
        "dens_so4a_host":       np.float64(dens_so4a_host),
    }


def run_instrumented(nstep: int, no_aitacc_transfer: bool = False,
                     amicphys_off: bool = False,
                     rename_only: bool = False,
                     gasaerexch_only: bool = False,
                     gasaerexch: bool = False,
                     gasaerexch_and_newnuc: bool = False,
                     coag_only: bool = False) -> list[Path]:
    dt = TOTAL_DURATION_S // nstep
    if coag_only:
        out_dir = PER_PROCESS_COAG_OUT_DIR
        flavour = "instrumented-coag-only"
    elif gasaerexch_and_newnuc:
        out_dir = PER_PROCESS_GASAEREXCH_AND_NEWNUC_OUT_DIR
        flavour = "instrumented-gasaerexch-and-newnuc-only"
    elif gasaerexch:
        out_dir = PER_PROCESS_GASAEREXCH_OUT_DIR
        flavour = "instrumented-gasaerexch-with-soaexch-only"
    elif gasaerexch_only:
        out_dir = PER_PROCESS_GASAEREXCH_ONLY_OUT_DIR
        flavour = "instrumented-gasaerexch-only"
    elif rename_only:
        out_dir = PER_PROCESS_RENAME_ONLY_OUT_DIR
        flavour = "instrumented-rename-only"
    elif amicphys_off:
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
    elif rename_only:
        write_namelist(dt, nstep,
                       mdo_gasaerexch=0, mdo_rename=1,
                       mdo_newnuc=0, mdo_coag=0)
    elif gasaerexch_only or gasaerexch:
        write_namelist(dt, nstep,
                       mdo_gasaerexch=1, mdo_rename=0,
                       mdo_newnuc=0, mdo_coag=0)
    elif gasaerexch_and_newnuc:
        write_namelist(dt, nstep,
                       mdo_gasaerexch=1, mdo_rename=0,
                       mdo_newnuc=1, mdo_coag=0)
    elif coag_only:
        write_namelist(dt, nstep,
                       mdo_gasaerexch=0, mdo_rename=0,
                       mdo_newnuc=0, mdo_coag=1)
    else:
        write_namelist(dt, nstep)
    print(f"[capture_reference] {flavour} dt={dt}s nstep={nstep} ...", flush=True)
    subprocess.run(["./mam_box_test.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)

    written: list[Path] = []

    # Index tables (written once at init, before the time loop). Only the
    # canonical full-physics run writes the canonical indices.
    if (not no_aitacc_transfer and not amicphys_off
            and not rename_only and not gasaerexch_only
            and not gasaerexch and not gasaerexch_and_newnuc
            and not coag_only):
        indices_txt = RUN_DIR / "mam4_indices.txt"
        if not indices_txt.is_file():
            raise RuntimeError(f"expected indices dump missing: {indices_txt}")
        contents = _read_indices(indices_txt)
        # Merge amicphys-internal init tables (M3.6 PR-C). Same .npz so
        # consumers can load both with one file.
        amicphys_txt = RUN_DIR / "mam4_amicphys_init.txt"
        if not amicphys_txt.is_file():
            raise RuntimeError(f"expected amicphys init dump missing: {amicphys_txt}")
        contents.update(_read_amicphys_init(amicphys_txt))
        indices_npz = INDICES_OUT_DIR / "reference.npz"
        np.savez(indices_npz, **contents)
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


def _read_newnuc_helpers(path: Path) -> dict[str, np.ndarray]:
    """Parse the multi-section newnuc helpers reference file.

    Sections:
      binary_inputs:   (ntot, 3)    temp, rh, so4vol
      binary_outputs:  (ntot, 5)    ratenucl, rateloge, cnum_h2so4, cnum_tot, radius_cluster
      pbl11_outputs:   (ntot, 7)    flagaa2 ratenucl rateloge cnum_h2so4 cnum_tot cnum_nh3 radius_cluster
      pbl12_outputs:   (ntot, 7)    same shape as pbl11
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

    def floats_2d(name: str, n_cols: int) -> np.ndarray:
        rows = [[float(t) for t in ln.split()] for ln in sections[name]]
        return np.asarray(rows, dtype=np.float64)

    def mixed_2d(name: str, n_cols_int: int, n_cols_float: int) -> tuple[np.ndarray, np.ndarray]:
        ints, floats = [], []
        for ln in sections[name]:
            parts = ln.split()
            ints.append([int(parts[i]) for i in range(n_cols_int)])
            floats.append([float(parts[i + n_cols_int]) for i in range(n_cols_float)])
        return (np.asarray(ints,   dtype=np.int32),
                np.asarray(floats, dtype=np.float64))

    bin_in  = floats_2d("binary_inputs",  3)
    bin_out = floats_2d("binary_outputs", 5)
    pbl11_int, pbl11_flt = mixed_2d("pbl11_outputs", 1, 6)
    pbl12_int, pbl12_flt = mixed_2d("pbl12_outputs", 1, 6)

    return {
        "temp":                bin_in[:, 0],
        "rh":                  bin_in[:, 1],
        "so4vol":              bin_in[:, 2],
        "binary_ratenucl":     bin_out[:, 0],
        "binary_rateloge":     bin_out[:, 1],
        "binary_cnum_h2so4":   bin_out[:, 2],
        "binary_cnum_tot":     bin_out[:, 3],
        "binary_radius":       bin_out[:, 4],
        "pbl11_flagaa2":       pbl11_int[:, 0],
        "pbl11_ratenucl":      pbl11_flt[:, 0],
        "pbl11_rateloge":      pbl11_flt[:, 1],
        "pbl11_cnum_h2so4":    pbl11_flt[:, 2],
        "pbl11_cnum_tot":      pbl11_flt[:, 3],
        "pbl11_cnum_nh3":      pbl11_flt[:, 4],
        "pbl11_radius":        pbl11_flt[:, 5],
        "pbl12_flagaa2":       pbl12_int[:, 0],
        "pbl12_ratenucl":      pbl12_flt[:, 0],
        "pbl12_rateloge":      pbl12_flt[:, 1],
        "pbl12_cnum_h2so4":    pbl12_flt[:, 2],
        "pbl12_cnum_tot":      pbl12_flt[:, 3],
        "pbl12_cnum_nh3":      pbl12_flt[:, 4],
        "pbl12_radius":        pbl12_flt[:, 5],
    }


def run_newnuc_helpers() -> list[Path]:
    NEWNUC_HELPERS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[capture_reference] running newnuc helpers driver ...", flush=True)
    subprocess.run(["./newnuc_helpers_driver.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)
    arrays = _read_newnuc_helpers(RUN_DIR / "newnuc_helpers_reference.txt")
    out = NEWNUC_HELPERS_OUT_DIR / "reference.npz"
    np.savez(out, **arrays)
    return [out]


def _read_mer07_veh02(path: Path) -> dict[str, np.ndarray]:
    """Parse the mer07_veh02 reference text file.

    Two sections:
      inputs:  (ntot, 5)   temp rh zm qh2so4 uptkrate
      outputs: (ntot, 8)   isize_nuc(int) qnuma_del qso4a_del qnh4a_del
                           qh2so4_del qnh3_del dens_nh4so4a dnclusterdt
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

    inputs = np.asarray(
        [[float(t) for t in ln.split()] for ln in sections["inputs"]],
        dtype=np.float64,
    )
    outputs_int = []
    outputs_flt = []
    for ln in sections["outputs"]:
        parts = ln.split()
        outputs_int.append([int(parts[0])])
        outputs_flt.append([float(parts[i + 1]) for i in range(7)])
    outputs_int = np.asarray(outputs_int, dtype=np.int32)
    outputs_flt = np.asarray(outputs_flt, dtype=np.float64)

    return {
        "temp":         inputs[:, 0],
        "rh":           inputs[:, 1],
        "zm":           inputs[:, 2],
        "qh2so4":       inputs[:, 3],
        "uptkrate":     inputs[:, 4],
        "isize_nuc":    outputs_int[:, 0],
        "qnuma_del":    outputs_flt[:, 0],
        "qso4a_del":    outputs_flt[:, 1],
        "qnh4a_del":    outputs_flt[:, 2],
        "qh2so4_del":   outputs_flt[:, 3],
        "qnh3_del":     outputs_flt[:, 4],
        "dens_nh4so4a": outputs_flt[:, 5],
        "dnclusterdt":  outputs_flt[:, 6],
    }


def run_mer07_veh02() -> list[Path]:
    MER07_VEH02_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[capture_reference] running mer07_veh02 driver ...", flush=True)
    subprocess.run(["./mer07_veh02_driver.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)
    arrays = _read_mer07_veh02(RUN_DIR / "mer07_veh02_reference.txt")
    out = MER07_VEH02_OUT_DIR / "reference.npz"
    np.savez(out, **arrays)
    return [out]


def _read_coag_coefficients(path: Path) -> dict[str, np.ndarray]:
    """Parse the coag-coefficients reference text file.

    Four sections:
      physical_inputs:   (ntot, 4)   temp press dgnumA dgnumB
      getcoags_inputs:   (ntot, 5)   lamda knc kfmat kfmac kfmatac
      getcoags_outputs:  (ntot, 8)   qs11 qn11 qs22 qn22 qs12 qs21 qn12 qv12
      wrapper_outputs:   (ntot, 8)   betaij0 betaij2i betaij2j betaij3
                                     betaii0 betaii2  betajj0  betajj2
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

    def floats(name: str, n_cols: int) -> np.ndarray:
        rows = [[float(t) for t in ln.split()] for ln in sections[name]]
        return np.asarray(rows, dtype=np.float64)

    phys = floats("physical_inputs",  4)
    gi   = floats("getcoags_inputs",  5)
    go   = floats("getcoags_outputs", 8)
    wo   = floats("wrapper_outputs",  8)

    return {
        "temp":     phys[:, 0],
        "press":    phys[:, 1],
        "dgnumA":   phys[:, 2],
        "dgnumB":   phys[:, 3],
        "lamda":    gi[:, 0],
        "knc":      gi[:, 1],
        "kfmat":    gi[:, 2],
        "kfmac":    gi[:, 3],
        "kfmatac":  gi[:, 4],
        "qs11":     go[:, 0],
        "qn11":     go[:, 1],
        "qs22":     go[:, 2],
        "qn22":     go[:, 3],
        "qs12":     go[:, 4],
        "qs21":     go[:, 5],
        "qn12":     go[:, 6],
        "qv12":     go[:, 7],
        "betaij0":  wo[:, 0],
        "betaij2i": wo[:, 1],
        "betaij2j": wo[:, 2],
        "betaij3":  wo[:, 3],
        "betaii0":  wo[:, 4],
        "betaii2":  wo[:, 5],
        "betajj0":  wo[:, 6],
        "betajj2":  wo[:, 7],
    }


def run_coag_coefficients() -> list[Path]:
    COAG_COEFFICIENTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[capture_reference] running coag_coefficients driver ...", flush=True)
    subprocess.run(["./coag_coefficients_driver.exe"], cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL)
    arrays = _read_coag_coefficients(RUN_DIR / "coag_coefficients_reference.txt")
    out = COAG_COEFFICIENTS_OUT_DIR / "reference.npz"
    np.savez(out, **arrays)
    return [out]


# ----- entry point ----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode",
        choices=("sweep", "instrumented", "instrumented-no-aitacc",
                 "instrumented-amicphys-off", "instrumented-rename-only",
                 "instrumented-gasaerexch-only",
                 "instrumented-gasaerexch-with-soaexch-only",
                 "instrumented-gasaerexch-and-newnuc-only",
                 "instrumented-coag-only",
                 "polysvp", "qsat", "makoh", "kohler", "newnuc-helpers",
                 "mer07-veh02", "coag-coefficients"),
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
    elif args.mode == "instrumented-rename-only":
        # Uses the default instrumented build (rename hook captures the
        # local view); single-toggle namelist isolates the rename
        # contribution from gasaerexch/newnuc/coag.
        ensure_built(instrumented=True)
        nstep = args.nstep if args.nstep is not None else 60
        if nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(nstep, rename_only=True)
        out_root = PER_PROCESS_RENAME_ONLY_OUT_DIR
    elif args.mode == "instrumented-gasaerexch-only":
        # Builds with the soaexch-skip overlay so that the JAX port
        # (which doesn't implement soaexch yet — that's PR-E) matches
        # the Fortran 1:1 with no SOA divergence.
        ensure_built(instrumented=True, skip_soaexch=True)
        nstep = args.nstep if args.nstep is not None else 60
        if nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(nstep, gasaerexch_only=True)
        out_root = PER_PROCESS_GASAEREXCH_ONLY_OUT_DIR
    elif args.mode == "instrumented-gasaerexch-with-soaexch-only":
        # Like instrumented-gasaerexch-only but WITHOUT the
        # gasaerexch_skip_soaexch.patch (we still apply skip_pcarbon_aging
        # because pcarbon aging is a separate sub-process outside M3.6).
        # The build script's --skip-soaexch flag applies both patches; we
        # need only the pcarbon one here, so we don't pass --skip-soaexch.
        # Instead we apply skip_pcarbon_aging directly via a new build flag.
        # (TODO: split build_reference.sh's --skip-soaexch into two flags.)
        ensure_built(instrumented=True, skip_pcarbon_aging=True)
        nstep = args.nstep if args.nstep is not None else 60
        if nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(nstep, gasaerexch=True)
        out_root = PER_PROCESS_GASAEREXCH_OUT_DIR
    elif args.mode == "instrumented-gasaerexch-and-newnuc-only":
        # gasaerexch + newnuc on (newnuc needs qgas_avg from gasaerexch
        # to fire), with pcarbon_aging skipped (still M3.6 out of scope).
        ensure_built(instrumented=True, skip_pcarbon_aging=True)
        nstep = args.nstep if args.nstep is not None else 60
        if nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(nstep, gasaerexch_and_newnuc=True)
        out_root = PER_PROCESS_GASAEREXCH_AND_NEWNUC_OUT_DIR
    elif args.mode == "instrumented-coag-only":
        # mdo_coag=1, others=0, with pcarbon_aging skipped (separate
        # sub-process outside M3.6 scope; same pattern as PR-E/F3).
        # Coag operates on the current state's dgn_a / dgn_awet /
        # wetdens (set by calcsize + wateruptake upstream of amicphys);
        # it does not need gasaerexch outputs, unlike newnuc.
        ensure_built(instrumented=True, skip_pcarbon_aging=True)
        nstep = args.nstep if args.nstep is not None else 60
        if nstep not in NSTEP_SWEEP:
            print(f"[capture_reference] warning: --nstep={nstep} is outside the canonical sweep "
                  f"{NSTEP_SWEEP}", file=sys.stderr)
        written = run_instrumented(nstep, coag_only=True)
        out_root = PER_PROCESS_COAG_OUT_DIR
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
    elif args.mode == "kohler":
        ensure_built(kohler=True)
        written = run_kohler()
        out_root = KOHLER_OUT_DIR
    elif args.mode == "newnuc-helpers":
        ensure_built(newnuc_helpers=True)
        written = run_newnuc_helpers()
        out_root = NEWNUC_HELPERS_OUT_DIR
    elif args.mode == "mer07-veh02":
        ensure_built(mer07_veh02=True)
        written = run_mer07_veh02()
        out_root = MER07_VEH02_OUT_DIR
    else:  # coag-coefficients
        ensure_built(coag_coefficients=True)
        written = run_coag_coefficients()
        out_root = COAG_COEFFICIENTS_OUT_DIR

    print(f"\n[capture_reference] {len(written)} file(s) written under {out_root}")
    for p in written:
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
