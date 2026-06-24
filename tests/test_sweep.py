"""24-hour validation sweep for the diffrax branch (M7 PR-D1).

Replaces the M5 12-point convergence sweep (1800 s window at
``rtol=1e-6``) with a 4-dt × 24h trajectory test at the relaxed
diffrax-branch acceptance bar (ADR-015): **max rel-err < 3% over
24h at dt ≤ 5s**. Coarser dt (30s, 300s) record diagnostics
without asserting — the dt-dependence at coarse dt is operator-
splitting truncation, not a diffrax issue, and gating on it would
conflate solver-port validation with driver-architecture work.

Background and rationale: ADR-015 (`docs/KEY_DECISIONS.md`),
`docs/plans/016-diffrax-soaexch.md`, and the
`project-diffrax-structural-offset` memory. The 6 previously-
`xfail`ed M5 cases on `main` (nstep ≤ 30) are deleted here — their
failure mode (single-substep semi-implicit gap) is fixed by
diffrax, and what remains is the structural diffrax-vs-Fortran
offset which is the focus of this rewritten sweep.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default; JAX_ENABLE_X64=0 to opt out
from mam4_jax import data
from mam4_jax.driver import run_timesteps

REF_24H_DIR = (Path(__file__).resolve().parent
               / "reference" / "sweep_24h_no_pcarbon_aging")
REF_IC_DIR = (Path(__file__).resolve().parent
              / "reference" / "per_process_full_minus_pcarbon_aging")

# 24h-validation dt values.
DT_GATED = (1, 5)        # asserted at the <3% bar
DT_DIAGNOSTIC = (30, 300)  # recorded, not asserted

TOTAL_DURATION_S = 86400  # 24 hours
ACCEPTANCE_BAR = 3.0e-2   # ADR-015: max rel-err < 3% on diffrax at dt ≤ 5s

T_BOX_MODEL = 273.0
PMID_BOX_MODEL = 1.0e5
ZMID_BOX_MODEL = 3.0e3
PBLH_BOX_MODEL = 1.1e3
RH_BOX_MODEL = 0.9


@pytest.fixture(scope="module")
def initial_state() -> dict[str, np.ndarray]:
    """``calcsize_before[0]`` IC — depends only on the namelist, not dt."""
    return {k: np.asarray(v)
            for k, v in np.load(REF_IC_DIR / "calcsize_before.npz").items()}


def _build_state(snapshot: dict[str, np.ndarray], deltat: float):
    ncol, pver = snapshot["q"].shape[1], snapshot["q"].shape[2]
    return {
        "q":           jnp.asarray(snapshot["q"][0]),
        "qqcw":        jnp.asarray(snapshot["qqcw"][0]),
        "dgncur_a":    jnp.asarray(snapshot["dgncur_a"][0]),
        "dgncur_awet": jnp.asarray(snapshot["dgncur_awet"][0]),
        "qaerwat":     jnp.asarray(snapshot["qaerwat"][0]),
        "wetdens":     jnp.asarray(snapshot["wetdens"][0]),
        "t":           jnp.asarray(np.full((ncol, pver), T_BOX_MODEL)),
        "pmid":        jnp.asarray(np.full((ncol, pver), PMID_BOX_MODEL)),
        "cldn":        jnp.asarray(np.full((ncol, pver), 0.0)),
        "zmid":        jnp.asarray(np.full((ncol, pver), ZMID_BOX_MODEL)),
        "pblh":        jnp.asarray(np.full((ncol, pver), PBLH_BOX_MODEL)),
        "relhum":      jnp.asarray(np.full((ncol, pver), RH_BOX_MODEL)),
        "deltat":      jnp.asarray(deltat),
    }


def _so4_pcnst_indices_per_mode() -> list[int]:
    out: list[int] = []
    for m in range(data.NTOT_AMODE):
        type_row = data.LSPECTYPE_AMODE[m]
        mass_row = data.LMASSPTR_AMODE[m]
        slot = next((s for s, t in enumerate(type_row) if t == 0), -1)
        out.append(int(mass_row[slot]) if slot >= 0 else -1)
    return out


def _soa_pcnst_indices_per_mode() -> list[int]:
    out: list[int] = []
    for m in range(data.NTOT_AMODE):
        type_row = data.LSPECTYPE_AMODE[m]
        mass_row = data.LMASSPTR_AMODE[m]
        slot = next((s for s, t in enumerate(type_row) if t == 4), -1)
        out.append(int(mass_row[slot]) if slot >= 0 else -1)
    return out


SO4_IDX = _so4_pcnst_indices_per_mode()
SOA_IDX = _soa_pcnst_indices_per_mode()
H2SO4_PCNST_IDX = int(data.LMAP_GAS[1])
SOAG_PCNST_IDX = int(data.LMAP_GAS[0])


def _run_and_compare(state: dict, nstep: int, dt: int) -> dict[str, float]:
    """Run JAX for `nstep` steps, compare to Fortran NetCDF at the same dt.

    Returns a dict of max per-field per-mode rel-err.
    """
    traj = run_timesteps(state, n_steps=nstep)
    nc_path = REF_24H_DIR / f"mam_dt{dt}_ndt{nstep}.nc"
    ds = nc.Dataset(nc_path, "r")
    try:
        f_num = np.asarray(ds.variables["num_aer"][:])
        f_so4 = np.asarray(ds.variables["so4_aer"][:])
        f_soa = np.asarray(ds.variables["soa_aer"][:])
        f_h2so4 = np.asarray(ds.variables["h2so4_gas"][:])
        f_soag = np.asarray(ds.variables["soag_gas"][:])
    finally:
        ds.close()

    j_q = np.asarray(traj["q"])
    j_num = np.stack(
        [j_q[:, 0, 0, int(data.NUMPTR_AMODE[m])] for m in range(4)], axis=0)
    j_so4 = np.stack(
        [j_q[:, 0, 0, SO4_IDX[m]] if SO4_IDX[m] >= 0 else np.zeros(nstep)
         for m in range(4)], axis=0)
    j_soa = np.stack(
        [j_q[:, 0, 0, SOA_IDX[m]] if SOA_IDX[m] >= 0 else np.zeros(nstep)
         for m in range(4)], axis=0)
    j_h2so4 = j_q[:, 0, 0, H2SO4_PCNST_IDX]
    j_soag = j_q[:, 0, 0, SOAG_PCNST_IDX]

    out: dict[str, float] = {}
    for fld, jv, fv in (
        ("num_aer", j_num, f_num),
        ("so4_aer", j_so4, f_so4),
        ("soa_aer", j_soa, f_soa),
    ):
        for m in range(4):
            if not np.any(fv[m]):
                continue
            rel = np.abs(jv[m] - fv[m]) / np.maximum(np.abs(fv[m]), 1e-300)
            out[f"{fld}_mode{m}"] = float(rel.max())
    for fld, jv, fv in (
        ("h2so4_gas", j_h2so4, f_h2so4),
        ("soag_gas", j_soag, f_soag),
    ):
        rel = np.abs(jv - fv) / np.maximum(np.abs(fv), 1e-300)
        out[fld] = float(rel.max())
    out["MAX"] = max(out.values())
    return out


@pytest.mark.parametrize("dt", DT_GATED)
def test_sweep_24h_diffrax_within_3pct(initial_state, dt: int) -> None:
    """At dt ∈ {1, 5}, max per-field per-mode rel-err over 24h must be < 3%
    (ADR-015's diffrax-branch bar). soag_gas typically dominates at ~2.5%.
    """
    nstep = TOTAL_DURATION_S // dt
    state = _build_state(initial_state, deltat=float(dt))
    rel = _run_and_compare(state, nstep, dt)
    worst = max(rel, key=lambda k: rel[k] if k != "MAX" else -1)
    assert rel["MAX"] < ACCEPTANCE_BAR, (
        f"dt={dt}s, 24h: max rel-err {rel['MAX']:.3e} exceeds 3% bar. "
        f"Worst field: {worst} at {rel[worst]:.3e}. Full breakdown: {rel}"
    )


@pytest.mark.parametrize("dt", DT_DIAGNOSTIC)
def test_sweep_24h_diffrax_diagnostic(initial_state, dt: int,
                                       capsys) -> None:
    """At dt ∈ {30, 300}, record max rel-err for visibility but do NOT
    assert against the 3% bar. The dt-dependence at coarse dt is
    operator-splitting truncation in the driver (ADR-015 §coarse-dt);
    gating on it conflates solver-port work with M6 driver work.

    This test always passes; its purpose is to print the diagnostic to
    pytest output so it stays visible in CI logs.
    """
    nstep = TOTAL_DURATION_S // dt
    state = _build_state(initial_state, deltat=float(dt))
    rel = _run_and_compare(state, nstep, dt)
    with capsys.disabled():
        worst = max((k for k in rel if k != "MAX"), key=lambda k: rel[k])
        print(f"\n[24h diag] dt={dt}s nstep={nstep}: max rel-err "
              f"{rel['MAX']:.3e} on {worst}. (Not gated; ADR-015.)")
