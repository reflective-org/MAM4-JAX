"""Validate JAX vs Fortran on the 12-point convergence sweep (M5).

For each ``(deltat, nstep)`` in the canonical sweep, drive
``mam4_jax.driver.run_timesteps(ic, nstep)`` with ``state["deltat"] = deltat``
and compare the per-step trajectory against the matching Fortran NetCDF
in ``tests/reference/sweep_no_pcarbon_aging/``.

**On ``main``: restricted to nstep >= 60.** At nstep <= 30 (i.e.
``deltat >= 60s``) Fortran's ``mam_soaexch_1subarea``
(``modal_aero_amicphys.F90:3835-3843``) triggers adaptive substepping
(``dtcur = alpha_astem/tmpa``) — multiple smaller integration steps
within one amicphys call. The ``main``-branch JAX port assumes
single-substep (``dtcur = dtfull``). **Permanently deferred on
``main`` per ADR-013** (``docs/KEY_DECISIONS.md``): adaptive
substepping is solely the ``diffrax`` branch's job, since diffrax's
standard adaptive controller provides it natively. The 6
small-``nstep`` cases here are marked ``xfail`` and should flip to
expected-pass on the ``diffrax`` branch with the test file
structurally identical to ``main`` — only the solver differs.

The 60-step fixture used by M4 PR-B stayed below Fortran's adaptive-
substep trigger because ``deltat = 30s`` is small relative to the SOA
``alpha_astem / tmpa`` threshold. The empirical threshold is sharp:
nstep >= 60 matches at ~2e-8; nstep <= 30 jumps to 3e-3 to 1e-1.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax import data
from mam4_jax.driver import run_timesteps

REF_SWEEP_DIR = (Path(__file__).resolve().parent
                 / "reference" / "sweep_no_pcarbon_aging")
REF_IC_DIR    = (Path(__file__).resolve().parent
                 / "reference" / "per_process_full_minus_pcarbon_aging")

#: Step counts validated in this PR — restricted to where JAX's
#: single-substep SOA assumption holds (deltat <= 30s).
NSTEP_OK   = (60, 120, 180, 360, 900, 1800)
#: Step counts that require adaptive SOA substepping (PR-E2 follow-up).
NSTEP_DEFER = (1, 2, 4, 9, 18, 30)

T_BOX_MODEL    = 273.0
PMID_BOX_MODEL = 1.0e5
ZMID_BOX_MODEL = 3.0e3
PBLH_BOX_MODEL = 1.1e3
RH_BOX_MODEL   = 0.9
TOTAL_DURATION_S = 1800


@pytest.fixture(scope="module")
def initial_state() -> dict[str, np.ndarray]:
    """``calcsize_before[0]`` from the M4 PR-A fixture. The IC depends
    only on the namelist (not ``nstep``), so the same snapshot serves
    every sweep point."""
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
    """Match Fortran's ``lptr_so4_a_amode(i)`` — the SO4 mass-tracer
    pcnst index for each mode. Inferred from ``LMASSPTR_AMODE`` +
    ``LSPECTYPE_AMODE`` (sulfate is species type 0 → first slot)."""
    out: list[int] = []
    for m in range(data.NTOT_AMODE):
        type_row = data.LSPECTYPE_AMODE[m]
        mass_row = data.LMASSPTR_AMODE[m]
        sulfate_slot = next(
            (s for s, t in enumerate(type_row) if t == 0), -1)
        out.append(int(mass_row[sulfate_slot]) if sulfate_slot >= 0 else -1)
    return out


def _soa_pcnst_indices_per_mode() -> list[int]:
    """Match Fortran's ``lptr_soa_a_amode(i)``. SOA = 's-organic' =
    species type 4 in SPECNAME_AMODE (sulfate=0, ammonium=1, nitrate=2,
    p-organic=3, s-organic=4, black-c=5, seasalt=6, dust=7, m-organic=8).
    Pcarbon mode has no SOA slot — returns -1, equivalent to Fortran's
    ``lptr_soa_a_amode(pcarbon) = 0`` (no mass tracer)."""
    out: list[int] = []
    for m in range(data.NTOT_AMODE):
        type_row = data.LSPECTYPE_AMODE[m]
        mass_row = data.LMASSPTR_AMODE[m]
        slot = next((s for s, t in enumerate(type_row) if t == 4), -1)
        out.append(int(mass_row[slot]) if slot >= 0 else -1)
    return out


SO4_IDX = _so4_pcnst_indices_per_mode()
SOA_IDX = _soa_pcnst_indices_per_mode()
H2SO4_PCNST_IDX = int(data.LMAP_GAS[1])   # tracer 6
SOAG_PCNST_IDX  = int(data.LMAP_GAS[0])   # tracer 9


@pytest.mark.parametrize("nstep", NSTEP_OK)
def test_sweep_matches_fortran(initial_state, nstep: int) -> None:
    """JAX 12-point convergence sweep: nstep in {60, 120, ..., 1800}.

    Validates that the JAX driver reproduces Fortran's NetCDF outputs
    (``num_aer``, ``so4_aer``, ``soa_aer``, ``h2so4_gas``, ``soag_gas``)
    at every captured timestep within ADR-003's 1e-6 budget. Size
    fields (``dgn_a``) get the 1e-3 caveat consistent with prior
    tests.
    """
    deltat = TOTAL_DURATION_S // nstep
    state = _build_state(initial_state, deltat=float(deltat))
    traj = run_timesteps(state, n_steps=nstep)

    nc_path = REF_SWEEP_DIR / f"mam_dt{deltat}_ndt{nstep}.nc"
    ds = nc.Dataset(nc_path, "r")
    try:
        f_num   = np.asarray(ds.variables["num_aer"][:])
        f_so4   = np.asarray(ds.variables["so4_aer"][:])
        f_soa   = np.asarray(ds.variables["soa_aer"][:])
        f_h2so4 = np.asarray(ds.variables["h2so4_gas"][:])
        f_soag  = np.asarray(ds.variables["soag_gas"][:])
        f_dgn_a = np.asarray(ds.variables["dgn_a"][:])
    finally:
        ds.close()

    j_q   = np.asarray(traj["q"])           # (nstep, 1, 1, 35)
    j_dgn = np.asarray(traj["dgncur_a"])    # (nstep, 1, 1, 4)

    j_num = np.stack(
        [j_q[:, 0, 0, int(data.NUMPTR_AMODE[m])] for m in range(4)], axis=0)
    j_so4 = np.stack(
        [j_q[:, 0, 0, SO4_IDX[m]] if SO4_IDX[m] >= 0 else np.zeros(nstep)
         for m in range(4)], axis=0)
    j_soa = np.stack(
        [j_q[:, 0, 0, SOA_IDX[m]] if SOA_IDX[m] >= 0 else np.zeros(nstep)
         for m in range(4)], axis=0)
    j_h2so4 = j_q[:, 0, 0, H2SO4_PCNST_IDX]
    j_soag  = j_q[:, 0, 0, SOAG_PCNST_IDX]
    j_dgn_a = np.stack([j_dgn[:, 0, 0, m] for m in range(4)], axis=0)

    # rtol=1e-6 on tracers, rtol=1e-3 on dgn_a (size-field caveat).
    for name, jv, fv in (
        ("num_aer",    j_num,   f_num),
        ("so4_aer",    j_so4,   f_so4),
        ("soa_aer",    j_soa,   f_soa),
        ("h2so4_gas",  j_h2so4, f_h2so4),
        ("soag_gas",   j_soag,  f_soag),
    ):
        np.testing.assert_allclose(
            jv, fv, rtol=1e-6, atol=1e-20,
            err_msg=f"sweep nstep={nstep}: {name} diverged from Fortran",
        )
    np.testing.assert_allclose(
        j_dgn_a, f_dgn_a, rtol=1e-3, atol=1e-15,
        err_msg=f"sweep nstep={nstep}: dgn_a drifted",
    )


@pytest.mark.parametrize("nstep", NSTEP_DEFER)
def test_sweep_xfail_without_adaptive_soa_substep(initial_state, nstep: int) -> None:
    """At nstep <= 30 (``deltat >= 60s``) Fortran's SOA exchange
    triggers adaptive substepping. The ``main``-branch JAX port assumes
    single-substep so it diverges. **Permanently deferred on ``main``**
    per ADR-013 (``docs/KEY_DECISIONS.md``); resolved on the long-lived
    ``diffrax`` branch where diffrax's standard adaptive controller
    provides substepping for free.

    Marked ``xfail`` so the gap remains visible in pytest output and the
    per-nstep rel-err is quoted in the xfail message. These cases will
    flip to expected-pass on the ``diffrax`` branch (with the test file
    structurally identical to ``main`` — only the solver differs).
    """
    deltat = TOTAL_DURATION_S // nstep
    state = _build_state(initial_state, deltat=float(deltat))
    traj = run_timesteps(state, n_steps=nstep)

    nc_path = REF_SWEEP_DIR / f"mam_dt{deltat}_ndt{nstep}.nc"
    ds = nc.Dataset(nc_path, "r")
    try:
        f_num = np.asarray(ds.variables["num_aer"][:])
    finally:
        ds.close()
    j_q   = np.asarray(traj["q"])
    j_num = np.stack(
        [j_q[:, 0, 0, int(data.NUMPTR_AMODE[m])] for m in range(4)], axis=0)
    rel = np.abs(j_num - f_num) / np.maximum(np.abs(f_num), 1e-300)

    pytest.xfail(
        f"nstep={nstep} (dt={deltat}s) — adaptive SOA substepping. "
        f"Permanently deferred on main per ADR-013; the `diffrax` "
        f"branch closes this. Worst num_aer rel-err: {rel.max():.2e}."
    )
