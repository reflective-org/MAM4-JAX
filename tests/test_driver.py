"""Validate the M4 PR-A operator-splitting time-loop driver.

Reference: ``tests/reference/per_process_full_minus_pcarbon_aging/*.npz``
— full-physics namelist (all four ``mdo_*=1``) with
``scripts/patches/skip_pcarbon_aging.patch`` applied at build time so
the pcarbon-aging sub-process is no-op'd. This matches the JAX port's
M3.6 scope: pcarbon aging is deferred and isn't in ``amicphys``. The
canonical ``per_process/`` fixture (pcarbon aging on) would diverge
from JAX on every step's Aitken / pcarbon tracers by amounts much
larger than ADR-003's 1e-6 budget.

PR-M4-A ships only a single-step driver test plus a few wiring smoke
tests. PR-M4-B will extend to the full 60-step trajectory + the
mode-by-mode size-distribution comparison figure.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.driver import run_step, run_timesteps

REF_DIR = (Path(__file__).resolve().parent
           / "reference" / "per_process_full_minus_pcarbon_aging")

# Box-model constants. Same convention as test_amicphys / test_calcsize.
T_BOX_MODEL    = 273.0
PMID_BOX_MODEL = 1.0e5
CLDN_BOX_MODEL = 0.0
DELTAT_60      = 30.0   # 1800 s / 60 snapshots
ZMID_BOX_MODEL = 3.0e3
PBLH_BOX_MODEL = 1.1e3
RH_BOX_MODEL   = 0.9


def _build_state(snapshot: dict[str, np.ndarray], step: int) -> dict[str, jnp.ndarray]:
    """Build a JAX state dict from one timestep of a per_process snapshot."""
    ncol, pver = snapshot["q"].shape[1], snapshot["q"].shape[2]
    return {
        "q":           jnp.asarray(snapshot["q"][step]),
        "qqcw":        jnp.asarray(snapshot["qqcw"][step]),
        "dgncur_a":    jnp.asarray(snapshot["dgncur_a"][step]),
        "dgncur_awet": jnp.asarray(snapshot["dgncur_awet"][step]),
        "qaerwat":     jnp.asarray(snapshot["qaerwat"][step]),
        "wetdens":     jnp.asarray(snapshot["wetdens"][step]),
        "t":           jnp.asarray(np.full((ncol, pver), T_BOX_MODEL)),
        "pmid":        jnp.asarray(np.full((ncol, pver), PMID_BOX_MODEL)),
        "cldn":        jnp.asarray(np.full((ncol, pver), CLDN_BOX_MODEL)),
        "zmid":        jnp.asarray(np.full((ncol, pver), ZMID_BOX_MODEL)),
        "pblh":        jnp.asarray(np.full((ncol, pver), PBLH_BOX_MODEL)),
        "relhum":      jnp.asarray(np.full((ncol, pver), RH_BOX_MODEL)),
        "deltat":      jnp.asarray(DELTAT_60),
    }


@pytest.fixture(scope="module")
def per_process() -> dict[str, dict[str, np.ndarray]]:
    """Load the relevant ``per_process/*.npz`` files into one dict."""
    return {
        tag: {k: np.asarray(v) for k, v in np.load(REF_DIR / f"{tag}.npz").items()}
        for tag in ("calcsize_before", "amicphys_after_writeback")
    }


def test_run_step_one_step_matches_fortran(per_process) -> None:
    """JAX ``run_step(state)`` starting from ``calcsize_before[0]``
    reproduces Fortran's ``amicphys_after_writeback[0]`` at 1e-6 on
    ``q`` / ``qqcw``.

    This validates the operator-splitting wiring (calcsize →
    wateruptake → cloud-chem no-op → amicphys → writeback) end-to-end
    over a single timestep.
    """
    ic = _build_state(per_process["calcsize_before"], step=0)
    new_state = run_step(ic)

    target = per_process["amicphys_after_writeback"]
    for key in ("q", "qqcw"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), target[key][0],
            rtol=1e-6, atol=1e-20,
            err_msg=f"driver 1-step diverged on {key!r}",
        )
    # Size fields: same caveat as the per-process amicphys tests —
    # Fortran's mid-substep update_aerosol_props re-uptake is out of
    # M3.6 scope, so JAX drifts on dgn_awet / qaerwat / wetdens.
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), target[key][0],
            rtol=1e-3, atol=1e-15,
            err_msg=f"driver 1-step drifted on {key!r}",
        )


def test_run_timesteps_shapes(per_process) -> None:
    """``run_timesteps(state, n)`` returns a trajectory with leading axis
    length ``n`` (matches Fortran's NetCDF output convention — index ``i``
    is post-step ``i+1``, the IC is not included).

    Smoke-tests the loop wiring without claiming bit-comparable accuracy
    across many steps; that's PR-M4-B's job.
    """
    ic = _build_state(per_process["calcsize_before"], step=0)
    traj = run_timesteps(ic, n_steps=3)
    for key in ("q", "qqcw", "dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        assert traj[key].shape[0] == 3, f"{key} leading axis != 3"
    # Step 0 of the trajectory should equal the run_step output.
    one_step = run_step(ic)
    for key in ("q", "qqcw"):
        np.testing.assert_array_equal(
            np.asarray(traj[key][0]), np.asarray(one_step[key]),
            err_msg=f"trajectory step 0 != run_step result on {key!r}",
        )


def test_run_timesteps_rejects_zero(per_process) -> None:
    """``run_timesteps(state, 0)`` raises (matches Fortran's
    ``do nstep = 1, nstop`` which requires ``nstop >= 1``)."""
    ic = _build_state(per_process["calcsize_before"], step=0)
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        run_timesteps(ic, n_steps=0)
