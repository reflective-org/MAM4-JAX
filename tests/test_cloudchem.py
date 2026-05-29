"""M8 PR-K2: per-process validation of cloudchem_simple_sub vs Fortran.

Fixture: ``tests/reference/per_process_cloudchem/cloudchem_{before,after}.npz``
captured at full physics + mdo_cloudchem=1 + cldn=0.5 over 60 timesteps
of dt=30s (PR-K1 + PR-K1c). Slots use ``vmr/vmrcw`` keys with
``gas_pcnst=30`` third-dim (volume mixing ratios), per PR-K1c.

Cloudchem is algebraic (no internal ODE) — the JAX port matches
Fortran bit-exact at float64. Bar = ADR-003 ``1e-6`` × safety margin;
machine ε is the realistic expectation.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mam4_jax.processes.cloudchem import cloudchem_simple_sub


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "reference" / "per_process_cloudchem"

# Fixture is captured with cldn = 0.5 (cloudchem_enable.patch) and
# dt = 30s (nstep=60 over 1800s). Both have to match what cloudchem
# saw inside Fortran for the per-step JAX call to be apples-to-apples.
CLDN_FIXTURE = 0.5
DT_FIXTURE = 30.0
N_STEPS = 60


@pytest.fixture(scope="module")
def cloudchem_fixture():
    before = np.load(FIXTURE_DIR / "cloudchem_before.npz")
    after  = np.load(FIXTURE_DIR / "cloudchem_after.npz")
    assert before["vmr"].shape == (N_STEPS, 1, 1, 30), \
        f"unexpected vmr shape {before['vmr'].shape}"
    assert before["vmrcw"].shape == (N_STEPS, 1, 1, 30), \
        f"unexpected vmrcw shape {before['vmrcw'].shape}"
    return before, after


def test_cloudchem_matches_fortran_per_step(cloudchem_fixture):
    """JAX cloudchem_simple_sub vs Fortran across all 60 fixture steps."""
    before, after = cloudchem_fixture
    cldn = jnp.full((1, 1), CLDN_FIXTURE)

    max_vmr_err = 0.0
    max_vmrcw_err = 0.0
    for step in range(N_STEPS):
        vmr_in   = jnp.asarray(before["vmr"][step])
        vmrcw_in = jnp.asarray(before["vmrcw"][step])
        vmr_out, vmrcw_out = cloudchem_simple_sub(
            vmr_in, vmrcw_in, cldn, DT_FIXTURE
        )

        # Compare against Fortran "after" state. Algebraic step; expect
        # machine eps. atol floor avoids 0/0 on initially-zero slots
        # (notably vmrcw[SO4_c1] which starts at 0 and accumulates).
        np.testing.assert_allclose(
            np.asarray(vmr_out),   after["vmr"][step],
            rtol=1e-6, atol=1e-30,
            err_msg=f"vmr mismatch at step {step}",
        )
        np.testing.assert_allclose(
            np.asarray(vmrcw_out), after["vmrcw"][step],
            rtol=1e-6, atol=1e-30,
            err_msg=f"vmrcw mismatch at step {step}",
        )

        # Also track the max for reporting.
        vmr_denom   = np.maximum(np.abs(after["vmr"][step]),   1e-30)
        vmrcw_denom = np.maximum(np.abs(after["vmrcw"][step]), 1e-30)
        max_vmr_err = max(max_vmr_err, float(np.max(
            np.abs(np.asarray(vmr_out)   - after["vmr"][step])   / vmr_denom)))
        max_vmrcw_err = max(max_vmrcw_err, float(np.max(
            np.abs(np.asarray(vmrcw_out) - after["vmrcw"][step]) / vmrcw_denom)))

    # Sanity-record the worst case: algebraic step, must be at or near eps.
    assert max_vmr_err   < 1e-12, f"vmr   worst rel-err {max_vmr_err:.3e}"
    assert max_vmrcw_err < 1e-12, f"vmrcw worst rel-err {max_vmrcw_err:.3e}"


def test_cloudchem_cycle_threshold_zeros_tendencies():
    """At cldn <= 0.009 the body is no-op; outputs equal inputs.

    Mirrors Fortran's ``if (cldn(i,k) <= 0.009_r8) cycle``. JAX
    implements this via ``jnp.where(fired, tendency, 0.0)`` masking.
    """
    rng = np.random.default_rng(0)
    vmr_in   = jnp.asarray(rng.uniform(1e-13, 1e-3, size=(1, 1, 30)))
    vmrcw_in = jnp.asarray(rng.uniform(1e-13, 1e-3, size=(1, 1, 30)))
    # Two test points just below and at the threshold.
    for cldn_val in (0.0, 0.005, 0.009):
        cldn = jnp.full((1, 1), cldn_val)
        vmr_out, vmrcw_out = cloudchem_simple_sub(vmr_in, vmrcw_in, cldn, DT_FIXTURE)
        np.testing.assert_array_equal(np.asarray(vmr_out),   np.asarray(vmr_in))
        np.testing.assert_array_equal(np.asarray(vmrcw_out), np.asarray(vmrcw_in))


def test_cloudchem_soag_unmodified(cloudchem_fixture):
    """SOAG gas is not touched by cloudchem (negative-control tracer).

    The fixture captures step 0 through step 59 in the post-cloudchem
    state. SOAG sits at vmr[..., VMR_SOAG] and should be byte-identical
    between cloudchem_before and cloudchem_after for every step.
    """
    from mam4_jax import data
    before, after = cloudchem_fixture
    soag_before = before["vmr"][:, :, :, data.VMR_SOAG]
    soag_after  = after["vmr"][:,  :, :, data.VMR_SOAG]
    np.testing.assert_array_equal(soag_before, soag_after)
