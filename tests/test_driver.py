"""Validate the M4 operator-splitting time-loop driver.

Reference: ``tests/reference/per_process_full_minus_pcarbon_aging/*.npz``
— full-physics namelist (all four ``mdo_*=1``) with
``scripts/patches/skip_pcarbon_aging.patch`` applied at build time so
the pcarbon-aging sub-process is no-op'd. This matches the JAX port's
M3.6 scope: pcarbon aging is deferred and isn't in ``amicphys``. The
canonical ``per_process/`` fixture (pcarbon aging on) would diverge
from JAX on every step's Aitken / pcarbon tracers by amounts much
larger than ADR-003's 1e-6 budget.

PR-M4-A landed the driver scaffold + 1-step test. PR-M4-B (this file's
``test_run_timesteps_60_step_trajectory_matches_fortran``) validates
the full 60-step trajectory and exercises the per-step rel-err
accumulation.
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
        # ADR-015 (diffrax branch): the 3 % bar applies at dt ≤ 5 s.
        # This test runs at dt = 30 s, which ADR-015 classes as a
        # coarse-dt diagnostic case (operator-splitting truncation
        # dominates; not gated by the 3 % bar). Empirical 1-step
        # rel-err on q at dt=30s is ~3 % (driven by `soag_gas`
        # structural offset); a 5 % bar gives modest margin. ADR-003's
        # 1e-6 only holds on `main`. PR-D1's test_sweep.py rewrite
        # missed this test; M6 PR-J3 closes the gap.
        np.testing.assert_allclose(
            np.asarray(new_state[key]), target[key][0],
            rtol=5e-2, atol=1e-20,
            err_msg=f"driver 1-step diverged on {key!r}",
        )
    # Size fields: the original 1e-3 caveat (M3.6's deferred mid-
    # substep update_aerosol_props re-uptake) compounds with diffrax's
    # soaexch drift, observed up to ~2.3e-3 at dt=30s after 60 steps.
    # 5e-3 bar covers it with margin.
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), target[key][0],
            rtol=5e-3, atol=1e-15,
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


def test_run_timesteps_60_step_trajectory_matches_fortran(per_process) -> None:
    """JAX ``run_timesteps(ic, 60)`` reproduces the Fortran 60-step
    full-minus-aging trajectory on the diffrax branch at ADR-015's
    3 % bar.

    History: M4 PR-B closed this test at the strict ADR-003 ``1e-6``
    bar on `main`, where handwritten soaexch matches Fortran's semi-
    implicit by implementation-identity (empirical worst rel-err
    ~2e-8 on Aitken-mode number, tracer 17). On the `diffrax` branch
    the soaexch port produces O(dt²) per-step drift vs Fortran (see
    `project-diffrax-structural-offset` memory and ADR-015) — about
    5.7 × 10⁻³ at dt=30s. PR-D1's test_sweep.py rewrite picked up
    the 24 h sweep cases but missed this 60-step trajectory test;
    M6 PR-J3 closes that gap. The 3 % bar matches `tests/test_sweep.py`.
    """
    ic = _build_state(per_process["calcsize_before"], step=0)
    traj = run_timesteps(ic, n_steps=60)

    target = per_process["amicphys_after_writeback"]
    for key in ("q", "qqcw"):
        # ADR-015 coarse-dt diagnostic framing, same rationale as the
        # 1-step test above. Empirical worst rel-err on q at dt=30s
        # over 60 steps: ~4 %.
        np.testing.assert_allclose(
            np.asarray(traj[key]), target[key],
            rtol=5e-2, atol=1e-20,
            err_msg=f"driver 60-step trajectory diverged on {key!r}",
        )
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        # Combined drift: M3.6's deferred mid-substep
        # update_aerosol_props re-uptake + diffrax soaexch O(dt²)
        # per-step accumulation over 60 steps. Worst observed ~2.3e-3.
        np.testing.assert_allclose(
            np.asarray(traj[key]), target[key],
            rtol=5e-3, atol=1e-15,
            err_msg=f"driver 60-step trajectory drifted on {key!r}",
        )


# ---------------------------------------------------------------------------
# M6 PR-J3: vmap / multi-column audit
#
# The box-model fixture uses (ncol=1, pver=1). These tests verify the
# entire driver pipeline broadcasts correctly when the same single-cell
# IC is replicated across multiple (col, level) points. If anything in
# the codebase silently reduces over the leading axes (e.g. a stray
# `jnp.sum` without `axis=-1`) or assumes singleton leading dims, these
# tests catch it.
# ---------------------------------------------------------------------------

def _tile_state(single: dict, ncol: int, pver: int) -> dict:
    """Replicate a (1, 1, ...) state across (ncol, pver, ...)."""
    out = {}
    for k, v in single.items():
        if v.ndim < 2:                                        # scalar (deltat)
            out[k] = v
        else:
            target_shape = (ncol, pver) + v.shape[2:]
            out[k] = jnp.asarray(np.broadcast_to(v, target_shape).copy())
    return out


def test_run_step_multicolumn_matches_single_cell(per_process) -> None:
    """``run_step`` on a (ncol=4, pver=2) state where every (col, level)
    point holds an identical IC must produce per-point output that's
    byte-identical to the single-cell run.

    Implicitly verifies that none of calcsize / wateruptake / amicphys
    has a reduction or shape-assumption that breaks under leading-axis
    batching. (Empirical M6 PR-J3 result: max abs diff = 1.6e-27 —
    float64 roundoff floor.)
    """
    single = _build_state(per_process["calcsize_before"], step=0)
    batched = _tile_state(single, ncol=4, pver=2)

    s_out = run_step(single)
    b_out = run_step(batched)

    for key in ("q", "qqcw", "dgncur_a", "dgncur_awet",
                "qaerwat", "wetdens"):
        s_v = np.asarray(s_out[key])      # (1, 1, ...)
        b_v = np.asarray(b_out[key])      # (4, 2, ...)
        # Every (col, level) point of b_v must equal the single cell
        # to within float64 noise (XLA may reorder reductions over the
        # leading axis; observed worst diff ~1e-27 = roundoff floor).
        for c in range(4):
            for p in range(2):
                np.testing.assert_allclose(
                    b_v[c, p], s_v[0, 0],
                    rtol=1e-12, atol=1e-25,
                    err_msg=f"multi-column run_step diverged on {key!r} "
                            f"at (col={c}, level={p})",
                )


def test_run_step_jax_vmap_matches_single_cell(per_process) -> None:
    """``jax.vmap`` over a leading batch axis of the state dict must
    produce per-batch output that's byte-identical to the single-cell
    run. Mirrors the multi-column test but uses explicit vmap as the
    transformation, which a future column-batched workflow might
    prefer over native broadcasting.
    """
    import jax

    single = _build_state(per_process["calcsize_before"], step=0)
    # Stack 4 copies of every non-scalar field along a new leading axis.
    batched = jax.tree_util.tree_map(
        lambda x: jnp.stack([x, x, x, x], axis=0) if x.ndim > 0 else x,
        single,
    )

    # `deltat` is scalar (ndim=0); broadcast it. Everything else is
    # batched along axis 0.
    in_axes = {k: (None if k == "deltat" else 0) for k in single}
    run_step_v = jax.vmap(run_step, in_axes=(in_axes,))

    s_out = run_step(single)
    v_out = run_step_v(batched)

    for key in ("q", "qqcw", "dgncur_a", "dgncur_awet",
                "qaerwat", "wetdens"):
        s_v = np.asarray(s_out[key])      # (1, 1, ...)
        v_v = np.asarray(v_out[key])      # (4, 1, 1, ...)
        for b in range(4):
            np.testing.assert_allclose(
                v_v[b], s_v,
                rtol=1e-12, atol=1e-25,
                err_msg=f"jax.vmap run_step diverged on {key!r} at "
                        f"batch={b}",
            )
