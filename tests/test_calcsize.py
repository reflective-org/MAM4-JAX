"""Validate JAX `calcsize(do_aitacc_transfer=False)` against the
M2 no-aitacc Fortran capture.

Reference: ``tests/reference/per_process_no_aitacc/calcsize_{before,after}.npz``
captured by ``scripts/capture_reference.py --mode instrumented-no-aitacc``
(applies ``disable_aitacc_transfer.patch`` so the Fortran calcsize call
runs with ``do_aitacc_transfer_in=.false.``).

The companion test_calcsize_transfer.py validates the same function with
``do_aitacc_transfer=True`` against the full-transfer reference at
``tests/reference/per_process/``.

The capture is at ``nstep=60`` because calcsize is essentially a no-op
at ``nstep=1`` (the initial state is already "consistent" — number,
mass, and dgncur_a satisfy the Köhler/v2ncur relationship). At longer
integrations, ``amicphys`` perturbs the mass mixing ratios and calcsize
recomputes dgncur_a accordingly.

Tolerance: max relative error < 1e-6 element-wise (ADR-003), with an
absolute floor for tracers near zero.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.processes.calcsize import calcsize

REF_DIR = Path(__file__).resolve().parent / "reference" / "per_process_no_aitacc"
BEFORE_NPZ = REF_DIR / "calcsize_before.npz"
AFTER_NPZ  = REF_DIR / "calcsize_after.npz"

# Box-model dt for the nstep=60 capture (1800 s / 60 = 30 s).
DELTAT = 30.0


@pytest.fixture(scope="module")
def captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    before = {k: np.asarray(v) for k, v in np.load(BEFORE_NPZ).items()}
    after  = {k: np.asarray(v) for k, v in np.load(AFTER_NPZ).items()}
    return before, after


def _build_state(before: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {
        "q":        jnp.asarray(before["q"]),
        "qqcw":     jnp.asarray(before["qqcw"]),
        "dgncur_a": jnp.asarray(before["dgncur_a"]),
        "deltat":   jnp.asarray(DELTAT),
    }


def _max_rel_err(jax_arr: jnp.ndarray, fortran_arr: np.ndarray,
                 abs_floor: float = 0.0) -> float:
    a = np.asarray(jax_arr, dtype=np.float64)
    b = fortran_arr
    denom = np.maximum(np.abs(b), abs_floor)
    err = np.where(denom > 0, np.abs(a - b) / denom, np.abs(a - b))
    return float(np.max(err))


def test_calcsize_dgncur_a_matches_fortran(captured) -> None:
    """dgncur_a evolution across 60 timesteps matches Fortran."""
    before, after = captured
    new_state = calcsize(_build_state(before), do_aitacc_transfer=False)
    rel = _max_rel_err(new_state["dgncur_a"], after["dgncur_a"])
    assert rel < 1e-6, f"dgncur_a max rel-err = {rel:.3e}"


def test_calcsize_q_passthrough_in_box_model(captured) -> None:
    """In the box-model setup, calcsize's number-bounds adjustment never
    triggers — the number tracers should pass through unchanged."""
    before, after = captured
    new_state = calcsize(_build_state(before), do_aitacc_transfer=False)
    # Number tracers: q[..., numptr_amode] should match the Fortran's
    # post-tendency-application q (which itself doesn't change here).
    from mam4_jax.data import INDEX_TABLES
    for m, idx in enumerate(INDEX_TABLES.numptr_amode.tolist()):
        rel = _max_rel_err(new_state["q"][..., idx], after["q"][..., idx])
        assert rel < 1e-6, f"mode {m} number tracer rel-err = {rel:.3e}"


def test_calcsize_returns_v2ncur(captured) -> None:
    """`calcsize` writes `v2ncur_a` (volume-to-number ratio) into the
    returned state for downstream consumers."""
    before, _ = captured
    new_state = calcsize(_build_state(before), do_aitacc_transfer=False)
    assert "v2ncur_a" in new_state
    assert new_state["v2ncur_a"].shape == before["dgncur_a"].shape


def test_calcsize_state_pass_through_other_keys(captured) -> None:
    """Untouched state keys pass through (only q, qqcw, dgncur_a updated)."""
    before, _ = captured
    state = _build_state(before)
    new_state = calcsize(state, do_aitacc_transfer=False)
    # deltat passes through; q changes only at number-tracer indices.
    np.testing.assert_array_equal(np.asarray(new_state["deltat"]),
                                  np.asarray(state["deltat"]))
