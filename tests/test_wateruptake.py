"""Validate JAX wateruptake end-to-end against the M2 Fortran capture.

The M2 instrumentation overlay (ADR-012) captured per-process I/O for
the box-model run. ``wateruptake_before.npz`` holds the inputs we feed
the JAX port; ``wateruptake_after.npz`` holds the Fortran's outputs.

Tolerance: max element-wise relative error < 1e-6 (ADR-003).
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.processes.wateruptake import wateruptake

REF_DIR = Path(__file__).resolve().parent / "reference" / "per_process"
BEFORE_NPZ = REF_DIR / "wateruptake_before.npz"
AFTER_NPZ  = REF_DIR / "wateruptake_after.npz"

# Box-model constants the M2 instrumentation does not capture but which
# are pinned by the namelist (driver.F90:577, 591; run_test.csh sets
# temp=273 and press=1e5). cld is hard-coded to 0 in driver.F90:591.
T_BOX_MODEL    = 273.0
PMID_BOX_MODEL = 1.0e5
CLDN_BOX_MODEL = 0.0


@pytest.fixture(scope="module")
def captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    before = {k: np.asarray(v) for k, v in np.load(BEFORE_NPZ).items()}
    after  = {k: np.asarray(v) for k, v in np.load(AFTER_NPZ).items()}
    return before, after


def _build_state(before: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    # The .npz captures one snapshot of the (1, 1, 1, ...) box-model
    # state. Strip the leading "snapshot" axis to get (1, 1, ...).
    return {
        "q":        jnp.asarray(before["q"][0]),
        "dgncur_a": jnp.asarray(before["dgncur_a"][0]),
        "t":        jnp.asarray(np.full((1, 1), T_BOX_MODEL)),
        "pmid":     jnp.asarray(np.full((1, 1), PMID_BOX_MODEL)),
        "cldn":     jnp.asarray(np.full((1, 1), CLDN_BOX_MODEL)),
    }


def _max_rel_err(jax_arr: jnp.ndarray, fortran_arr: np.ndarray,
                 abs_floor: float = 0.0) -> float:
    a = np.asarray(jax_arr, dtype=np.float64)
    b = fortran_arr
    denom = np.maximum(np.abs(b), abs_floor)
    err = np.where(denom > 0, np.abs(a - b) / denom, np.abs(a - b))
    return float(np.max(err))


def test_dgncur_awet_matches_fortran(captured) -> None:
    before, after = captured
    new_state = wateruptake(_build_state(before))
    rel = _max_rel_err(new_state["dgncur_awet"], after["dgncur_awet"][0])
    assert rel < 1e-6, f"dgncur_awet max rel-err = {rel:.3e}"


def test_qaerwat_matches_fortran(captured) -> None:
    before, after = captured
    new_state = wateruptake(_build_state(before))
    # qaerwat has values spanning ~10^-20 to ~10^-9; tiny absolute values
    # are physically meaningless. Use an absolute floor in the comparison
    # so we don't blow up on the mode-3 (primary-carbon) ~1e-20 value.
    rel = _max_rel_err(new_state["qaerwat"], after["qaerwat"][0],
                       abs_floor=1e-30)
    assert rel < 1e-6, f"qaerwat max rel-err = {rel:.3e}"


def test_wetdens_matches_fortran(captured) -> None:
    before, after = captured
    new_state = wateruptake(_build_state(before))
    rel = _max_rel_err(new_state["wetdens"], after["wetdens"][0])
    assert rel < 1e-6, f"wetdens max rel-err = {rel:.3e}"


def test_state_passthrough(captured) -> None:
    """Wateruptake doesn't modify q or dgncur_a; they pass through unchanged."""
    before, _ = captured
    state = _build_state(before)
    new_state = wateruptake(state)
    np.testing.assert_array_equal(np.asarray(new_state["q"]),
                                  np.asarray(state["q"]))
    np.testing.assert_array_equal(np.asarray(new_state["dgncur_a"]),
                                  np.asarray(state["dgncur_a"]))
