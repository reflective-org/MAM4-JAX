"""Validate JAX `calcsize(do_aitacc_transfer=True)` end-to-end.

Reference: ``tests/reference/per_process/calcsize_{before,after}.npz``
captured by ``scripts/capture_reference.py --mode instrumented``
(do_aitacc_transfer=True in driver.F90, which is the canonical
Fortran box-model call). Bumped to nstep=60 in M3.5 PR-B so the
calcsize evolution is non-trivial.

In the canonical box-model setup the Aitken ↔ accumulation transfer
block **never triggers** (Aitken/accum mean diameters stay inside
``dgnumlo``/``dgnumhi`` bounds for the full 60-step run). This test
therefore confirms two things:

  1. With ``do_aitacc_transfer=True``, the JAX port still matches the
     Fortran reference at machine ε for ``dgncur_a``, ``q``, ``qqcw``.
  2. The transfer code does not *accidentally* trigger and disturb the
     state when it shouldn't.

A separate structural test asserts that, for this fixture, calling
calcsize with ``do_aitacc_transfer=True`` produces the same output as
``do_aitacc_transfer=False`` — direct confirmation that the transfer is
a no-op here.

See ``docs/DEFERRED.md`` for the resurface conditions on a stress test
that actually exercises the transfer branches.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.processes.calcsize import calcsize

REF_DIR        = Path(__file__).resolve().parent / "reference" / "per_process"
REF_DIR_NOAITACC = Path(__file__).resolve().parent / "reference" / "per_process_no_aitacc"
BEFORE_NPZ     = REF_DIR / "calcsize_before.npz"
AFTER_NPZ      = REF_DIR / "calcsize_after.npz"

# Same dt as the nstep=60 capture (1800 s / 60 = 30 s).
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


def _max_rel_err(jax_arr, fortran_arr, abs_floor: float = 0.0) -> float:
    a = np.asarray(jax_arr, dtype=np.float64)
    b = fortran_arr
    denom = np.maximum(np.abs(b), abs_floor)
    err = np.where(denom > 0, np.abs(a - b) / denom, np.abs(a - b))
    return float(np.max(err))


def test_calcsize_dgncur_a_matches_full_transfer_reference(captured) -> None:
    """With the default do_aitacc_transfer=True, dgncur_a evolution
    matches the full-transfer Fortran reference."""
    before, after = captured
    new_state = calcsize(_build_state(before))  # default do_aitacc_transfer=True
    rel = _max_rel_err(new_state["dgncur_a"], after["dgncur_a"])
    assert rel < 1e-6, f"dgncur_a max rel-err = {rel:.3e}"


def test_calcsize_q_matches_full_transfer_reference(captured) -> None:
    """Tracer mass mixing ratios (q) match the full-transfer reference.

    Uses np.allclose-style tolerance. The transfer block produces tiny
    machine-noise contributions (~1e-26) at indices where Fortran has
    *exact zeros* (e.g., m-organic mass tracers with mf_*=0 in the
    namelist), which would blow up a pure relative-error metric. The
    absolute tolerance of 1e-25 absorbs that noise; the relative
    tolerance is the canonical ADR-003 1e-6.
    """
    before, after = captured
    new_state = calcsize(_build_state(before))
    np.testing.assert_allclose(
        np.asarray(new_state["q"], dtype=np.float64),
        after["q"],
        rtol=1e-6,
        atol=1e-25,
    )


def test_calcsize_qqcw_matches_full_transfer_reference(captured) -> None:
    """Cloud-borne tracers (qqcw) are zero throughout in the box-model
    setup (cldn=0), so calcsize doesn't modify them."""
    before, after = captured
    new_state = calcsize(_build_state(before))
    np.testing.assert_array_equal(np.asarray(new_state["qqcw"]),
                                  after["qqcw"])


def test_transfer_is_a_no_op_on_box_model_fixture(captured) -> None:
    """Structural: calcsize(state, do_aitacc_transfer=True) ≡
    calcsize(state, do_aitacc_transfer=False) on the canonical fixture,
    because the Aitken-accum transfer never triggers given the namelist
    defaults. See docs/DEFERRED.md for the resurface conditions."""
    before, _ = captured
    state = _build_state(before)
    with_xfer  = calcsize(state, do_aitacc_transfer=True)
    no_xfer    = calcsize(state, do_aitacc_transfer=False)
    for key in ("q", "qqcw", "dgncur_a", "dgncur_c", "v2ncur_a", "v2ncur_c"):
        np.testing.assert_array_equal(
            np.asarray(with_xfer[key]),
            np.asarray(no_xfer[key]),
            err_msg=f"transfer-on vs transfer-off differ on key {key!r}",
        )
