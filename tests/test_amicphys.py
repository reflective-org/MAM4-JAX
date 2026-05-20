"""Validate the M3.6 PR-A amicphys orchestration shell.

Reference: ``tests/reference/per_process_amicphys_off/amicphys_{before,after}.npz``
captured by ``scripts/capture_reference.py --mode instrumented-amicphys-off``
(namelist with ``mdo_gasaerexch=mdo_rename=mdo_newnuc=mdo_coag=0``).
Under those toggles the Fortran ``modal_aero_amicphys_intr`` is a true
state passthrough — every captured array is bit-exact identical between
the before and after snapshots.

PR-A's JAX shell is correspondingly trivial: the orchestration calls
four sub-process stubs that each return state unchanged. PR-B through
PR-E will replace one stub at a time, and that's when the
``per_process/`` (full-bundle) reference becomes the validation target.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.processes.amicphys import amicphys

REF_DIR    = Path(__file__).resolve().parent / "reference" / "per_process_amicphys_off"
BEFORE_NPZ = REF_DIR / "amicphys_before.npz"
AFTER_NPZ  = REF_DIR / "amicphys_after.npz"

# Box-model constants. Same convention as test_wateruptake / test_calcsize.
T_BOX_MODEL    = 273.0
PMID_BOX_MODEL = 1.0e5
CLDN_BOX_MODEL = 0.0
DELTAT_60      = 30.0   # 1800 s / 60 snapshots


@pytest.fixture(scope="module")
def captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    before = {k: np.asarray(v) for k, v in np.load(BEFORE_NPZ).items()}
    after  = {k: np.asarray(v) for k, v in np.load(AFTER_NPZ).items()}
    return before, after


def _build_state(before: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    nstep, ncol, pver, _ = before["q"].shape
    return {
        "q":           jnp.asarray(before["q"]),
        "qqcw":        jnp.asarray(before["qqcw"]),
        "dgncur_a":    jnp.asarray(before["dgncur_a"]),
        "dgncur_awet": jnp.asarray(before["dgncur_awet"]),
        "qaerwat":     jnp.asarray(before["qaerwat"]),
        "wetdens":     jnp.asarray(before["wetdens"]),
        "t":           jnp.asarray(np.full((nstep, ncol, pver), T_BOX_MODEL)),
        "pmid":        jnp.asarray(np.full((nstep, ncol, pver), PMID_BOX_MODEL)),
        "cldn":        jnp.asarray(np.full((nstep, ncol, pver), CLDN_BOX_MODEL)),
        "deltat":      jnp.asarray(DELTAT_60),
    }


def test_amicphys_all_off_is_passthrough(captured) -> None:
    """With all four mdo_* set to 0, amicphys must return its input
    state unchanged for every key the Fortran captures."""
    before, after = captured
    state = _build_state(before)
    new_state = amicphys(state,
                         mdo_gasaerexch=0, mdo_rename=0,
                         mdo_newnuc=0, mdo_coag=0)
    for key in ("q", "qqcw", "dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_array_equal(
            np.asarray(new_state[key]), after[key],
            err_msg=f"all-mdo-off amicphys disturbed {key!r}",
        )


def test_amicphys_all_on_with_stubs_is_passthrough(captured) -> None:
    """With the default mdo_*=1 the sub-process stubs are still no-ops
    (PR-A scope), so the function should still pass state through.

    This test will start *failing* once PR-B/C/D/E fill in physics — at
    which point we'll switch to validating against
    ``tests/reference/per_process/amicphys_{before,after}.npz``. Until
    then it acts as a tripwire confirming PR-A really hasn't introduced
    any state modification."""
    before, after = captured
    state = _build_state(before)
    new_state = amicphys(state)  # all mdo_* default to 1
    for key in ("q", "qqcw", "dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_array_equal(
            np.asarray(new_state[key]), after[key],
            err_msg=f"default-mdo amicphys disturbed {key!r} (stubs are not no-ops)",
        )


def test_amicphys_returns_all_state_keys(captured) -> None:
    """The returned state preserves keys that weren't part of the
    aerosol-state inputs (meteorology, deltat, etc.)."""
    before, _ = captured
    state = _build_state(before)
    new_state = amicphys(state)
    for key in ("t", "pmid", "cldn", "deltat"):
        assert key in new_state, f"amicphys dropped state key {key!r}"
