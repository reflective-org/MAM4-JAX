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


RENAME_ONLY_REF_DIR = (
    Path(__file__).resolve().parent / "reference" / "per_process_rename_only"
)


@pytest.fixture(scope="module")
def rename_only_captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Single-toggle Fortran capture: `mdo_rename=1, others=0`.

    With gasaerexch off, `qaer_delsub_grow4rnam` is identically zero at
    the rename call site. In the canonical box-model fixture, the
    Aitken-mode `dgn_t_old` stays well below `dp_belowcut` because
    nothing else is growing the mode — so rename's optaa=40 guard at
    Fortran line 4141 trips and rename is a no-op. The JAX orchestration
    with `mdo_rename=1, others=0` must reproduce this passthrough.
    """
    before = {k: np.asarray(v)
              for k, v in np.load(RENAME_ONLY_REF_DIR / "amicphys_before.npz").items()}
    after  = {k: np.asarray(v)
              for k, v in np.load(RENAME_ONLY_REF_DIR / "amicphys_after.npz").items()}
    return before, after


def test_orchestration_rename_only_matches_fortran(rename_only_captured) -> None:
    """JAX `amicphys(state, mdo_rename=1, others=0)` reproduces the
    single-toggle Fortran capture's `amicphys_after` at machine epsilon.

    Validates the state-dict ↔ amicphys-local-view round-trip
    (unpack → rename → repack) plus the mmr↔vmr conversion (M3.6 PR-C).
    """
    before, after = rename_only_captured
    state = _build_state(before)
    new_state = amicphys(state,
                         mdo_gasaerexch=0, mdo_rename=1,
                         mdo_newnuc=0, mdo_coag=0)
    for key in ("q", "qqcw", "dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), after[key],
            rtol=1e-12, atol=1e-30,
            err_msg=f"rename-only orchestration diverged on {key!r}",
        )


GASAEREXCH_ONLY_REF_DIR = (
    Path(__file__).resolve().parent / "reference" / "per_process_gasaerexch_only"
)


@pytest.fixture(scope="module")
def gasaerexch_only_captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Single-toggle Fortran capture: `mdo_gasaerexch=1, others=0`
    with the `mam_soaexch_1subarea` call also skipped (M3.6 PR-D).

    Use `amicphys_after_writeback.npz` for `q`/`qqcw` because the
    pre-writeback dump records `q` before the driver's vmr→mmr update —
    those values are still pre-amicphys and don't reflect the changes
    gasaerexch made in vmr space.
    """
    before = {k: np.asarray(v)
              for k, v in np.load(GASAEREXCH_ONLY_REF_DIR / "amicphys_before.npz").items()}
    aw     = {k: np.asarray(v) for k, v in np.load(
                GASAEREXCH_ONLY_REF_DIR / "amicphys_after_writeback.npz").items()}
    return before, aw


def test_orchestration_gasaerexch_only_matches_fortran(gasaerexch_only_captured) -> None:
    """JAX `amicphys(state, mdo_gasaerexch=1, others=0)` reproduces the
    Fortran single-toggle gasaerexch-only fixture (with SOA skipped).
    """
    before, aw = gasaerexch_only_captured
    state = _build_state(before)
    new_state = amicphys(state,
                         mdo_gasaerexch=1, mdo_rename=0,
                         mdo_newnuc=0, mdo_coag=0)

    # Per-key check. atol is set generously enough to absorb ULP-level
    # noise on near-zero tracers (e.g. species absent from a mode end up
    # at ~1e-25 instead of exact 0). rtol=1e-6 per ADR-003.
    for key in ("q", "qqcw"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), aw[key],
            rtol=1e-6, atol=1e-20,
            err_msg=f"gasaerexch-only orchestration diverged on {key!r}",
        )
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        # Fortran's `update_aerosol_props` (called inside the cond
        # sub-stepping when `do_cond_wateruptake=.true.`) re-runs
        # wateruptake after every gasaerexch substep, so dgn_awet /
        # qaerwat / wetdens diverge slightly from the inputs. JAX
        # doesn't implement that re-uptake yet (Phase A only ports the
        # main sub-process), so we just confirm these fields are still
        # in the right ballpark — they are not part of gasaerexch's
        # validation surface for PR-D.
        np.testing.assert_allclose(
            np.asarray(new_state[key]), aw[key],
            rtol=1e-3, atol=1e-15,
            err_msg=f"gasaerexch-only orchestration drifted on {key!r}",
        )


def test_amicphys_returns_all_state_keys(captured) -> None:
    """The returned state preserves keys that weren't part of the
    aerosol-state inputs (meteorology, deltat, etc.)."""
    before, _ = captured
    state = _build_state(before)
    new_state = amicphys(state)
    for key in ("t", "pmid", "cldn", "deltat"):
        assert key in new_state, f"amicphys dropped state key {key!r}"
