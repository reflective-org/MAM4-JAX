"""The ``do_aitacc_transfer`` size-bound turn-off in calcsize.

The Fortran deliberately disables the per-mode number bounds on the two modes
that take part in the aitken<->accum transfer, so that the transfer block --
not the bound clamp -- decides their number. Identical in both reference trees:

    CAM   modal_aero_calcsize.F90:556-561
    E3SM  box_model_utils/modal_aero_calcsize.F90:756-761

with E3SM's own comment at :738-741 stating the intent:

    for n=nacc, multiply v2nyy by 1.0e6 to effectively turn off the
        adjustment when number is too big (size is too small)
    for n=nait, divide   v2nxx by 1.0e6 to effectively turn off the
        adjustment when number is too small (size is too big)

WHY THIS FILE EXISTS SEPARATELY. ``test_calcsize_transfer.py`` already runs
``do_aitacc_transfer=True`` against a Fortran capture and passed both with and
without the turn-off -- the captured regime simply never drives either mode past
its bound, so the branch was unreachable by the existing suite. These tests
construct states that DO cross the bounds, which is what makes the branch
observable at all.

Each test is written as a differential against ``do_aitacc_transfer=False``:
with the turn-off present the two settings must disagree, and without it they
collapse to the same clamped answer. That difference is the whole behaviour.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.core.data import (
    ACCUM_MODE_IDX,
    AITKEN_MODE_IDX,
    LMASSPTR_AMODE,
    NSPEC_AMODE,
    NUMPTR_AMODE,
    PCNST,
    SPECDENS_AMODE,
    LSPECTYPE_AMODE,
    VOLTONUMB_AMODE,
    VOLTONUMBHI_AMODE,
    VOLTONUMBLO_AMODE,
)
from mam4_jax.physics.calcsize import calcsize

# calcsize relaxes each adjustment over tadj = max(86400 s, deltat), so with a
# 30 s step only ~3.5e-4 of the correction is applied and nothing visibly
# clamps. Using deltat = tadj makes fracadj exactly 1, i.e. the bound is applied
# in full in a single call. Chosen so these tests probe the bound itself rather
# than the relaxation rate.
DELTAT = 86400.0

# The turn-off is a factor of 1e6 and the relaxed bounds are frelaxadj = 27
# looser than the hard ones. Overshooting by 1e3 therefore sits between the two:
# it crosses the ordinary (relaxed) bound but stays well inside the turned-off
# one, so the two do_aitacc_transfer settings must disagree.
OVERSHOOT = 1.0e3


def _slot_density(mode: int, slot: int) -> float:
    """Bulk density of the species occupying (mode, slot)."""
    return float(SPECDENS_AMODE[LSPECTYPE_AMODE[mode][slot]])


def _state_with_mode_number(mode: int, dry_volume: float,
                            number: float) -> dict[str, jnp.ndarray]:
    """A minimal state whose `mode` carries `dry_volume` m3/kg and `number` #/kg.

    Every other mode is given a self-consistent number at its own
    ``VOLTONUMB_AMODE``, so it sits mid-range and cannot itself trip a bound --
    that keeps the assertions attributable to `mode` alone.
    """
    q = np.zeros(PCNST, dtype=np.float64)

    for m in range(len(NSPEC_AMODE)):
        vol = dry_volume if m == mode else 1.0e-12
        # Put the whole dry volume in slot 0 of the mode.
        idx = LMASSPTR_AMODE[m][0]
        q[idx] = vol * _slot_density(m, 0)
        q[NUMPTR_AMODE[m]] = (number if m == mode
                              else vol * float(VOLTONUMB_AMODE[m]))

    return {
        "q":        jnp.asarray(q),
        "qqcw":     jnp.zeros(PCNST, dtype=jnp.float64),
        "dgncur_a": jnp.zeros(len(NSPEC_AMODE), dtype=jnp.float64),
        "deltat":   jnp.asarray(DELTAT),
    }


def _number_out(state: dict[str, jnp.ndarray], mode: int, *,
                transfer: bool) -> float:
    out = calcsize(state, do_aitacc_transfer=transfer)
    return float(np.asarray(out["q"])[NUMPTR_AMODE[mode]])


# --------------------------------------------------------------------------
# aitken: v2nxx is the LOWER bound on number. Too few particles for the volume
# (i.e. the mode is too big) is what the turn-off is meant to leave alone.
# --------------------------------------------------------------------------
def test_aitken_lower_number_bound_is_disabled_by_transfer() -> None:
    dry_volume = 1.0e-12
    # 1/1000th of the smallest number the bound would permit.
    starved = dry_volume * float(VOLTONUMBHI_AMODE[AITKEN_MODE_IDX]) / OVERSHOOT
    state = _state_with_mode_number(AITKEN_MODE_IDX, dry_volume, starved)

    clamped = _number_out(state, AITKEN_MODE_IDX, transfer=False)
    free    = _number_out(state, AITKEN_MODE_IDX, transfer=True)

    # Without the transfer the bound bites and pulls number UP toward v2nxx.
    assert clamped > starved * 10.0, (
        "expected the lower number bound to clamp when do_aitacc_transfer=False; "
        f"got {clamped:.6e} from a starved {starved:.6e}"
    )
    # With the transfer the bound is turned off, so number is left far below it.
    assert free < clamped / 10.0, (
        "do_aitacc_transfer=True must disable the aitken lower number bound "
        f"(CAM calcsize:557); got free={free:.6e} vs clamped={clamped:.6e}"
    )


# --------------------------------------------------------------------------
# accum: v2nyy is the UPPER bound on number. Too many particles for the volume
# (i.e. the mode is too small) is the case turned off here.
# --------------------------------------------------------------------------
def test_accum_upper_number_bound_is_disabled_by_transfer() -> None:
    dry_volume = 1.0e-12
    flooded = dry_volume * float(VOLTONUMBLO_AMODE[ACCUM_MODE_IDX]) * OVERSHOOT
    state = _state_with_mode_number(ACCUM_MODE_IDX, dry_volume, flooded)

    clamped = _number_out(state, ACCUM_MODE_IDX, transfer=False)
    free    = _number_out(state, ACCUM_MODE_IDX, transfer=True)

    # With the bound in force, number is pinned to the hard bound drv*v2nyy --
    # exactly the 1e3 overshoot removed.
    hard_bound = dry_volume * float(VOLTONUMBLO_AMODE[ACCUM_MODE_IDX])
    assert clamped == pytest.approx(hard_bound, rel=1e-9), (
        "do_aitacc_transfer=False should pin accum number to the hard upper "
        f"bound {hard_bound:.6e}; got {clamped:.6e}"
    )

    # With the turn-off, the bound no longer decides. What decides instead is
    # the transfer block: an accum mode this over-populated has far-too-small
    # particles, so acc2ait ships them to aitken and accum empties. That is the
    # documented intent -- "the transfer block, not the bound clamp, decides" --
    # so the assertion is that the two settings DISAGREE, not that number ends
    # up higher. Without the turn-off both paths return the bound and this
    # fails.
    assert abs(free - clamped) > clamped / 10.0, (
        "do_aitacc_transfer=True must disable the accum upper number bound "
        f"(CAM calcsize:558); got free={free:.6e} vs clamped={clamped:.6e}"
    )
    assert free < clamped, (
        "expected acc2ait to drain the over-populated accum mode once the "
        f"bound is off; got free={free:.6e} vs clamped={clamped:.6e}"
    )


# --------------------------------------------------------------------------
# The turn-off is scoped to those two modes only.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", [
    m for m in range(len(NSPEC_AMODE))
    if m not in (AITKEN_MODE_IDX, ACCUM_MODE_IDX)
])
def test_other_modes_keep_their_bounds(mode: int) -> None:
    """Coarse and primary-carbon must clamp identically either way.

    Guards the plausible over-fix of applying the 1e6 factors to every mode.
    """
    dry_volume = 1.0e-12
    flooded = dry_volume * float(VOLTONUMBLO_AMODE[mode]) * OVERSHOOT
    state = _state_with_mode_number(mode, dry_volume, flooded)

    off = _number_out(state, mode, transfer=False)
    on  = _number_out(state, mode, transfer=True)

    assert on == pytest.approx(off, rel=1e-12), (
        f"mode {mode} must be unaffected by do_aitacc_transfer; "
        f"got {on:.6e} vs {off:.6e}"
    )
    assert on < flooded / 10.0, (
        f"mode {mode} upper number bound should still clamp; got {on:.6e}"
    )
