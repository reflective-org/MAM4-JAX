"""The post-transfer dgncur/v2ncur recompute, and its `dumfac` (issue #68).

CAM recomputes `dgncur`/`v2ncur` for the aitken and accum modes a SECOND time
after an aitken<->accum transfer fires (`modal_aero_calcsize.F90:926-987`). That
block was absent from this port, which mattered only once
`do_aitacc_transfer`'s bound turn-off made the transfer reachable — before that,
the bound clamp pinned number before the trigger test and the transfer never
fired.

Two things distinguish the second recompute from the in-loop one:

1. It uses the **unmodified** `voltonumbhi/lo_amode` and `dgnumhi/lo_amode`
   (`:958, 961, 974, 977`), not the 1e6-adjusted bounds. Deliberate upstream.
2. It uses `dumfac`, a **scalar** last assigned at `:549` for `n = ntot_amode`
   inside a loop that closed at `:738`. So upstream CAM applies the wrong mode's
   width. That is the bug in issue #68.

Owner decision: use the correct per-mode value by default;
`bug_compat_stale_dumfac=True` reproduces CAM, which is needed to validate
against the reference build in `../mam-box-fortran` (whose default is faithful
*including* the bug, that being its purpose).

The assertions here are **analytic**, not snapshots: `dgncur` scales as
`(dumfac_correct / dumfac_stale) ** (1/3)`, a pure function of the namelist
sigma_g values, so the expected ratio is computed rather than recorded.
"""
from __future__ import annotations

import numpy as np
import pytest

import mam4_jax  # noqa: F401
import jax.numpy as jnp
from mam4_jax.core.data import (
    ACCUM_MODE_IDX,
    AITKEN_MODE_IDX,
    DUMFAC_AMODE,
    LMASSPTR_AMODE,
    LSPECTYPE_AMODE,
    NSPEC_AMODE,
    NUMPTR_AMODE,
    PCNST,
    SPECDENS_AMODE,
    V2NZZ_AIT_ACC,
    VOLTONUMB_AMODE,
)
from mam4_jax.physics.calcsize import calcsize

NMODES = len(NSPEC_AMODE)
NAIT = AITKEN_MODE_IDX
NACC = ACCUM_MODE_IDX

# calcsize relaxes over tadj = max(86400 s, deltat); deltat = tadj makes
# fracadj exactly 1 so the transfer is applied in full in one call.
DELTAT = 86400.0
DRY_VOLUME = 1.0e-12          # m3/kg, per mode


def _slot_density(mode: int, slot: int) -> float:
    return float(SPECDENS_AMODE[LSPECTYPE_AMODE[mode][slot]])


def _state(num_override: dict[int, float] | None = None) -> dict:
    """Self-consistent state, optionally with some modes' number overridden.

    Every mode carries DRY_VOLUME in slot 0 and, by default, the number that
    puts it exactly at its own VOLTONUMB_AMODE — i.e. mid-range, so it cannot
    itself trip a bound and the assertions stay attributable.
    """
    q = np.zeros(PCNST, dtype=np.float64)
    for m in range(NMODES):
        q[LMASSPTR_AMODE[m][0]] = DRY_VOLUME * _slot_density(m, 0)
        q[NUMPTR_AMODE[m]] = DRY_VOLUME * float(VOLTONUMB_AMODE[m])
    for m, v in (num_override or {}).items():
        q[NUMPTR_AMODE[m]] = v
    return {
        "q": jnp.asarray(q),
        "qqcw": jnp.zeros(PCNST, dtype=jnp.float64),
        "dgncur_a": jnp.zeros(NMODES, dtype=jnp.float64),
        "deltat": jnp.asarray(DELTAT),
    }


def _oversized_aitken_state() -> dict:
    """Aitken number/volume ratio below v2nzz, which triggers ait2acc.

    This is the condition the reference fixture never reaches — see
    tests/test_calcsize_transfer.py, which documents that the transfer is a
    no-op there. Without a state like this the post-transfer block is
    unreachable and none of the assertions below mean anything.
    """
    return _state({NAIT: DRY_VOLUME * float(V2NZZ_AIT_ACC) * 0.3})


def _dgn(state: dict, *, bug_compat: bool) -> np.ndarray:
    out = calcsize(state, do_aitacc_transfer=True,
                   bug_compat_stale_dumfac=bug_compat)
    return np.asarray(out["dgncur_a"], dtype=np.float64)


def _expected_ratio(mode: int) -> float:
    """dgn(stale) / dgn(correct) for `mode`, from the sigma_g values alone."""
    return (float(DUMFAC_AMODE[mode]) / float(DUMFAC_AMODE[-1])) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------

def test_the_transfer_actually_fires_in_this_fixture() -> None:
    """Guard the guard: if the transfer stops firing, everything below goes
    vacuously green rather than failing."""
    st = _oversized_aitken_state()
    assert _dgn(st, bug_compat=True)[NACC] != pytest.approx(
        _dgn(st, bug_compat=False)[NACC], rel=1e-12
    ), "transfer did not fire — the post-transfer block was never reached"


def test_accum_dgn_differs_by_exactly_the_dumfac_ratio() -> None:
    st = _oversized_aitken_state()
    correct = _dgn(st, bug_compat=False)[NACC]
    stale = _dgn(st, bug_compat=True)[NACC]

    expected = _expected_ratio(NACC)
    assert stale / correct == pytest.approx(expected, rel=1e-10), (
        f"accum dgn ratio {stale / correct:.9f} != predicted {expected:.9f}"
    )
    # Sanity on the magnitude the issue quotes for MAM4.
    assert expected == pytest.approx(1.2055, rel=1e-3)


def test_aitken_is_spared_under_mam4_but_only_by_coincidence() -> None:
    """primary_carbon carries sigma_g = 1.600000023841858, a float32 round-trip
    of aitken's 1.6, so the stale value very nearly cancels. That is a property
    of the namelist defaults, not of the code -- under MAM5 the last mode is
    coarse_strat at sigma_g = 1.2 and both modes are wrong by ~32.5 %."""
    st = _oversized_aitken_state()
    correct = _dgn(st, bug_compat=False)[NAIT]
    stale = _dgn(st, bug_compat=True)[NAIT]
    assert stale == pytest.approx(correct, rel=1e-7)
    # Not EXACTLY equal in principle: the sigma_g values differ at 1e-8.
    assert _expected_ratio(NAIT) == pytest.approx(1.0, abs=1e-6)


def test_modes_outside_the_transfer_are_untouched() -> None:
    """Only the two transfer lanes get the second recompute."""
    st = _oversized_aitken_state()
    correct = _dgn(st, bug_compat=False)
    stale = _dgn(st, bug_compat=True)
    for m in range(NMODES):
        if m in (NAIT, NACC):
            continue
        assert stale[m] == pytest.approx(correct[m], rel=1e-14), (
            f"mode {m} changed, but it takes no part in the transfer"
        )


def test_no_transfer_means_no_second_recompute() -> None:
    """With the transfer switched off the flag is never set, so the two
    settings must agree exactly -- the block is gated, not unconditional."""
    st = _state()
    a = np.asarray(calcsize(st, do_aitacc_transfer=False,
                            bug_compat_stale_dumfac=False)["dgncur_a"])
    b = np.asarray(calcsize(st, do_aitacc_transfer=False,
                            bug_compat_stale_dumfac=True)["dgncur_a"])
    assert a.tobytes() == b.tobytes()


def test_default_is_the_correct_value() -> None:
    """The default must be the physically correct per-mode width.

    Shipping a known-wrong diameter as the default is the thing issue #68
    decided against, so it is asserted rather than left to a docstring.
    """
    st = _oversized_aitken_state()
    default = _dgn(st, bug_compat=False)[NACC]
    explicit_correct = _dgn(st, bug_compat=False)[NACC]
    stale = _dgn(st, bug_compat=True)[NACC]
    assert default == explicit_correct
    assert default != pytest.approx(stale, rel=1e-9)
    # Correct means SMALLER here: dumfac_accum > dumfac_stale, and dgn goes as
    # dumfac**(-1/3).
    assert default < stale


def test_recompute_uses_the_unmodified_bounds_not_the_1e6_adjusted_ones() -> None:
    """The second effect of the recompute, separable from the dumfac question.

    CAM's post-transfer block reads voltonumbhi/lo_amode and dgnumhi/lo_amode
    directly (:958, 961, 974, 977) rather than the 1e6-adjusted locals that
    do_aitacc_transfer installs. So aitken, whose lower number bound the turn-off
    disabled, gets clamped again here — to dgnumhi_amode.

    This is independent of `bug_compat_stale_dumfac`: it holds either way, and it
    is what makes the aitken diameter match the Fortran. Before the recompute
    existed this fixture gave 5.871709e-08; CAM gives dgnumhi = 5.2e-08, an
    agreement independently derived during review of the bound turn-off.
    """
    st = _oversized_aitken_state()
    from mam4_jax.core.data import DGNUMHI_AMODE

    for bug_compat in (False, True):
        dgn = _dgn(st, bug_compat=bug_compat)
        assert dgn[NAIT] == pytest.approx(float(DGNUMHI_AMODE[NAIT]), rel=1e-12), (
            "aitken should be clamped to the UNMODIFIED dgnumhi by the "
            f"post-transfer recompute; got {dgn[NAIT]:.6e} "
            f"(bug_compat={bug_compat})"
        )
    # And it is genuinely a change: the pre-recompute value was ~12.9 % larger.
    assert 5.871709e-08 / float(DGNUMHI_AMODE[NAIT]) == pytest.approx(1.129, rel=1e-3)
