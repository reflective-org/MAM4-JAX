"""Validate the pcarbon → accum aging port (mam_pcarbon_aging_1subarea).

Three layers:

1. Unit tests of ``_mam_pcarbon_aging_1subarea`` against hand-computed
   monolayer-criterion values (self-contained, machine precision).
2. A sulfur-conservation test showing aging closes the repack leak:
   without it, so4/soa condensed or coagulated onto the pcarbon mode is
   silently dropped when the dense ``qaer`` view is scattered back to
   ``q`` (the pcarbon ``LMAP_AER`` row maps only pom/bc/mom).
3. End-to-end parity of ``run_step`` / ``run_timesteps`` (aging ON, the
   default) against the CANONICAL Fortran bundle
   ``tests/reference/per_process/`` — which, unlike every
   ``*_no_pcarbon_aging`` fixture, was captured from the unpatched box
   model with aging active (see ``tests/reference/SCHEMA.md``). Bars
   mirror ``tests/test_driver.py``'s ADR-015 coarse-dt framing (dt=30s).
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default
from mam4_jax.core import data
from mam4_jax.coupling.amicphys import (
    _FAC_VOLSFC_PCARBON,
    _PCAGING,
    _PCARBON_CORE_MASK,
    _mam_pcarbon_aging_1subarea,
    _unpack_state_to_amicphys_view,
    amicphys,
    configure_pcarbon_aging,
)
from mam4_jax.driver import run_step, run_timesteps

REF_DIR = (Path(__file__).resolve().parent / "reference" / "per_process")

T_BOX_MODEL    = 273.0
PMID_BOX_MODEL = 1.0e5
CLDN_BOX_MODEL = 0.0
DELTAT_60      = 30.0
ZMID_BOX_MODEL = 3.0e3
PBLH_BOX_MODEL = 1.1e3
RH_BOX_MODEL   = 0.9

NPCA = data.AMICPHYS_NPCA
NACC = data.ACCUM_MODE_IDX
ISO4 = data.AMICPHYS_IAER_SO4
ISOA = data.AMICPHYS_IAER_SOA
IPOM = data.AMICPHYS_IAER_POM
IBC  = 3   # amicphys iaer order: soa, so4, pom, bc, ncl, dst, mom
IMOM = 6


def _synthetic_view():
    """A single-cell amicphys-local view with a thin shell on pcarbon."""
    qnum = jnp.asarray([2.0e9, 5.0e9, 1.0e6, 3.0e9])
    qaer = jnp.zeros((data.AMICPHYS_NAER, data.NTOT_AMODE), dtype=jnp.float64)
    # pcarbon: a carbon core plus a sub-monolayer so4/soa shell.
    qaer = qaer.at[IPOM, NPCA].set(5.0e-11)
    qaer = qaer.at[IBC,  NPCA].set(1.0e-10)
    qaer = qaer.at[IMOM, NPCA].set(2.0e-13)
    qaer = qaer.at[ISO4, NPCA].set(4.0e-14)
    qaer = qaer.at[ISOA, NPCA].set(6.0e-14)
    # Non-empty accum so the destination add is visible.
    qaer = qaer.at[ISO4, NACC].set(1.0e-11)
    qaer = qaer.at[IPOM, NACC].set(2.0e-11)
    dgn_a = jnp.asarray(data.DGNUM_AMODE)
    return qnum, qaer, dgn_a


def _expected_xferfrac(qaer, dgn_a) -> float:
    """The F90:5168-5239 criterion, computed independently in numpy."""
    vol_shell = (float(qaer[ISO4, NPCA]) * data.FAC_M2V_AER[ISO4]
                 + float(qaer[ISOA, NPCA]) * data.FAC_M2V_EQVHYG_AER[ISOA])
    vol_core = sum(
        float(qaer[i, NPCA]) * data.FAC_M2V_AER[i]
        for i in range(data.AMICPHYS_NAER) if _PCARBON_CORE_MASK[i])
    dr = _PCAGING["n_so4_monolayers"] * data.DR_SO4_MONOLAYER
    tmp1 = vol_shell * float(dgn_a[NPCA]) * _FAC_VOLSFC_PCARBON
    tmp2 = 6.0 * dr * vol_core
    xmax = 1.0 - 10.0 * np.finfo(np.float64).eps
    return xmax if tmp1 >= tmp2 else min(tmp1 / tmp2, xmax)


def test_aging_matches_hand_computed_criterion() -> None:
    """Sub-saturation case: the core species and number move by exactly
    the monolayer-criterion fraction; the shell species move entirely."""
    qnum, qaer, dgn_a = _synthetic_view()
    xfer = _expected_xferfrac(qaer, dgn_a)
    assert 0.0 < xfer < 1.0, "fixture must exercise the sub-saturation branch"

    new_qnum, new_qaer = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a)

    for iaer in (IPOM, IBC, IMOM):
        moved = float(qaer[iaer, NPCA]) * xfer
        np.testing.assert_allclose(
            float(new_qaer[iaer, NPCA]), float(qaer[iaer, NPCA]) - moved,
            rtol=1e-14, err_msg=f"core species {iaer} source")
        np.testing.assert_allclose(
            float(new_qaer[iaer, NACC]), float(qaer[iaer, NACC]) + moved,
            rtol=1e-14, err_msg=f"core species {iaer} destination")
    # Shell species transfer wholesale, landing pcarbon on exact zero
    # (mirrors the Fortran's explicit zeroing, F90:5267).
    for iaer in (ISO4, ISOA):
        assert float(new_qaer[iaer, NPCA]) == 0.0
    np.testing.assert_allclose(
        float(new_qaer[ISO4, NACC]),
        float(qaer[ISO4, NACC]) + float(qaer[ISO4, NPCA]), rtol=1e-14)
    # Number moves by the same fraction as the core.
    np.testing.assert_allclose(
        float(new_qnum[NPCA]), float(qnum[NPCA]) * (1.0 - xfer), rtol=1e-12)
    np.testing.assert_allclose(
        float(new_qnum[NACC]), float(qnum[NACC]) + float(qnum[NPCA]) * xfer,
        rtol=1e-12)
    # Other modes untouched.
    np.testing.assert_array_equal(np.asarray(new_qnum[1:3]),
                                  np.asarray(qnum[1:3]))


def test_aging_conserves_species_and_number() -> None:
    qnum, qaer, dgn_a = _synthetic_view()
    new_qnum, new_qaer = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a)
    np.testing.assert_allclose(
        np.asarray(new_qaer).sum(axis=-1), np.asarray(qaer).sum(axis=-1),
        rtol=1e-14, err_msg="per-species mass over modes must be conserved")
    np.testing.assert_allclose(
        float(jnp.sum(new_qnum)), float(jnp.sum(qnum)), rtol=1e-14)


def test_aging_saturates_at_one_minus_eps() -> None:
    """A shell far past the monolayer requirement transfers the maximum
    fraction 1-10eps — never exactly all of the mode (F90:5231)."""
    qnum, qaer, dgn_a = _synthetic_view()
    qaer = qaer.at[ISO4, NPCA].set(1.0e-6)     # enormous shell
    new_qnum, new_qaer = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a)
    xmax = 1.0 - 10.0 * np.finfo(np.float64).eps
    # Same arithmetic form as the code (q - q*xmax): the analytically
    # equivalent q*(1-xmax) differs by cancellation quantization at
    # q ~ 3e9, so compare in the computed form with a few-ULP budget.
    expected = float(qnum[NPCA]) - float(qnum[NPCA]) * xmax
    np.testing.assert_allclose(
        float(new_qnum[NPCA]), expected,
        rtol=0.0, atol=4.0 * np.spacing(float(qnum[NPCA])))
    assert float(new_qnum[NPCA]) > 0.0
    assert float(new_qaer[IBC, NPCA]) > 0.0


def test_aging_zero_core_is_finite_and_saturated() -> None:
    """No core mass ⇒ tmp2 = 0 ⇒ the saturated branch, with no NaN/inf
    from the masked division (the #558 double-where guard)."""
    qnum, qaer, dgn_a = _synthetic_view()
    for iaer in (IPOM, IBC, IMOM):
        qaer = qaer.at[iaer, NPCA].set(0.0)
    new_qnum, new_qaer = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a)
    assert np.all(np.isfinite(np.asarray(new_qaer)))
    assert np.all(np.isfinite(np.asarray(new_qnum)))
    xmax = 1.0 - 10.0 * np.finfo(np.float64).eps
    expected = float(qnum[NPCA]) - float(qnum[NPCA]) * xmax
    np.testing.assert_allclose(
        float(new_qnum[NPCA]), expected,
        rtol=0.0, atol=4.0 * np.spacing(float(qnum[NPCA])))


def test_aging_batched_matches_single_cell() -> None:
    """The routine is broadcasting-native: a (ncol, pver) batch of the
    same cell must reproduce the single-cell answer per point."""
    qnum, qaer, dgn_a = _synthetic_view()
    s_num, s_aer = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a)
    shape = (3, 2)

    def tile(a):
        return jnp.broadcast_to(a, shape + a.shape)

    b_num, b_aer = _mam_pcarbon_aging_1subarea(tile(qnum), tile(qaer),
                                               tile(dgn_a))
    for c in range(shape[0]):
        for p in range(shape[1]):
            np.testing.assert_array_equal(np.asarray(b_num[c, p]),
                                          np.asarray(s_num))
            np.testing.assert_array_equal(np.asarray(b_aer[c, p]),
                                          np.asarray(s_aer))


def test_default_threshold_is_the_amicphys_reference_value() -> None:
    """3.0 — what the amicphys path actually receives via phys_control
    (modal_aero_initialize_data.F90:417,585 → modal_aero_amicphys_init;
    box_model_utils/phys_control.F90:26). NOT 8.0: that constant
    (modal_aero_gasaerexch.F90:37) feeds the legacy modal_aero_coag
    aging path (modal_aero_coag.F90:87), a different code path. A
    changed default silently un-reproduces every canonical fixture (the
    60-step trajectory moves ~434% on pcarbon tracers at 8.0)."""
    assert _PCAGING["n_so4_monolayers"] == 3.0


def test_configure_pcarbon_aging_threshold() -> None:
    """A larger monolayer requirement must age a smaller fraction; the
    per-call override and the process-global configure hook must agree
    (the override exists so hosts with several differently-configured
    instances are not order-dependent — jax-gcm#726 review)."""
    qnum, qaer, dgn_a = _synthetic_view()
    n1_num, _ = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a,
                                            n_so4_monolayers=1.0)  # HAMMOZ
    n8_num, _ = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a,
                                            n_so4_monolayers=8.0)  # E3SM
    aged_1 = float(qnum[NPCA] - n1_num[NPCA])
    aged_8 = float(qnum[NPCA] - n8_num[NPCA])
    assert aged_1 > aged_8 > 0.0
    np.testing.assert_allclose(aged_1, 8.0 * aged_8, rtol=1e-10)
    # The global configure hook is the default when no override is given.
    saved = _PCAGING["n_so4_monolayers"]
    try:
        configure_pcarbon_aging(n_so4_monolayers=1.0)
        g1_num, _ = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a)
    finally:
        configure_pcarbon_aging(n_so4_monolayers=saved)
    np.testing.assert_array_equal(np.asarray(g1_num), np.asarray(n1_num))


# ---------------------------------------------------------------------------
# Leak closure: sulfur through the full amicphys call
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def per_process() -> dict[str, dict[str, np.ndarray]]:
    return {
        tag: {k: np.asarray(v)
              for k, v in np.load(REF_DIR / f"{tag}.npz").items()}
        for tag in ("calcsize_before", "amicphys_before",
                    "amicphys_after_writeback")
    }


def _build_state(snapshot: dict[str, np.ndarray], step: int):
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


def _total_sulfur(state) -> float:
    """H2SO4 gas + so4 across all modes, in the amicphys-local mol/mol
    view (1 S atom per molecule of either, so the sum is conserved by
    condensation, nucleation, coagulation, rename, and aging)."""
    qgas, qaer, _, _ = _unpack_state_to_amicphys_view(state)
    return float(jnp.sum(qgas[..., 1]) + jnp.sum(qaer[..., ISO4, :]))


def test_aging_closes_the_pcarbon_sulfur_leak(per_process) -> None:
    """Without aging, so4 condensed/coagulated onto pcarbon is dropped
    at the LMAP_AER repack (no pcnst slot) — a real per-step sulfur
    sink. With aging (the default) sulfur closes to the gas-production
    source term."""
    state = _build_state(per_process["amicphys_before"], step=0)
    s0 = _total_sulfur(state)
    # The gasaerexch stub production (configure_gas_netprod default,
    # driver.F90:1248) is the single legitimate source in the call.
    from mam4_jax.coupling.amicphys import _GAS_NETPROD
    prod = _GAS_NETPROD["h2so4"] * float(state["deltat"])

    with_aging = amicphys(state)
    without    = amicphys(state, mdo_pcarbonaging=0)

    s_on  = _total_sulfur(with_aging)
    s_off = _total_sulfur(without)

    np.testing.assert_allclose(
        s_on, s0 + prod, rtol=1e-9,
        err_msg="with aging, sulfur must close to the production term")
    leak = (s0 + prod) - s_off
    assert leak > 0.0, "aging off must reproduce the repack leak"
    residual = abs(s_on - (s0 + prod))
    assert leak > 100.0 * max(residual, 1e-30), (
        f"leak ({leak:.3e}) should dwarf the aging-on residual "
        f"({residual:.3e}); otherwise this test proves nothing")


# ---------------------------------------------------------------------------
# End-to-end parity vs the CANONICAL (aging-on) Fortran bundle
# ---------------------------------------------------------------------------

def test_run_step_matches_canonical_fortran_with_aging(per_process) -> None:
    """``run_step`` with the faithful default (aging ON) reproduces the
    canonical ``per_process/`` bundle — the unpatched Fortran box model.
    Bars are test_driver.py's ADR-015 coarse-dt values (dt=30s): the
    diffrax soaexch offset dominates, not the aging port."""
    ic = _build_state(per_process["calcsize_before"], step=0)
    # Plain defaults: the package default (3.0) IS the amicphys-path
    # reference value the capture was made with (phys_control →
    # modal_aero_amicphys_init) — defaults reproduce the reference.
    new_state = run_step(ic)
    target = per_process["amicphys_after_writeback"]
    for key in ("q", "qqcw"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), target[key][0],
            rtol=5e-2, atol=1e-20,
            err_msg=f"aging-on 1-step diverged on {key!r}")
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), target[key][0],
            rtol=5e-3, atol=1e-15,
            err_msg=f"aging-on 1-step drifted on {key!r}")


def test_60_step_trajectory_matches_canonical_fortran(per_process) -> None:
    """60-step trajectory against the canonical bundle. The no-aging
    companion of this test (test_driver.py) passes at the same bars
    against the patched build; SCHEMA.md records ~20% divergence between
    the two Fortran builds on Aitken/pcarbon tracers, so passing BOTH at
    5% is a real constraint on the aging port, not a tautology."""
    ic = _build_state(per_process["calcsize_before"], step=0)
    traj = run_timesteps(ic, n_steps=60)   # defaults reproduce the reference
    target = per_process["amicphys_after_writeback"]
    # The two GAS slots (SOAG pcnst 9, H2SO4 pcnst 6) carry the known
    # diffrax soaexch structural offset (ADR-015 coarse-dt regime;
    # dt=30s is outside the gated dt<=5s window), which aging slightly
    # amplifies by feeding the wholesale-transferred pcm soa back into
    # the soaexch equilibrium. Measured worst: soag_gas 5.3% at step 59.
    # Every AEROSOL tracer — including the aging-affected pcm/accum
    # bc/pom and numbers — measured <= 2.1%, so the aerosol bar stays at
    # the companion test's 5%.
    gas_slots = np.asarray(data.LMAP_GAS)
    aer_mask = np.ones(target["q"].shape[-1], dtype=bool)
    aer_mask[gas_slots] = False
    j_q, f_q = np.asarray(traj["q"]), target["q"]
    np.testing.assert_allclose(
        j_q[..., aer_mask], f_q[..., aer_mask],
        rtol=5e-2, atol=1e-20,
        err_msg="aging-on 60-step trajectory diverged on aerosol tracers")
    np.testing.assert_allclose(
        j_q[..., gas_slots], f_q[..., gas_slots],
        rtol=7.5e-2, atol=1e-20,
        err_msg="aging-on 60-step trajectory diverged on gas tracers")
    np.testing.assert_allclose(
        np.asarray(traj["qqcw"]), target["qqcw"],
        rtol=5e-2, atol=1e-20,
        err_msg="aging-on 60-step trajectory diverged on 'qqcw'")
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_allclose(
            np.asarray(traj[key]), target[key],
            rtol=5e-3, atol=1e-15,
            err_msg=f"aging-on 60-step trajectory drifted on {key!r}")


# ---------------------------------------------------------------------------
# AmicphysParams: the differentiable / sweep-safe parameter path
# ---------------------------------------------------------------------------

def test_params_override_matches_configured_global(per_process) -> None:
    """AmicphysParams(n_so4_monolayers=x) must produce exactly the state
    the process-global configure path produces at the same x."""
    from mam4_jax.coupling.amicphys import AmicphysParams
    state = _build_state(per_process["amicphys_before"], step=0)

    via_params = amicphys(state,
                          params=AmicphysParams(n_so4_monolayers=1.0))
    saved = _PCAGING["n_so4_monolayers"]
    try:
        configure_pcarbon_aging(n_so4_monolayers=1.0)
        via_global = amicphys(state)
    finally:
        configure_pcarbon_aging(n_so4_monolayers=saved)
    for key in ("q", "qqcw", "qaerwat"):
        np.testing.assert_array_equal(
            np.asarray(via_params[key]), np.asarray(via_global[key]),
            err_msg=f"params vs configured-global mismatch on {key!r}")


def test_threshold_gradient_finite_and_signed() -> None:
    """d(aged-into-accum BC)/d(n_so4_monolayers) through the aging
    routine: finite and NEGATIVE (a thicker required coating ages less)
    in the sub-saturation regime."""
    import jax
    qnum, qaer, dgn_a = _synthetic_view()

    def aged_bc(n):
        _, new_qaer = _mam_pcarbon_aging_1subarea(
            qnum, qaer, dgn_a, n_so4_monolayers=n)
        return new_qaer[IBC, NACC]

    g = jax.grad(aged_bc)(3.0)
    assert np.isfinite(float(g))
    assert float(g) < 0.0
    # And zero in the saturated regime (whole mode already ages): the
    # criterion clamps, so the threshold stops mattering.
    qaer_sat = qaer.at[ISO4, NPCA].set(1.0e-6)

    def aged_bc_sat(n):
        _, new_qaer = _mam_pcarbon_aging_1subarea(
            qnum, qaer_sat, dgn_a, n_so4_monolayers=n)
        return new_qaer[IBC, NACC]

    assert float(jax.grad(aged_bc_sat)(3.0)) == 0.0


def test_full_amicphys_gradient_wrt_params(per_process) -> None:
    """Reverse-mode through the FULL amicphys call w.r.t. both live
    AmicphysParams fields: finite, and correctly signed where the sign
    is unambiguous (more H2SO4 production ⇒ more accum sulfate)."""
    import jax
    from mam4_jax.coupling.amicphys import AmicphysParams
    state = _build_state(per_process["amicphys_before"], step=0)
    acc_so4_slot = int(data.LMAP_AER[data.ACCUM_MODE_IDX][ISO4])

    def accum_so4(n, prod):
        out = amicphys(state, params=AmicphysParams(
            n_so4_monolayers=n, qgas_netprod_h2so4=prod))
        return jnp.sum(out["q"][..., acc_so4_slot])

    gn, gp = jax.grad(accum_so4, argnums=(0, 1))(3.0, 1.0e-16)
    assert np.isfinite(float(gn)), "d/d(n_so4_monolayers) non-finite"
    assert np.isfinite(float(gp)), "d/d(qgas_netprod_h2so4) non-finite"
    assert float(gp) > 0.0, "more H2SO4 production must add accum so4"


def test_run_step_params_is_traced_not_static(per_process) -> None:
    """Two different threshold VALUES must reuse one compiled run_step —
    the whole point of params being a traced pytree (a calibration sweep
    must not recompile per value)."""
    from mam4_jax.coupling.amicphys import AmicphysParams
    from mam4_jax.driver import _run_step_jit   # the jitted inner; run_step
    # is an unjitted validation wrapper (clear-sky guard), so the cache
    # bookkeeping lives one level down.
    ic = _build_state(per_process["calcsize_before"], step=0)
    _run_step_jit.clear_cache()
    out3 = run_step(ic, AmicphysParams(n_so4_monolayers=jnp.asarray(3.0)))
    size_after_first = _run_step_jit._cache_size()
    out8 = run_step(ic, AmicphysParams(n_so4_monolayers=jnp.asarray(8.0)))
    assert _run_step_jit._cache_size() == size_after_first, (
        "a new threshold VALUE triggered a recompile — params must be "
        "traced, not static")
    # And the values genuinely differ (the sweep measures something).
    assert not np.allclose(np.asarray(out3["q"]), np.asarray(out8["q"]))


def test_negative_threshold_equals_zero_never_reverses() -> None:
    """A negative threshold must behave exactly like 0 (instant full
    aging via the saturated branch) — never a reversed/negative
    transfer. The max(0, tmp2) guard (mirroring F90:5234) provides
    this even unclamped; the explicit clamp keeps it refactor-proof."""
    qnum, qaer, dgn_a = _synthetic_view()
    n_neg, a_neg = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a,
                                               n_so4_monolayers=-2.0)
    n_zero, a_zero = _mam_pcarbon_aging_1subarea(qnum, qaer, dgn_a,
                                                 n_so4_monolayers=0.0)
    np.testing.assert_array_equal(np.asarray(n_neg), np.asarray(n_zero))
    np.testing.assert_array_equal(np.asarray(a_neg), np.asarray(a_zero))
    # Instant full aging, and strictly no reversal: accum gains, pcm
    # keeps only the 10-eps sliver.
    assert float(a_neg[IBC, NACC]) > float(qaer[IBC, NACC])
    assert 0.0 < float(n_neg[NPCA]) < 1e-8 * float(qnum[NPCA])
