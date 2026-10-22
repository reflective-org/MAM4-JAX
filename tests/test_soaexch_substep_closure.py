"""Organic-mass closure of the fixed-substep SOA exchange kernel.

jax-gcm#1064: the ``"substep"`` condensation backend's SOA exchange
(:func:`_mam_soaexch_1subarea_substep`) solves the gas/aerosol uptake ODE
pair in closed form per substep — an exact, mass-conserving solution of the
*unconstrained* problem — and then floors each mode's new aerosol mass at
zero. In a warm, gas-poor cell the raw (unconstrained) update can go deeply
negative when a mode's reservoir is small relative to the equilibrium SOA
concentration at that temperature; flooring it without correcting the gas
("evaporated" mass the floored reservoir never actually gave up) manufactures
organic mass out of nowhere. ASTEM's own semi-implicit corrector
(``mam_soaexch_1subarea``'s step 2, ported here as
:func:`_mam_soaexch_1subarea_astem`) does not have this defect — it solves
for the new gas self-consistently from the conserved total, so a floored
aerosol mode is automatically compensated. This module locks in that the
``"substep"`` backend now matches it.

The scenario below (tiny SOA-gas mixing ratio, small per-mode SOA aerosol
reservoir, warm temperature) is chosen to make at least one mode's
unconstrained per-substep update go negative — the reservoir-exhaustion
mechanism in jax-gcm#1064's bug report — while a cooler, gas-rich cell in
the same batch exercises ordinary condensation (never near the floor), per
jax-gcm's own acceptance test, "organic budget closes in cold and
evaporating cells".
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default
from mam4_jax.coupling import amicphys as _amicphys
from mam4_jax.core import data

# Relative organic-mass-closure tolerance ASTEM meets (jax-gcm's acceptance
# bar for this issue, matched here for the substep backend).
_CLOSURE_TOL = 2.0e-6

_NTOT = data.NTOT_AMODE
_NAER = data.AMICPHYS_NAER
_IAER_SOA = data.AMICPHYS_IAER_SOA
_IAER_POM = data.AMICPHYS_IAER_POM


def _warm_evaporating_and_cold_condensing_state():
    """Two-cell batch: cell 0 cold/gas-rich (condensing), cell 1 warm with a
    small SOA reservoir and almost no gas (evaporating, reservoir-exhausting).
    """
    temp = jnp.asarray([260.0, 320.0])
    pmid = jnp.asarray([9.0e4, 9.0e4])

    qgas_cur = jnp.zeros((2, 2), dtype=jnp.float64)
    qgas_cur = qgas_cur.at[:, 0].set(jnp.asarray([1.0e-8, 1.0e-14]))
    qgas_avg = jnp.zeros_like(qgas_cur)

    qaer_cur = jnp.zeros((2, _NAER, _NTOT), dtype=jnp.float64)
    qaer_cur = qaer_cur.at[:, _IAER_SOA, :].set(1.0e-10)
    qaer_cur = qaer_cur.at[:, _IAER_POM, :].set(1.0e-10)

    uptkaer = jnp.zeros((2, 2, _NTOT), dtype=jnp.float64)
    uptkaer = uptkaer.at[:, 0, :].set(1.0e-3)   # igas=0 (SOA) uptake rate

    return qgas_cur, qgas_avg, qaer_cur, temp, pmid, uptkaer


def _organic_total(qgas, qaer):
    g = qgas[..., 0]
    a = jnp.sum(qaer[..., _IAER_SOA, :], axis=-1)
    return g + a


def test_substep_soaexch_reservoir_exhaustion_is_the_expected_regime():
    """Sanity check on the scenario itself: cell 1's *unconstrained* raw
    aerosol update must actually go negative on at least one mode in the
    first substep, i.e. the reservoir-exhaustion branch this test targets is
    really being exercised (not a vacuous pass at any tolerance).
    """
    qgas_cur, qgas_avg, qaer_cur, temp, pmid, uptkaer = (
        _warm_evaporating_and_cold_condensing_state())
    dtsubstep, n_substeps = 1800.0, 4
    dts = dtsubstep / n_substeps

    r_univ = _amicphys.RGAS / 1.0e3
    p0_soa = _amicphys._P0_SOA_298 * np.exp(
        -(_amicphys._DELH_VAP_SOA / r_univ) * (1.0 / 320.0 - 1.0 / 298.0))
    g0_soa = _amicphys._PSTD * p0_soa / 9.0e4

    a = np.full(_NTOT, 1.0e-10)
    a_opoa = np.full(_NTOT, 0.1 * 1.0e-10)
    g = 1.0e-14
    uptk = np.full(_NTOT, 1.0e-3)
    a_tot = np.maximum(a_opoa + a, 1.0e-20)
    g_star = (g0_soa / a_tot) * a
    G = np.sum(uptk)
    S = np.sum(uptk * g_star)
    e = np.exp(-G * dts)
    int_g = g * (1.0 - e) / G + (S / G) * (dts - (1.0 - e) / G)
    a_raw = a + uptk * (int_g - g_star * dts)
    assert np.any(a_raw < 0.0), (
        "scenario must drive at least one mode's unconstrained update "
        "negative, or this test exercises ordinary condensation instead "
        "of the reservoir-exhaustion bug")


def test_substep_soaexch_closes_gas_plus_aerosol_organic():
    """The fix: gas + aerosol organic mass must close to 2e-6 per cell,
    including the warm, reservoir-exhausting cell.

    Before the bounded corrector this assertion fails by orders of
    magnitude on the warm cell (mass is manufactured, not lost — the gas
    collects the full unconstrained evaporative flux while the floored
    aerosol never actually supplied it).
    """
    qgas_cur, qgas_avg, qaer_cur, temp, pmid, uptkaer = (
        _warm_evaporating_and_cold_condensing_state())
    dtsubstep, n_substeps = 1800.0, 4

    total_before = _organic_total(qgas_cur, qaer_cur)
    qgas_new, _qgas_avg_new, qaer_new = (
        _amicphys._mam_soaexch_1subarea_substep(
            qgas_cur, qgas_avg, qaer_cur, dtsubstep, temp, pmid, uptkaer,
            n_substeps,
        ))
    total_after = _organic_total(qgas_new, qaer_new)

    rel = np.asarray((total_after - total_before) / total_before)
    np.testing.assert_allclose(
        rel, 0.0, atol=_CLOSURE_TOL,
        err_msg="gas + aerosol organic mass must close across the substep "
                "SOA exchange (jax-gcm#1064)")
    # The warm cell must actually have moved mass between gas and aerosol
    # (not a vacuous no-op): the reservoir really was driven toward zero.
    assert float(qaer_new[1, _IAER_SOA, :].sum()) < float(
        qaer_cur[1, _IAER_SOA, :].sum())


def test_substep_soaexch_matches_astem_closure_on_the_same_state():
    """Cross-check: the two backends should now agree the budget closes.

    ASTEM's step-2 corrector never had this defect (its implicit solve
    backs out the gas from the conserved total), so it is the reference
    the substep backend is brought up to.
    """
    qgas_cur, qgas_avg, qaer_cur, temp, pmid, uptkaer = (
        _warm_evaporating_and_cold_condensing_state())
    dtsubstep = 1800.0

    total_before = _organic_total(qgas_cur, qaer_cur)

    qgas_sub, _, qaer_sub = _amicphys._mam_soaexch_1subarea_substep(
        qgas_cur, qgas_avg, qaer_cur, dtsubstep, temp, pmid, uptkaer, 4)
    qgas_astem, _, qaer_astem = _amicphys._mam_soaexch_1subarea_astem(
        qgas_cur, qgas_avg, qaer_cur, dtsubstep, temp, pmid, uptkaer)

    for name, qgas_new, qaer_new in (
        ("substep", qgas_sub, qaer_sub), ("astem", qgas_astem, qaer_astem),
    ):
        total_after = _organic_total(qgas_new, qaer_new)
        rel = np.asarray((total_after - total_before) / total_before)
        np.testing.assert_allclose(
            rel, 0.0, atol=_CLOSURE_TOL,
            err_msg=f"{name} backend organic budget does not close")


def test_substep_soaexch_corrector_is_reverse_mode_differentiable():
    """The bounded corrector must stay inside the ``lax.scan`` (static
    shape, no data-dependent control flow), so `jax.grad` through the
    kernel — including through a cell that actually hits the floor — stays
    finite. This is the whole point of using a bounded corrector instead of
    ASTEM's adaptive while_loop: the substep backend is the one hosts use
    under `jax.grad` (jax-gcm's default core).
    """
    qgas_cur, qgas_avg, qaer_cur, temp, pmid, uptkaer = (
        _warm_evaporating_and_cold_condensing_state())
    dtsubstep, n_substeps = 1800.0, 4

    def loss(qaer_soa_mode0_cell1):
        qaer = qaer_cur.at[1, _IAER_SOA, 0].set(qaer_soa_mode0_cell1)
        qgas_new, _, qaer_new = _amicphys._mam_soaexch_1subarea_substep(
            qgas_cur, qgas_avg, qaer, dtsubstep, temp, pmid, uptkaer,
            n_substeps,
        )
        return jnp.sum(qgas_new[..., 0]) + jnp.sum(qaer_new[..., _IAER_SOA, :])

    g = jax.grad(loss)(jnp.asarray(1.0e-10))
    assert np.isfinite(float(g))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
