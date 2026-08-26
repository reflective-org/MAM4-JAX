"""Validate the CAM sulfeq cluster port (physics/strat_sulfate.py).

Reference: ``tests/reference/cam_sulfeq/sulfeq.json``, captured from the
CESM3 Fortran (``cam6_4_187``) by ``mam-box-fortran/tools/capture_sulfeq``
— see that tool's header for the branch-coverage design of the grid
(both T clamps, all three Tabazadeh activity regimes plus both activity
clamps, Kelvin-strong through Kelvin-negligible diameters).

Three parity layers, so a disagreement is attributable:

1. ``_qsat_water_cam`` alone (es and qs) — the saturation base.
2. ``calc_h2so4_wtpct`` alone — the Tabazadeh composition.
3. ``calc_h2so4_equilib_mixrat`` — the full equilibrium routine
   (qh2so4_equilib, wtpct, sulden).

Plus property tests the capture cannot express: Kelvin monotonicity in
diameter and reverse-mode gradient finiteness (the cluster must be usable
inside a differentiable driver).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default
from mam4_jax.physics.strat_sulfate import (
    _goffgratch_svp_water,
    _qsat_water_cam,
    calc_h2so4_equilib_mixrat,
    calc_h2so4_wtpct,
)

REF = Path(__file__).resolve().parent / "reference" / "cam_sulfeq" / "sulfeq.json"


@pytest.fixture(scope="module")
def ref():
    with open(REF) as f:
        d = json.load(f)
    return {k: np.asarray(v) for k, v in d.items()
            if k in ("qsat", "wtpct", "cases")}


def test_qsat_water_cam_matches_fortran(ref) -> None:
    """es and qs at machine precision across the full T x p grid,
    including T=135 K (es ~ 7.5e-10 Pa) and T=460 K (es >> p, the
    ``qs = 1`` saturated branch that distinguishes CAM's svp_to_qsat
    from the E3SM box's)."""
    t, p, es_ref, qs_ref = ref["qsat"].T
    es = np.asarray(_goffgratch_svp_water(jnp.asarray(t)))
    qs = np.asarray(_qsat_water_cam(jnp.asarray(t), jnp.asarray(p)))
    # The Fortran caps the RETURNED es at p after computing qs
    # (wv_sat_methods.F90:248, "consistent with limiters on qs"); the cap
    # never feeds qs or wtpct, so the port keeps es raw and the capture's
    # es column is compared capped.
    np.testing.assert_allclose(np.minimum(es, p), es_ref, rtol=1e-14,
                               atol=0.0, err_msg="Goff-Gratch es diverged")
    np.testing.assert_allclose(qs, qs_ref, rtol=1e-14, atol=0.0,
                               err_msg="qsat_water qs diverged")
    # The saturated branch is genuinely exercised by the grid.
    assert np.any(qs_ref == 1.0), "grid never hit the p<=es branch"


def test_wtpct_matches_fortran(ref) -> None:
    """Tabazadeh composition at machine precision, all three activity
    regimes (the capture grid pins <0.05, 0.05-0.85, >0.85 and both
    activity clamps by construction)."""
    t, p, qh2o, w_ref = ref["wtpct"].T
    w = np.asarray(calc_h2so4_wtpct(jnp.asarray(t), jnp.asarray(p),
                                    jnp.asarray(qh2o)))
    np.testing.assert_allclose(w, w_ref, rtol=1e-13, atol=0.0,
                               err_msg="calc_h2so4_wtpct diverged")
    # Both clamps of the output range appear in the reference.
    assert w_ref.min() == 25.0
    assert w_ref.max() < 100.0


def test_equilib_mixrat_matches_fortran(ref) -> None:
    """The full routine. qeq spans ~46 decades across the grid, so the
    bar is relative-only; exp() amplifies its ~100-magnitude exponent's
    last ULP by ~1e-14 relative, hence 5e-13 rather than 1e-15."""
    t, p, qh2o, d, qeq_ref, w_ref, sd_ref = ref["cases"].T
    qeq, w, sd = calc_h2so4_equilib_mixrat(
        jnp.asarray(t), jnp.asarray(p), jnp.asarray(qh2o), jnp.asarray(d))
    np.testing.assert_allclose(np.asarray(qeq), qeq_ref, rtol=5e-13, atol=0.0,
                               err_msg="qh2so4_equilib diverged")
    np.testing.assert_allclose(np.asarray(w), w_ref, rtol=1e-13, atol=0.0,
                               err_msg="Kelvin-adjusted wtpct diverged")
    np.testing.assert_allclose(np.asarray(sd), sd_ref, rtol=1e-13, atol=0.0,
                               err_msg="sulden diverged")


def test_kelvin_raises_equilibrium_for_smaller_particles() -> None:
    """akas = exp(+akelvin/r) > 1 and increases as r shrinks, so qeq must
    be strictly decreasing in dmean at fixed (T, p, qh2o). (The upstream
    comment says 'reduce'; the arithmetic multiplies — this pins the
    ported arithmetic, not the comment.)"""
    t, p = 210.0, 3.0e3
    qh2o = 0.5 * float(_qsat_water_cam(jnp.asarray(t), jnp.asarray(p)))
    d = jnp.asarray([1.0e-8, 1.1e-7, 9.0e-7])
    qeq, _, _ = calc_h2so4_equilib_mixrat(t, p, qh2o, d)
    qeq = np.asarray(qeq)
    assert qeq[0] > qeq[1] > qeq[2] > 0.0


def test_equilib_grad_finite_across_branches() -> None:
    """Reverse-mode gradients stay finite in every activity regime and at
    both T clamp edges — the double-where guards on the Tabazadeh
    branches and the saturated-qs branch are what this locks in."""
    def qeq_scalar(t, qh2o, d):
        out, _, _ = calc_h2so4_equilib_mixrat(t, 3.0e3, qh2o, d)
        return out

    grad = jax.grad(qeq_scalar, argnums=(0, 1, 2))
    for t, activ in [(135.0, 0.5), (210.0, 1e-7), (210.0, 0.5),
                     (210.0, 0.95), (460.0, 0.5)]:
        qs = float(_qsat_water_cam(jnp.asarray(min(max(t, 140.0), 450.0)),
                                   jnp.asarray(3.0e3)))
        g = grad(jnp.asarray(t), jnp.asarray(activ * qs), jnp.asarray(1.1e-7))
        for name, gi in zip(("t", "qh2o", "dmean"), g):
            assert np.isfinite(float(gi)), (
                f"d(qeq)/d({name}) non-finite at T={t}, activ={activ}")


# ---------------------------------------------------------------------------
# h2so4_reversible_uptake — the gasaerexch consumer of the cluster
# ---------------------------------------------------------------------------

def _fortran_reversible_uptake(qgas, qaer_so4, uptk, sulfeq, deltat, ido):
    """Verbatim NumPy transcription of modal_aero_gasaerexch.F90:523-566
    (loops, ``1 - exp``, per-mode cycle), as an independent reference.
    The JAX port's only arithmetic deviation is ``-expm1(-kxt)``."""
    dtxx = deltat * (1.0 + 1.0e-15)
    n_modes = len(ido)
    kxt = dtxx * sum(uptk[n] for n in range(n_modes) if ido[n] > 0)
    pxt = sum(uptk[n] * sulfeq[n] for n in range(n_modes) if ido[n] > 0)
    pxt = max(0.0, pxt * dtxx)
    if kxt >= 1.0e-5:
        g_equ = pxt / kxt
        g_avg = g_equ + (qgas - g_equ) * (1.0 - np.exp(-kxt)) / kxt
    else:
        g_avg = qgas * (1.0 - 0.5 * kxt) + 0.5 * pxt
    dqdt = np.zeros(n_modes)
    for n in range(n_modes):
        if ido[n] <= 0:
            continue
        a_bgn = qaer_so4[n] if ido[n] == 1 else 0.0
        a_end = max(0.0, a_bgn + dtxx * uptk[n] * (g_avg - sulfeq[n]))
        dqdt[n] = (a_end - a_bgn) / dtxx
    return dqdt


def test_reversible_uptake_matches_fortran_transcription() -> None:
    """Randomized sweep spanning both g_avg branches, condensation and
    evaporation, the a_end floor, and all three ido classes. Bar 5e-11:
    the expm1-vs-(1-exp) form difference peaks at ~2e-11 relative right
    at the kxt = 1e-5 branch threshold (documented deviation)."""
    from mam4_jax.physics.strat_sulfate import h2so4_reversible_uptake
    rng = np.random.default_rng(20260826)
    ido = np.array([1, 1, 2, 1])                       # accum/aitken/pcarbon/coarse
    for _ in range(300):
        deltat = float(rng.uniform(1.0, 1800.0))
        # uptake rates spanning kxt from ~1e-8 to ~1e2 across the sweep
        uptk = rng.uniform(0.1, 1.0, 4) * 10.0 ** rng.uniform(-11, -1)
        qgas = float(10.0 ** rng.uniform(-16, -9))
        qaer = rng.uniform(0.1, 1.0, 4) * 10.0 ** rng.uniform(-15, -9)
        sulfeq = rng.uniform(0.1, 1.0, 4) * 10.0 ** rng.uniform(-18, -8)
        ref = _fortran_reversible_uptake(qgas, qaer, uptk, sulfeq,
                                         deltat, ido)
        dqdt, total = h2so4_reversible_uptake(
            jnp.asarray(qgas), jnp.asarray(qaer), jnp.asarray(uptk),
            jnp.asarray(sulfeq), jnp.asarray(deltat), jnp.asarray(ido))
        np.testing.assert_allclose(
            np.asarray(dqdt), ref, rtol=5e-11, atol=1e-40,
            err_msg="dqdt_so4 diverged from the Fortran transcription")
        np.testing.assert_allclose(float(total), ref.sum(),
                                   rtol=5e-11, atol=1e-40)


def test_reversible_uptake_equilibrium_is_a_fixed_point() -> None:
    """g_bgn == sulfeq_n == s for every active mode: pxt = kxt*s exactly,
    so g_avg = s and every tendency is exactly zero — no drift at
    equilibrium, to the bit."""
    from mam4_jax.physics.strat_sulfate import h2so4_reversible_uptake
    s = 3.7e-12
    dqdt, total = h2so4_reversible_uptake(
        jnp.asarray(s), jnp.asarray([1e-11, 2e-12, 0.0, 5e-13]),
        jnp.asarray([1e-4, 3e-5, 2e-6, 4e-7]), jnp.full((4,), s),
        jnp.asarray(600.0), jnp.asarray([1, 1, 2, 1]))
    np.testing.assert_array_equal(np.asarray(dqdt), np.zeros(4))
    assert float(total) == 0.0


def test_reversible_uptake_evaporates_but_never_below_zero() -> None:
    """Over-saturated modes (sulfeq > g) evaporate: dqdt < 0 where the
    mode holds so4, and the floor stops evaporation at a_end = 0 (a mode
    can lose at most a_bgn/dtxx). The slotless ido=2 mode and the
    inactive ido=0 mode contribute nothing."""
    from mam4_jax.physics.strat_sulfate import h2so4_reversible_uptake
    deltat = 600.0
    dtxx = deltat * (1.0 + 1.0e-15)
    qaer = np.array([1e-11, 1e-18, 0.0, 0.0])
    dqdt, _ = h2so4_reversible_uptake(
        jnp.asarray(1e-15), jnp.asarray(qaer),
        jnp.asarray([1e-3, 1e-3, 1e-3, 1e-3]),
        jnp.full((4,), 1e-9),                 # sulfeq >> gas: evaporation
        jnp.asarray(deltat), jnp.asarray([1, 1, 2, 0]))
    dqdt = np.asarray(dqdt)
    assert dqdt[0] < 0.0
    # mode 1 has almost nothing: the floor limits the loss to a_bgn/dtxx
    np.testing.assert_allclose(dqdt[1], -qaer[1] / dtxx, rtol=1e-12)
    assert dqdt[2] == 0.0 and dqdt[3] == 0.0


def test_reversible_uptake_branch_continuity_and_grads() -> None:
    """The two g_avg forms agree to first order at the kxt = 1e-5
    threshold (relative gap ~kxt²/6 ≈ 2e-11), and reverse-mode gradients
    are finite on both sides of every where()."""
    from mam4_jax.physics.strat_sulfate import h2so4_reversible_uptake

    def total_at(scale):
        _, tot = h2so4_reversible_uptake(
            jnp.asarray(2e-12), jnp.asarray([1e-11, 4e-12]),
            scale * jnp.asarray([0.6, 0.4]), jnp.asarray([1e-12, 3e-12]),
            jnp.asarray(1.0), jnp.asarray([1, 1]))
        return tot

    lo = float(total_at(jnp.asarray(0.999e-5)))
    hi = float(total_at(jnp.asarray(1.001e-5)))
    np.testing.assert_allclose(lo, hi, rtol=1e-2)

    for s in (0.5e-5, 2e-5, 1e-1):
        g = jax.grad(lambda x: total_at(x))(jnp.asarray(s))
        assert np.isfinite(float(g)), f"grad non-finite at uptk scale {s}"
