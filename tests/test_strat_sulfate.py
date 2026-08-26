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
