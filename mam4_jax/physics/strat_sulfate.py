"""Stratospheric sulfate equilibrium — JAX port of CAM's ``sulfeq`` cluster.

CAM-only physics (plan 024 PR H / plan 025 remaining item 1): above the
tropopause, CAM treats H2SO4 condensation as REVERSIBLE, limited by the
equilibrium H2SO4 mixing ratio over each mode's particles (``sulfeq``),
instead of E3SM/amicphys' irreversible uptake. The equilibrium value is
computed per mode inside CAM's water uptake from the Tabazadeh weight-percent
composition, Ayers/Kulmala vapor pressure, and a Kelvin curvature factor.

Ported line-by-line from the CESM3 snapshot vendored in the sibling repo
``mam-box-fortran`` (pinned in its ``PROVENANCE.md`` to ``cam6_4_187``):

* :func:`calc_h2so4_wtpct` — ``modal_aero_wateruptake.F90:1087-1171``.
  Weight % H2SO4 of sulfate aerosol vs water activity (Tabazadeh et al.,
  GRL 1997; rated T=185-260 K, activity 0.01-1.0).
* :func:`calc_h2so4_equilib_mixrat` — ``modal_aero_wateruptake.F90:895-1083``.
  Equilibrium H2SO4 mixing ratio over particles of mean diameter ``dmean``,
  plus the (Kelvin-adjusted) composition and sulfate density.
* :func:`_qsat_water_cam` — CAM's ``wv_sat_methods.F90`` Goff-Gratch +
  ``wv_sat_svp_to_qsat``. NOT the same as :mod:`mam4_jax.physics.saturation`
  (the E3SM box ``qsat_water``): CAM returns ``qs = 1`` whenever ``p <= es``,
  the E3SM box clamps only when the denominator has gone negative
  (``qs < 0``). The two disagree on ``es in [p, p/(1-epsilo)]`` — reachable
  at the ``t = 450 K`` clamp edge — so the CAM cluster carries its own
  helper rather than reusing the E3SM one.

Faithfulness notes (upstream oddities preserved, not fixed):

* **First surface-tension interpolation uses a mismatched abscissa**
  (F90:1005): ``surf_tens = sig1 + dsigma_dwt*(wtpct_flat - stwtp(i))``
  pairs ``sig1`` (the ``i-1`` knot's ordinate) with the ``i`` knot's
  abscissa, so ``surf_tens`` is offset by ``-(sig2-sig1)`` relative to a
  correct linear interpolation. The second lookup (F90:1032-1036, ``frac``
  form) is correct. Ported as-is; recorded in mam-box-fortran
  ``docs/bugs/``.
* The Kelvin factor MULTIPLIES the equilibrium ratio by ``exp(+akelvin/r)``
  (> 1); the upstream comment says "reduce". The arithmetic is what is
  ported.
* ``deltat`` does not appear: the equilibrium is diagnostic per call.

Everything here is a pure element-wise function of scalars/arrays
(broadcasting), float64 under the package default, no ``jit`` (phase A —
callers compose and jit). Reverse-mode safe: every branch is a
``jnp.where`` over operands computed on a clipped-to-branch-domain copy of
the input, so no dead branch produces a NaN/Inf cotangent.

Validated at float64 machine precision against
``tests/reference/cam_sulfeq/sulfeq.json``, captured from the Fortran by
``mam-box-fortran/tools/capture_sulfeq``; see ``tests/test_strat_sulfate.py``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Constants (modal_aero_wateruptake.F90:918-926). RGAS here is the CGS ideal
# gas constant in erg/mol/K — a deliberate private copy, same policy as the
# Köhler CGS island (plan 024 §3: do NOT normalise to core.constants).
# ---------------------------------------------------------------------------
_T0_KULM = 340.0                       # K, low end of Ayers' range
_T_CRIT_KULM = 905.0                   # K, 1.5 x boiling point
_FK0 = -10156.0 / _T0_KULM + 16.259    # log of Kulmala correction factor
_FK2 = 1.0 / _T0_KULM
_FK3 = 0.38 / (_T_CRIT_KULM - _T0_KULM)
_RGAS_CGS = 8.31430e7                  # erg/mol/K
_WTMOL_H2SO4 = 98.078479               # g/mol

# CAM wv_sat: tboil (wv_saturation.F90:87) and epsilo = mwwv/mwdair
# (shr_const_mod.F90:27-28) — numerically identical to core.constants.EPSQS;
# spelled out here so the CAM cluster is self-contained and auditable
# against its own Fortran.
_TBOIL = 373.16                        # K
_EPSILO = 18.016 / 28.966
_OMEPS = 1.0 - _EPSILO

# --- surface tension vs (wt%, T): sigma = stc0 + stc1*T at each knot -------
# (modal_aero_wateruptake.F90:932-944)
_STWTP = np.array([
    0.0, 23.8141, 38.0279, 40.6856, 45.335, 52.9305,
    56.2735, 59.8557, 66.2364, 73.103, 79.432, 85.9195,
    91.7444, 97.6687, 100.0])
_STC0 = np.array([
    117.564, 103.303, 101.796, 100.42, 98.4993, 91.8866,
    88.3033, 86.5546, 84.471, 81.2939, 79.3556, 75.608,
    70.0777, 63.7412, 61.4591])
_STC1 = np.array([
    -0.153641, -0.0982007, -0.0872379, -0.0818509,
    -0.0746702, -0.0522399, -0.0407773, -0.0357946, -0.0317062,
    -0.025825, -0.0267212, -0.0269204, -0.0276187, -0.0302094,
    -0.0303081])

# --- density vs (wt%, T): rho = dnc0 + dnc1*T at each knot ------------------
# (modal_aero_wateruptake.F90:947-969)
_DNWTP = np.array([
    0.0, 1.0, 5.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0,
    41.0, 45.0, 50.0, 53.0, 55.0, 56.0, 60.0, 65.0, 66.0, 70.0,
    72.0, 73.0, 74.0, 75.0, 76.0, 78.0, 79.0, 80.0, 81.0, 82.0,
    83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0,
    93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 100.0])
_DNC0 = np.array([
    1.0, 1.13185, 1.17171, 1.22164, 1.3219, 1.37209,
    1.42185, 1.4705, 1.51767, 1.52731, 1.56584, 1.61834, 1.65191,
    1.6752, 1.68708, 1.7356, 1.7997, 1.81271, 1.86696, 1.89491,
    1.9092, 1.92395, 1.93904, 1.95438, 1.98574, 2.00151, 2.01703,
    2.03234, 2.04716, 2.06082, 2.07363, 2.08461, 2.09386, 2.10143,
    2.10764, 2.11283, 2.11671, 2.11938, 2.12125, 2.1219, 2.12723,
    2.12654, 2.12621, 2.12561, 2.12494, 2.12093])
_DNC1 = np.array([
    0.0, -0.000435022, -0.000479481, -0.000531558, -0.000622448,
    -0.000660866, -0.000693492, -0.000718251, -0.000732869, -0.000735755,
    -0.000744294, -0.000761493, -0.000774238, -0.00078392, -0.000788939,
    -0.00080946, -0.000839848, -0.000845825, -0.000874337, -0.000890074,
    -0.00089873, -0.000908778, -0.000920012, -0.000932184, -0.000959514,
    -0.000974043, -0.000988264, -0.00100258, -0.00101634, -0.00102762,
    -0.00103757, -0.00104337, -0.00104563, -0.00104458, -0.00104144,
    -0.00103719, -0.00103089, -0.00102262, -0.00101355, -0.00100249,
    -0.00100934, -0.000998299, -0.000990961, -0.000985845, -0.000984529,
    -0.000989315])


def _goffgratch_svp_water(temp):
    """Goff-Gratch (1946) saturation vapor pressure over water (Pa).

    ``wv_sat_methods.F90:569-580`` (``GoffGratch_svp_water``, the scalar
    form — the vectorised variant splits the log differently),
    ``tboil = 373.16`` (wv_saturation.F90:87).
    """
    return 10.0 ** (
        -7.90298 * (_TBOIL / temp - 1.0)
        + 5.02808 * jnp.log10(_TBOIL / temp)
        - 1.3816e-7 * (10.0 ** (11.344 * (1.0 - temp / _TBOIL)) - 1.0)
        + 8.1328e-3 * (10.0 ** (-3.49149 * (_TBOIL / temp - 1.0)) - 1.0)
        + jnp.log10(1013.246)
    ) * 100.0


def _qsat_water_cam(temp, pres):
    """CAM saturation specific humidity over liquid water (kg/kg).

    :func:`_goffgratch_svp_water`, then ``wv_sat_svp_to_qsat``
    (wv_sat_methods.F90:185-198): ``qs = 1`` when ``p - es <= 0``, else
    ``epsilo*es / (p - omeps*es)``. See the module docstring for why this
    is NOT :func:`mam4_jax.physics.saturation.qsat_water`.
    """
    es = _goffgratch_svp_water(temp)
    saturated = (pres - es) <= 0.0
    safe_den = jnp.where(saturated, 1.0, pres - _OMEPS * es)
    return jnp.where(saturated, 1.0, _EPSILO * es / safe_den)


def calc_h2so4_wtpct(temp, pres, qh2o):
    """Weight % H2SO4 of sulfate aerosol (25-100).

    Port of ``calc_h2so4_wtpct`` (modal_aero_wateruptake.F90:1087-1171),
    Tabazadeh et al. (GRL 1997). Water activity ``qh2o / qsat_water`` selects
    one of three coefficient sets (< 0.05, 0.05-0.85, > 0.85); the low set
    clamps activity up to 1e-6, the high set down to 1.0. Each branch is
    evaluated on the activity clipped into that branch's domain, so the
    selected branch reproduces the Fortran exactly and the dead branches
    stay finite (reverse-mode safe).

    Note the temperature interpolation ``(temp - 190)/70`` uses the RAW
    ``temp`` argument — the [140, 450] K clamp lives in the CALLER
    (:func:`calc_h2so4_equilib_mixrat`), matching the Fortran.
    """
    qs = _qsat_water_cam(temp, pres)
    activ = qh2o / qs

    low = activ < 0.05
    high = activ > 0.85
    # Per-branch activity, clipped into the branch's own domain. Inside the
    # selected branch the clip reduces to the Fortran's own clamps
    # (max(activ, 1e-6) low, min(activ, 1) high, identity mid).
    a_lo = jnp.clip(activ, 1.0e-6, 0.05)
    a_mid = jnp.clip(activ, 0.05, 0.85)
    a_hi = jnp.clip(activ, 0.85, 1.0)

    def _cont(a, atab, btab, ctab, dtab):
        return atab * a ** btab + ctab * a + dtab

    contl = jnp.where(
        low, _cont(a_lo, 12.37208932, -0.16125516114,
                   -30.490657554, -2.1133114241),
        jnp.where(
            high, _cont(a_hi, -180.06541028, -0.38601102592,
                        -93.317846778, 273.88132245),
            _cont(a_mid, 11.820654354, -0.20786404244,
                  -4.807306373, -5.1727540348)))
    conth = jnp.where(
        low, _cont(a_lo, 13.455394705, -0.1921312255,
                   -34.285174607, -1.7620073078),
        jnp.where(
            high, _cont(a_hi, -176.95814097, -0.36257048154,
                        -90.469744201, 267.45509988),
            _cont(a_mid, 12.891938068, -0.23233847708,
                  -6.4261237757, -4.9005471319)))

    contt = contl + (conth - contl) * ((temp - 190.0) / 70.0)
    conwtp = contt * 98.0 + 1000.0
    wtpct = (100.0 * contt * 98.0) / conwtp
    return jnp.clip(wtpct, 25.0, 100.0)


def _knot_index(knots, w, start):
    """0-based upper-knot index for the Fortran ``do while`` table walk.

    ``i = start; do while (w > knots(i)) i = i + 1`` (1-based) exits at the
    first ``i >= start`` with ``knots(i) >= w`` — i.e. ``searchsorted``
    (side='left') floored at ``start``. ``w`` is always <= 100 = the last
    knot, so no upper clip is needed; one is applied anyway for safety.
    """
    idx = jnp.searchsorted(jnp.asarray(knots), w, side="left")
    return jnp.clip(idx, start, len(knots) - 1)


def _surf_tension_flat(wtpct_flat, t):
    """First surface-tension lookup (F90:995-1005): value at ``wtpct_flat``
    plus the d(sigma)/d(wt%) slope. Preserves the upstream abscissa
    mismatch — see the module docstring."""
    i = _knot_index(_STWTP, wtpct_flat, 1)
    sig1 = jnp.asarray(_STC0)[i - 1] + jnp.asarray(_STC1)[i - 1] * t
    sig2 = jnp.asarray(_STC0)[i] + jnp.asarray(_STC1)[i] * t
    dsigma_dwt = (sig2 - sig1) / (jnp.asarray(_STWTP)[i]
                                  - jnp.asarray(_STWTP)[i - 1])
    # Faithful: sig1 belongs to knot i-1 but the offset is from knot i.
    surf_tens = sig1 + dsigma_dwt * (wtpct_flat - jnp.asarray(_STWTP)[i])
    return surf_tens, dsigma_dwt


def _density_flat(wtpct_flat, t):
    """First density lookup (F90:1008-1018): value + d(rho)/d(wt%)."""
    i = _knot_index(_DNWTP, wtpct_flat, 5)
    den1 = jnp.asarray(_DNC0)[i - 1] + jnp.asarray(_DNC1)[i - 1] * t
    den2 = jnp.asarray(_DNC0)[i] + jnp.asarray(_DNC1)[i] * t
    drho_dwt = (den2 - den1) / (jnp.asarray(_DNWTP)[i]
                                - jnp.asarray(_DNWTP)[i - 1])
    density = den1 + drho_dwt * (wtpct_flat - jnp.asarray(_DNWTP)[i - 1])
    return density, drho_dwt


def _surf_tension_mode(wtpct, t):
    """Second surface-tension lookup (F90:1026-1036), correct ``frac`` form."""
    i = _knot_index(_STWTP, wtpct, 1)
    sig1 = jnp.asarray(_STC0)[i - 1] + jnp.asarray(_STC1)[i - 1] * t
    sig2 = jnp.asarray(_STC0)[i] + jnp.asarray(_STC1)[i] * t
    frac = (jnp.asarray(_STWTP)[i] - wtpct) / (jnp.asarray(_STWTP)[i]
                                               - jnp.asarray(_STWTP)[i - 1])
    return sig1 * frac + sig2 * (1.0 - frac)


def _density_mode(wtpct, t):
    """Second density lookup (F90:1039-1047), correct ``frac`` form."""
    i = _knot_index(_DNWTP, wtpct, 5)
    den1 = jnp.asarray(_DNC0)[i - 1] + jnp.asarray(_DNC1)[i - 1] * t
    den2 = jnp.asarray(_DNC0)[i] + jnp.asarray(_DNC1)[i] * t
    frac = (jnp.asarray(_DNWTP)[i] - wtpct) / (jnp.asarray(_DNWTP)[i]
                                               - jnp.asarray(_DNWTP)[i - 1])
    return den1 * frac + den2 * (1.0 - frac)


def calc_h2so4_equilib_mixrat(temp, pres, qh2o, dmean):
    """Equilibrium H2SO4 mixing ratio over sulfate particles.

    Port of ``calc_h2so4_equilib_mixrat``
    (modal_aero_wateruptake.F90:895-1083). Sequence:

    1. Clamp T to [140, 450] K; flat-surface composition ``wtpct_flat``
       via :func:`calc_h2so4_wtpct`.
    2. Kelvin factor for WATER over the particle (from the flat-surface
       surface tension/density and their wt% derivatives); recompute the
       composition at the Kelvin-reduced water, floored at ``wtpct_flat``.
    3. Giauque (1959) enthalpy fit at that composition (floored at 0).
    4. Ayers (1980) pure-H2SO4 equilibrium vapor pressure with the
       Kulmala (1990) temperature correction, composition-adjusted by the
       enthalpy term; converted atm -> Pa -> mol/mol.
    5. Kelvin factor for H2SO4 (mode surface tension/density at the
       adjusted composition), exponent clamped to +/-100.

    Parameters
    ----------
    temp : temperature (K)
    pres : pressure (Pa)
    qh2o : water vapor specific humidity (kg/kg)
    dmean : mean particle diameter of the mode (m) — CAM passes
        ``dgncur_awet * exp(1.5*alnsg**2)``, the wet surface-mode mean
        diameter of the PREVIOUS step (the lagged carried state, plan 024
        §6).

    Returns
    -------
    (qh2so4_equilib, wtpct, sulden) :
        equilibrium H2SO4 mixing ratio (mol/mol), sulfate composition
        (weight % H2SO4), sulfate density (g/cm3).
    """
    t = jnp.clip(temp, 140.0, 450.0)

    wtpct_flat = calc_h2so4_wtpct(t, pres, qh2o)

    surf_tens, dsigma_dwt = _surf_tension_flat(wtpct_flat, t)
    sulfate_density, drho_dwt = _density_flat(wtpct_flat, t)

    r = dmean * 100.0 / 2.0   # mode radius (cm) from diameter (m)

    # Kelvin effect for water (F90:1021-1029)
    rkelvin_h2o_b = (1.0 + wtpct_flat * drho_dwt / sulfate_density
                     - 3.0 * wtpct_flat * dsigma_dwt / (2.0 * surf_tens))
    rkelvin_h2o_a = (2.0 * _WTMOL_H2SO4 * surf_tens
                     / (sulfate_density * _RGAS_CGS * t * r))
    rkelvin_h2o = jnp.exp(rkelvin_h2o_a * rkelvin_h2o_b)

    qh2o_kelvin = qh2o / rkelvin_h2o
    wtpct = calc_h2so4_wtpct(t, pres, qh2o_kelvin)
    wtpct = jnp.maximum(wtpct, wtpct_flat)

    # Giauque (1959) enthalpy fit (F90:1050-1052)
    en = 4.184 * (23624.8
                  - 1.14208e8 / ((wtpct - 105.318) ** 2 + 4798.69))
    en = jnp.maximum(en, 0.0)

    surf_tens_mode = _surf_tension_mode(wtpct, t)
    sulden = _density_mode(wtpct, t)

    # Ayers (1980) + Kulmala (1990) correction (F90:1055-1075)
    fk4 = 1.0 + jnp.log(_T0_KULM / t) - _T0_KULM / t
    factor_kulm = -1.0 / t + _FK2 + _FK3 * fk4
    sulfequil = _FK0 + 10156.0 * factor_kulm - en / (8.3143 * t)
    sulfequil = jnp.exp(sulfequil) * 1.01325e5 / pres   # atm -> Pa -> mol/mol

    # Kelvin curvature factor for H2SO4 (F90:1077-1082)
    akelvin = (2.0 * _WTMOL_H2SO4 * surf_tens_mode
               / (t * sulden * _RGAS_CGS))
    expon = jnp.clip(akelvin / r, -100.0, 100.0)
    qh2so4_equilib = sulfequil * jnp.exp(expon)

    return qh2so4_equilib, wtpct, sulden
