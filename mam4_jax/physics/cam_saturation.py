"""CAM's table-interpolated mixed-phase saturation — ``qsat`` / ``estblf``.

The CESM3 ``wv_saturation`` module serves TWO different saturation answers:

* ``qsat_water`` — direct Goff-Gratch over liquid water (already ported
  privately by :mod:`mam4_jax.physics.strat_sulfate`, whose Tabazadeh
  composition calls it);
* the generic ``qsat`` — a LOOKUP TABLE (``estbl``) built at init over
  ``tmin = 127.16`` … ``tmax = 375.16`` K at 1 K spacing from
  ``svp_trans``: Goff-Gratch over water above ``tmelt``, Goff-Gratch over
  ice below ``tmelt - ttrice`` (ttrice = 20 K), and a linear water/ice
  blend in between — then linearly interpolated in temperature
  (``estblf``, wv_saturation.F90:378-393).

CAM's ``modal_aero_newnuc_sub`` computes its relative humidity through the
GENERIC ``qsat`` (:246), so the CAM driver must reproduce the table, not
the direct formula: at stratospheric temperatures the two differ by the
full water-vs-ice SVP ratio (~2x at 200 K), and the table's 1 K linear
interpolation is itself a (small) part of the reference answer.

The table is built once at import with numpy — the same 250 entries the
Fortran builds in ``wv_sat_init``, bit-for-bit (same formulas, same
constants, same order of operations per entry).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

_TMIN = 127.16
_TMAX = 375.16
_TMELT = 273.15               # shr_const_tkfrz
_TTRICE = 20.00               # wv_saturation.F90:98
_H2OTRIP = 273.16             # shr_const_tktrip
_TBOIL = 373.16               # wv_saturation.F90:87
_EPSILO = 18.016 / 28.966
_OMEPS = 1.0 - _EPSILO


def _svp_water_np(t):
    """GoffGratch_svp_water (wv_sat_methods.F90:569-580), numpy scalar."""
    return 10.0 ** (
        -7.90298 * (_TBOIL / t - 1.0)
        + 5.02808 * np.log10(_TBOIL / t)
        - 1.3816e-7 * (10.0 ** (11.344 * (1.0 - t / _TBOIL)) - 1.0)
        + 8.1328e-3 * (10.0 ** (-3.49149 * (_TBOIL / t - 1.0)) - 1.0)
        + np.log10(1013.246)
    ) * 100.0


def _svp_ice_np(t):
    """GoffGratch_svp_ice (wv_sat_methods.F90, 'good down to -100 C')."""
    return 10.0 ** (
        -9.09718 * (_H2OTRIP / t - 1.0)
        - 3.56654 * np.log10(_H2OTRIP / t)
        + 0.876793 * (1.0 - t / _H2OTRIP)
        + np.log10(6.1071)
    ) * 100.0


def _svp_trans_np(t):
    """wv_sat_svp_trans: water above tmelt, ice below tmelt - ttrice,
    linear blend between (weight = (tmelt - t)/ttrice)."""
    es = _svp_water_np(t) if t >= (_TMELT - _TTRICE) else 0.0
    if t < _TMELT:
        esice = _svp_ice_np(t)
        weight = 1.0 if (_TMELT - t) > _TTRICE else (_TMELT - t) / _TTRICE
        es = weight * esice + (1.0 - weight) * es
    return es


#: The Fortran table: plenest = ceiling(tmax - tmin) + 2 = 250 entries at
#: ``tmin + (i - 1)`` K (wv_saturation.F90:235-246).
_PLENEST = int(np.ceil(_TMAX - _TMIN)) + 2
_ESTBL = np.array([_svp_trans_np(_TMIN + float(i)) for i in range(_PLENEST)])


def estblf_cam(t):
    """Linear table interpolation of mixed-phase SVP (Pa) —
    ``estblf`` (wv_saturation.F90:378-393)."""
    t_tmp = jnp.maximum(jnp.minimum(t, _TMAX) - _TMIN, 0.0)
    i = jnp.floor(t_tmp).astype(jnp.int32)          # Fortran int() + 1, 0-based
    weight = t_tmp - jnp.floor(t_tmp)               # aint == floor for t_tmp >= 0
    tbl = jnp.asarray(_ESTBL)
    return (1.0 - weight) * tbl[i] + weight * tbl[i + 1]


def qsat_cam(t, p):
    """CAM's generic ``qsat`` (mixed-phase, table): returns ``(es, qs)``.

    ``qs = 1`` when ``p <= es`` (``wv_sat_svp_to_qsat``), and the returned
    ``es`` is capped at ``p`` for consistency — both exactly as
    ``qsat_line/vect`` do."""
    es = estblf_cam(t)
    saturated = (p - es) <= 0.0
    safe_den = jnp.where(saturated, 1.0, p - _OMEPS * es)
    qs = jnp.where(saturated, 1.0, _EPSILO * es / safe_den)
    return jnp.minimum(es, p), qs
