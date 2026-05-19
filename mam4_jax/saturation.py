"""Saturation vapor pressure and specific humidity — JAX port of `wv_saturation`.

Direct line-by-line ports of the public table-free entry points of
``mam4-original-src-code/box_model_utils/wv_saturation.F90``:

* :func:`polysvp_water`, :func:`polysvp_ice`, :func:`polysvp` — Goff–Gratch
  (1946) closed-form vapor pressure, ported from lines 699–736.

* :func:`qsat_water`, :func:`qsat_ice` — saturation specific humidity given
  (T, p), ported from lines 758–862. **Note the Fortran inconsistency**:
  ``qsat_water`` computes ``es`` via the Goff–Gratch formula (matching
  ``polysvp(T, 0)``); ``qsat_ice`` uses a Clausius–Clapeyron approximation
  ``es = 611 · exp((hlatv+hlatf)/rgasv · (1/273 − 1/T))`` instead of
  ``polysvp(T, 1)``. The JAX port preserves both formulas verbatim so it
  matches the Fortran exactly; callers that want a consistent Goff–Gratch
  treatment for ice should call ``qs_from_es(polysvp_ice(T), p)``.

Validated to ADR-003's ``1e-6`` element-wise relative-error bound against
standalone Fortran drivers under ``scripts/reference_drivers/``; see
``tests/test_polysvp.py`` and ``tests/test_qsat.py``.

All functions accept and return float64.
"""
from __future__ import annotations

import jax.numpy as jnp

from mam4_jax.constants import EPSQS, HLATF, HLATV, RGASV


def polysvp_water(T):
    """Saturation vapor pressure over liquid water at temperature ``T`` (K).

    Goff–Gatch (1946) parameterization; "uncertain below -70 °C" per the
    Fortran comment.

    Returns ``e_sat`` in Pa.
    """
    # Fortran:
    #   polysvp = 10**(-7.90298*(373.16/t - 1) + 5.02808*log10(373.16/t)
    #             - 1.3816e-7*(10**(11.344*(1 - t/373.16)) - 1)
    #             + 8.1328e-3*(10**(-3.49149*(373.16/t - 1)) - 1)
    #             + log10(1013.246)) * 100
    T_ref = 373.16
    r = T_ref / T
    exponent = (
        -7.90298 * (r - 1.0)
        + 5.02808 * jnp.log10(r)
        - 1.3816e-7 * (jnp.power(10.0, 11.344 * (1.0 - T / T_ref)) - 1.0)
        + 8.1328e-3 * (jnp.power(10.0, -3.49149 * (r - 1.0)) - 1.0)
        + jnp.log10(1013.246)
    )
    return jnp.power(10.0, exponent) * 100.0


def polysvp_ice(T):
    """Saturation vapor pressure over ice at temperature ``T`` (K).

    Goff–Gatch (1946) parameterization; "good down to -100 °C" per the
    Fortran comment. Note the formula extrapolates above 273.16 K but does
    not reflect physical behaviour there.

    Returns ``e_sat`` in Pa.
    """
    # Fortran:
    #   polysvp = 10**(-9.09718*(273.16/t - 1) - 3.56654*log10(273.16/t)
    #             + 0.876793*(1 - t/273.16) + log10(6.1071)) * 100
    T_ref = 273.16
    r = T_ref / T
    exponent = (
        -9.09718 * (r - 1.0)
        - 3.56654 * jnp.log10(r)
        + 0.876793 * (1.0 - T / T_ref)
        + jnp.log10(6.1071)
    )
    return jnp.power(10.0, exponent) * 100.0


def polysvp(T, type_: int):
    """Dispatcher matching Fortran `polysvp(T, type)` for line-by-line parity.

    ``type_`` matches the Fortran integer: ``0`` returns water saturation
    vapor pressure, ``1`` returns ice. For vectorised use, call
    :func:`polysvp_water` or :func:`polysvp_ice` directly.
    """
    if type_ == 0:
        return polysvp_water(T)
    if type_ == 1:
        return polysvp_ice(T)
    raise ValueError(f"polysvp: type must be 0 (water) or 1 (ice); got {type_!r}")


# ---------------------------------------------------------------------------
# Saturation specific humidity
# ---------------------------------------------------------------------------

def qs_from_es(es, p):
    """Saturation specific humidity from saturation vapor pressure and pressure.

    Returns ``qs = epsqs · es / (p − (1 − epsqs) · es)`` with the same
    ``qs < 0 → qs = 1`` fallback the Fortran applies when ``p`` is
    too close to ``es`` (which would otherwise yield a negative
    denominator). Matches the inline formulas in
    ``wv_saturation.F90:791`` and ``:859``.
    """
    qs = EPSQS * es / (p - (1.0 - EPSQS) * es)
    return jnp.where(qs < 0.0, 1.0, qs)


def qsat_water(T, p):
    """Saturation specific humidity over liquid water given ``T`` (K) and ``p`` (Pa).

    Goff–Gratch via :func:`polysvp_water`. Ported from
    ``wv_saturation.F90:758-799``.
    """
    es = polysvp_water(T)
    return qs_from_es(es, p)


def qsat_ice(T, p):
    """Saturation specific humidity over ice given ``T`` (K) and ``p`` (Pa).

    Clausius–Clapeyron with the combined latent heat of sublimation
    (``hlatv + hlatf``), matching the Fortran ``qsat_ice`` scalar
    function at ``wv_saturation.F90:852-862``. **This does not call
    :func:`polysvp_ice`** — see the module docstring. Callers wanting a
    Goff–Gratch ice treatment can do
    ``qs_from_es(polysvp_ice(T), p)`` instead.
    """
    T0_INV = 1.0 / 273.0
    es = 611.0 * jnp.exp((HLATV + HLATF) / RGASV * (T0_INV - 1.0 / T))
    return qs_from_es(es, p)
