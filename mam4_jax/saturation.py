"""Saturation vapor pressure — JAX port of Fortran `wv_saturation::polysvp`.

Goff–Gratch (1946) closed-form polynomials. Direct line-by-line port of
`mam4-original-src-code/box_model_utils/wv_saturation.F90:699-736` so that
each line of JAX code traces 1:1 to the Fortran reference. Validated
against the standalone Fortran driver
(`scripts/reference_drivers/polysvp_driver.F90`) over 170 K – 320 K to
ADR-003's `1e-6` relative-error bound; see `tests/test_polysvp.py`.

Both branches accept and return float64.
"""
from __future__ import annotations

import jax.numpy as jnp


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
