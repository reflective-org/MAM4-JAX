"""Validate JAX qsat_water / qsat_ice against the Fortran reference.

Reference: ``tests/reference/qsat/reference.npz`` — produced by
``scripts/capture_reference.py --mode qsat``. Grid: 301 T values (170 K
– 320 K in 0.5 K steps) × 5 pressures (10² – 1.1·10⁵ Pa). Tolerance:
max relative error < 1e-6 element-wise (ADR-003).
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default; JAX_ENABLE_X64=0 to opt out
from mam4_jax.saturation import qsat_ice, qsat_water

REFERENCE_NPZ = (
    Path(__file__).resolve().parent / "reference" / "qsat" / "reference.npz"
)


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    data = np.load(REFERENCE_NPZ)
    return {k: np.asarray(data[k]) for k in data.files}


def _max_relative_error(jax_out: jnp.ndarray, fortran_out: np.ndarray) -> float:
    jax_np = np.asarray(jax_out, dtype=np.float64)
    return float(np.max(np.abs(jax_np - fortran_out) / np.abs(fortran_out)))


def test_qsat_water_matches_fortran(reference) -> None:
    T = jnp.asarray(reference["T"])
    p = jnp.asarray(reference["p"])
    rel_err = _max_relative_error(qsat_water(T, p), reference["qs_water"])
    assert rel_err < 1e-6, f"qsat_water max relative error = {rel_err:.3e}"


def test_qsat_ice_matches_fortran(reference) -> None:
    T = jnp.asarray(reference["T"])
    p = jnp.asarray(reference["p"])
    rel_err = _max_relative_error(qsat_ice(T, p), reference["qs_ice"])
    assert rel_err < 1e-6, f"qsat_ice max relative error = {rel_err:.3e}"


def test_constants_match_fortran() -> None:
    """Sanity-check that the canonical Fortran constants we transcribed
    into mam4_jax.constants match the values shr_const_mod.F90 declares."""
    from mam4_jax import constants as c

    # From shr_const_mod.F90:60-61
    assert c.LATICE == 3.337e5
    assert c.LATVAP == 2.501e6

    # From shr_const_mod.F90:36-37
    assert c.MWDAIR == 28.966
    assert c.MWWV == 18.016

    # Derived
    assert c.EPSQS == pytest.approx(18.016 / 28.966)
    # RGAS = 6.02214e26 * 1.38065e-23 ≈ 8314.46
    assert c.RGAS == pytest.approx(8314.4641, rel=1e-5)
