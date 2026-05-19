"""Validate the JAX polysvp port against committed Fortran reference data.

Reference: ``tests/reference/polysvp/reference.npz`` (1501 points from
170 K to 320 K), produced by ``scripts/capture_reference.py --mode polysvp``.
Tolerance: max relative error < 1e-6 element-wise (ADR-003).
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.saturation import polysvp_ice, polysvp_water

REFERENCE_NPZ = (
    Path(__file__).resolve().parent / "reference" / "polysvp" / "reference.npz"
)


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    data = np.load(REFERENCE_NPZ)
    return {k: np.asarray(data[k]) for k in data.files}


def _max_relative_error(jax_out: jnp.ndarray, fortran_out: np.ndarray) -> float:
    jax_np = np.asarray(jax_out, dtype=np.float64)
    return float(np.max(np.abs(jax_np - fortran_out) / np.abs(fortran_out)))


def test_dtype_is_float64(reference) -> None:
    T = jnp.asarray(reference["T"])
    assert polysvp_water(T).dtype == jnp.float64
    assert polysvp_ice(T).dtype == jnp.float64


def test_polysvp_water_matches_fortran(reference) -> None:
    T = jnp.asarray(reference["T"])
    jax_out = polysvp_water(T)
    rel_err = _max_relative_error(jax_out, reference["esat_water"])
    assert rel_err < 1e-6, f"polysvp_water max relative error = {rel_err:.3e}"


def test_polysvp_ice_matches_fortran(reference) -> None:
    T = jnp.asarray(reference["T"])
    jax_out = polysvp_ice(T)
    rel_err = _max_relative_error(jax_out, reference["esat_ice"])
    assert rel_err < 1e-6, f"polysvp_ice max relative error = {rel_err:.3e}"


def test_polysvp_dispatcher() -> None:
    """The Fortran-parity wrapper dispatches by integer type."""
    from mam4_jax.saturation import polysvp

    T = 273.16
    assert polysvp(T, 0) == polysvp_water(T)
    assert polysvp(T, 1) == polysvp_ice(T)
    with pytest.raises(ValueError):
        polysvp(T, 2)
