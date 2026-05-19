"""Validate JAX modal_aero_kohler against the Fortran reference.

Reference: ``tests/reference/kohler/reference.npz`` — produced by
``scripts/capture_reference.py --mode kohler``. 7 × 4 × 6 = 168 points
spanning insoluble particles, small-particle approximations, generic
quartic solutions, and near-saturation interpolation.

Tolerance: max relative error < 1e-6 element-wise (ADR-003).
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.kohler import modal_aero_kohler

REFERENCE_NPZ = (
    Path(__file__).resolve().parent / "reference" / "kohler" / "reference.npz"
)


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    data = np.load(REFERENCE_NPZ)
    return {k: np.asarray(data[k]) for k in data.files}


def _max_relative_error(jax_out, fortran_out: np.ndarray) -> float:
    jax_np = np.asarray(jax_out, dtype=np.float64)
    return float(np.max(np.abs(jax_np - fortran_out) / np.abs(fortran_out)))


def test_kohler_dtype_is_float64(reference) -> None:
    rdry = jnp.asarray(reference["rdry_in"])
    hygro = jnp.asarray(reference["hygro"])
    s = jnp.asarray(reference["s"])
    assert modal_aero_kohler(rdry, hygro, s).dtype == jnp.float64


def test_kohler_matches_fortran(reference) -> None:
    rdry = jnp.asarray(reference["rdry_in"])
    hygro = jnp.asarray(reference["hygro"])
    s = jnp.asarray(reference["s"])
    rwet = modal_aero_kohler(rdry, hygro, s)
    rel_err = _max_relative_error(rwet, reference["rwet"])
    assert rel_err < 1e-6, f"kohler max relative error = {rel_err:.3e}"


def test_kohler_very_small_returns_rdry() -> None:
    """The vol <= 1e-12 microns³ branch returns rwet = rdry unchanged."""
    # rdry = 1e-13 m → rdry_microns = 1e-7 → vol = 1e-21 < 1e-12. Insoluble.
    rdry = jnp.asarray([1.0e-13])
    hygro = jnp.asarray([0.5])
    s = jnp.asarray([0.99])
    rwet = modal_aero_kohler(rdry, hygro, s)
    assert float(rwet[0]) == 1.0e-13


def test_kohler_30_micron_cap() -> None:
    """The 30 micron upper bound caps unphysically large wet radii."""
    # Big rdry + very high hygro at near-saturation → growth would exceed
    # 30 microns; the Fortran caps at 30.
    rdry = jnp.asarray([1.0e-5])      # 10 microns
    hygro = jnp.asarray([1.4])        # sea-salt-class hygroscopicity
    s = jnp.asarray([0.9999])
    rwet = modal_aero_kohler(rdry, hygro, s)
    # Cap is applied as min(r, 30 microns) then * 1e-6, so equality is to
    # the float64 ULP rather than literal 3e-5.
    assert float(rwet[0]) == pytest.approx(3.0e-5, rel=1e-15)
