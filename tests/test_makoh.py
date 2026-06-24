"""Validate JAX makoh_cubic / makoh_quartic against the Fortran reference.

Reference: ``tests/reference/makoh/reference.npz`` — produced by
``scripts/capture_reference.py --mode makoh``. Tolerance: max relative
error < 1e-6 element-wise per root branch (ADR-003).

Because both algorithms return complex roots, we compare element-wise on
the complex difference relative to the magnitude of the Fortran root.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default; JAX_ENABLE_X64=0 to opt out
from mam4_jax.kohler import makoh_cubic, makoh_quartic

REFERENCE_NPZ = (
    Path(__file__).resolve().parent / "reference" / "makoh" / "reference.npz"
)


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    data = np.load(REFERENCE_NPZ)
    return {k: np.asarray(data[k]) for k in data.files}


def _max_complex_relative_error(jax_out, fortran_out: np.ndarray) -> float:
    jax_np = np.asarray(jax_out, dtype=np.complex128)
    diff = np.abs(jax_np - fortran_out)
    mag = np.abs(fortran_out)
    # Avoid divide-by-zero on cases where Fortran returns an exact 0 root.
    # Tiny absolute differences against 0 magnitudes are acceptable.
    rel = np.where(mag > 0, diff / np.maximum(mag, 1e-300), diff)
    return float(np.max(rel))


def test_makoh_cubic_dtype_is_complex128(reference) -> None:
    inputs = reference["cubic_inputs"]
    p0 = jnp.asarray(inputs[:, 0])
    p1 = jnp.asarray(inputs[:, 1])
    p2 = jnp.asarray(inputs[:, 2])
    roots = makoh_cubic(p0, p1, p2)
    assert roots.dtype == jnp.complex128


def test_makoh_cubic_matches_fortran(reference) -> None:
    inputs = reference["cubic_inputs"]
    p0 = jnp.asarray(inputs[:, 0])
    p1 = jnp.asarray(inputs[:, 1])
    p2 = jnp.asarray(inputs[:, 2])
    roots = makoh_cubic(p0, p1, p2)
    rel_err = _max_complex_relative_error(roots, reference["cubic_roots"])
    assert rel_err < 1e-6, f"makoh_cubic max relative error = {rel_err:.3e}"


def test_makoh_cubic_insoluble_branch() -> None:
    """The p1 == 0 path returns the real cube root of -p0 (×3)."""
    p0 = jnp.asarray([-8.0])
    p1 = jnp.asarray([0.0])
    p2 = jnp.asarray([0.0])
    roots = makoh_cubic(p0, p1, p2)
    assert roots.shape == (1, 3)
    for r in np.asarray(roots[0]):
        assert np.isclose(r, 2.0 + 0.0j)


def test_makoh_quartic_matches_fortran(reference) -> None:
    inputs = reference["quartic_inputs"]
    p0 = jnp.asarray(inputs[:, 0])
    p1 = jnp.asarray(inputs[:, 1])
    p2 = jnp.asarray(inputs[:, 2])
    p3 = jnp.asarray(inputs[:, 3])
    roots = makoh_quartic(p0, p1, p2, p3)
    rel_err = _max_complex_relative_error(roots, reference["quartic_roots"])
    assert rel_err < 1e-6, f"makoh_quartic max relative error = {rel_err:.3e}"
