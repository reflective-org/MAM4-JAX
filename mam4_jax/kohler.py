"""Köhler-equilibrium water uptake — internals.

JAX port of the Köhler-equilibrium solver chain from
``mam4-original-src-code/e3sm_src_modified/modal_aero_wateruptake.F90``,
which `mam4_jax.processes.wateruptake` (the process-level API stub from M1,
filled in later in M3.4 PR-C) will compose.

**Current contents (M3.4 PR-A):** the two analytical polynomial root finders
that the Köhler solver leans on:

* :func:`makoh_cubic` — Cardano-method roots of ``x³ + p₁·x + p₀ = 0``.
* :func:`makoh_quartic` — Ferrari-method roots of
  ``x⁴ + p₃·x³ + p₂·x² + p₁·x + p₀ = 0``.

Both are line-by-line ports of Fortran lines 684–793 (``makoh_cubic``,
``makoh_quartic``). They return ``complex128`` roots in the same order the
Fortran does — order matters for downstream Köhler code that picks a
specific root by position.

Subsequent PRs will add :func:`modal_aero_kohler` (PR-B) and the
orchestration that lives in ``modal_aero_wateruptake_sub`` / ``_dr`` (PR-C).
"""
from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_THIRD = 1.0 / 3.0


def _complex_cbrt(z):
    """Principal complex cube root via ``z**(1/3)``.

    Mirrors Fortran's ``z**third`` behavior under gfortran (which uses
    ``exp(log(z) / 3)``). Both languages use the principal branch of
    ``log``, so the cube root matches at float64 precision.
    """
    return z ** _THIRD


# ---------------------------------------------------------------------------
# makoh_cubic — Cardano-method depressed-cubic solver
# ---------------------------------------------------------------------------

def makoh_cubic(p0, p1, p2):
    """Solve a batch of depressed cubics ``x³ + p₁·x + p₀ = 0``.

    Direct port of Fortran ``makoh_cubic`` (lines 684–731). The Fortran
    signature includes ``p2`` (the quadratic coefficient) but the algorithm
    body does not use it — ``p2`` is accepted here for parity. Pass any
    value; it is silently ignored.

    Args:
        p0, p1, p2: 1D arrays of polynomial coefficients (any shape; broadcast
            together). ``p2`` is accepted but unused.

    Returns:
        Complex array of shape ``(N, 3)`` where ``N = jnp.broadcast_shapes(...)[0]``.
        Root order matches the Fortran: ``[-cy - cz, -cw·cy - cwsq·cz,
        -cwsq·cy - cw·cz]`` for the general path, and ``[(-p₀)^(1/3)] × 3``
        for the "insoluble particle" branch (``p1 == 0``).
    """
    del p2  # documented Fortran quirk

    p0 = jnp.asarray(p0, dtype=jnp.float64)
    p1 = jnp.asarray(p1, dtype=jnp.float64)

    EPS = 1.0e-20

    # Complex unit constants (Fortran lines 702-705).
    sqrt3 = jnp.sqrt(3.0)
    cw = 0.5 * (-1.0 + 1j * sqrt3)            # primitive cube root of unity
    cwsq = 0.5 * (-1.0 - 1j * sqrt3)          # its conjugate

    # General path: depressed-cubic substitution
    #     q = p1/3, r = p0/2
    #     crad = sqrt(r² + q³)        (complex)
    #     cy   = (r - crad)^(1/3)     (with the |cy|>eps guard)
    #     cz   = -q / cy
    # NaN propagation matches Fortran: if cy collapses to exactly 0 (which
    # only happens when p1 == 0 → q == 0 → cy_raw == 0, OR for the rare
    # pathological case where q and r conspire to give crad == r exactly),
    # the division by cy yields NaN here. The insoluble-branch `where`
    # below masks it back to the well-defined real cube root.
    q = (p1 / 3.0).astype(jnp.complex128)
    r = (p0 / 2.0).astype(jnp.complex128)
    crad = jnp.sqrt(r * r + q * q * q)
    cy_raw = r - crad
    # Fortran:  if (abs(cy) > eps) cy = cy ** third   else cy stays as cy_raw
    cy = jnp.where(jnp.abs(cy_raw) > EPS, _complex_cbrt(cy_raw), cy_raw)

    cz = -q / cy

    x1_g = -cy - cz
    x2_g = -cw  * cy - cwsq * cz
    x3_g = -cwsq * cy - cw   * cz

    # Insoluble branch (Fortran lines 708-712): if p1 == 0, all three roots
    # are (-p0)^(1/3) (principal real cube root for negative real p0).
    cx_insoluble = _complex_cbrt((-p0).astype(jnp.complex128))
    insoluble = p1 == 0.0

    x1 = jnp.where(insoluble, cx_insoluble, x1_g)
    x2 = jnp.where(insoluble, cx_insoluble, x2_g)
    x3 = jnp.where(insoluble, cx_insoluble, x3_g)

    return jnp.stack([x1, x2, x3], axis=-1)


# ---------------------------------------------------------------------------
# makoh_quartic — Ferrari-method quartic solver
# ---------------------------------------------------------------------------

def makoh_quartic(p0, p1, p2, p3):
    """Solve a batch of quartics ``x⁴ + p₃·x³ + p₂·x² + p₁·x + p₀ = 0``.

    Direct port of Fortran ``makoh_quartic`` (lines 735–793). Returns
    complex roots in Fortran order; downstream Köhler code picks roots
    by position, so the order is part of the contract.

    Args:
        p0, p1, p2, p3: 1D arrays of polynomial coefficients.

    Returns:
        Complex array of shape ``(N, 4)``. The "insoluble particle"
        branch (Fortran ``cb == 0``) returns ``[(-p₁)^(1/3)] × 4``.
    """
    p0 = jnp.asarray(p0, dtype=jnp.float64)
    p1 = jnp.asarray(p1, dtype=jnp.float64)
    p2 = jnp.asarray(p2, dtype=jnp.float64)
    p3 = jnp.asarray(p3, dtype=jnp.float64)

    # q, r — depressed-quartic helpers (Fortran lines 756-758).
    q = -p2 * p2 / 36.0 + (p3 * p1 - 4.0 * p0) / 12.0
    r = -(p2 / 6.0) ** 3 \
        + p2 * (p3 * p1 - 4.0 * p0) / 48.0 \
        + (4.0 * p0 * p2 - p0 * p3 * p3 - p1 * p1) / 16.0

    qc = q.astype(jnp.complex128)
    rc = r.astype(jnp.complex128)
    crad = jnp.sqrt(rc * rc + qc * qc * qc)

    cb_raw = rc - crad
    insoluble = cb_raw == 0.0

    # General path (Fortran lines 771-788). NaN propagation matches Fortran:
    # the cb_raw == 0 case sends cb to 0 and then `qc / cb` to NaN; the
    # insoluble-branch `where` below masks those cases back to the
    # well-defined real cube root of -p1.
    cb = _complex_cbrt(cb_raw)
    cy = -cb + qc / cb + p2.astype(jnp.complex128) / 6.0
    cb0 = jnp.sqrt(cy * cy - p0.astype(jnp.complex128))
    cb1 = (p3.astype(jnp.complex128) * cy - p1.astype(jnp.complex128)) / (2.0 * cb0)

    # First pair (Fortran lines 778-782).
    cb_a = p3.astype(jnp.complex128) / 2.0 + cb1
    crad_a = jnp.sqrt(cb_a * cb_a - 4.0 * (cy + cb0))
    x1_g = (-cb_a + crad_a) / 2.0
    x2_g = (-cb_a - crad_a) / 2.0

    # Second pair (Fortran lines 784-788).
    cb_b = p3.astype(jnp.complex128) / 2.0 - cb1
    crad_b = jnp.sqrt(cb_b * cb_b - 4.0 * (cy - cb0))
    x3_g = (-cb_b + crad_b) / 2.0
    x4_g = (-cb_b - crad_b) / 2.0

    # Insoluble branch (Fortran lines 764-769): all roots = (-p1)^(1/3).
    cx_insoluble = _complex_cbrt((-p1).astype(jnp.complex128))
    x1 = jnp.where(insoluble, cx_insoluble, x1_g)
    x2 = jnp.where(insoluble, cx_insoluble, x2_g)
    x3 = jnp.where(insoluble, cx_insoluble, x3_g)
    x4 = jnp.where(insoluble, cx_insoluble, x4_g)

    return jnp.stack([x1, x2, x3, x4], axis=-1)
