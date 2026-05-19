"""Köhler-equilibrium water uptake — internals.

JAX port of the Köhler-equilibrium solver chain from
``mam4-original-src-code/e3sm_src_modified/modal_aero_wateruptake.F90``,
which `mam4_jax.processes.wateruptake` (the process-level API stub from M1,
filled in later in M3.4 PR-C) will compose.

**Current contents (M3.4 PR-A + PR-B):**

* :func:`makoh_cubic` — Cardano-method roots of ``x³ + p₁·x + p₀ = 0``.
* :func:`makoh_quartic` — Ferrari-method roots of
  ``x⁴ + p₃·x³ + p₂·x² + p₁·x + p₀ = 0``.
* :func:`modal_aero_kohler` — Köhler-equilibrium wet-radius solver
  (lines 488–680). Composes the polynomial root finders, applies the
  small-particle approximation in the appropriate regime, and
  interpolates between the quartic (sub-saturation) and cubic (s=1)
  solutions near saturation.

All ports are line-by-line transcriptions of the Fortran. ``complex128``
roots are returned in the same order the Fortran does. NaN propagation
matches Fortran in degenerate cases.

PR-C will fill in :func:`mam4_jax.processes.wateruptake` (the
process-level entry point that today is a NotImplementedError stub),
porting the ``modal_aero_wateruptake_sub`` / ``_dr`` orchestration that
calls :func:`modal_aero_kohler` per (column, level, mode).
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


# ---------------------------------------------------------------------------
# modal_aero_kohler — Köhler-equilibrium wet-radius solver
# ---------------------------------------------------------------------------

# Constants from Fortran lines 533-539. These are the literal "in-routine"
# values the Fortran code uses; the comments at 525-531 show what the
# physically-derived versions WOULD be. We match the in-routine literals
# for bit-equivalence with the reference.
_KH_MW      = 18.0           # molecular weight of water (kg/kmole), literal
_KH_RHOW    = 1.0            # water density scaled (rhoh2o / 1e3)
_KH_SURFTEN = 76.0           # surface tension constant
_KH_TAIR    = 273.0          # reference temperature (K)
_KH_THIRD   = 1.0 / 3.0
_KH_UGASCON = 8.3e7          # gas constant scaled (r_universal * 1e4)

# Kelvin curvature coefficient (Fortran line 546).
_KH_A = 2.0e4 * _KH_MW * _KH_SURFTEN / (_KH_UGASCON * _KH_TAIR * _KH_RHOW)

# `eps` parameter from Fortran line 524 — used as RH clamp and root tolerance.
_KH_EPS = 1.0e-4


def _pick_smallest_valid_real_root(roots, rdry):
    """Select the smallest valid real root per point.

    "Valid" mirrors the Fortran loop body (lines 595–604 / 642–651):
      * ``|imag(root)| <= |real(root)| * eps`` — effectively real
      * ``real(root) > rdry * (1 - eps)`` — above the dry-radius floor
      * ``real(root) <= 1000 * rdry``       — Fortran's implicit upper bound
        (``r`` is initialised to ``1000*rdry`` and only smaller candidates win)
      * ``real(root)`` is not NaN

    Args:
        roots: complex array of shape ``(N, k)``.
        rdry: real array of shape ``(N,)``.

    Returns:
        ``(r_picked, has_solution)`` where ``r_picked`` is the smallest
        valid real root per point (``+inf`` if no valid root) and
        ``has_solution`` is a boolean mask.
    """
    eps = _KH_EPS
    xr = jnp.real(roots)
    xi = jnp.imag(roots)
    rdry_b = rdry[:, None]

    is_real   = jnp.abs(xi) <= jnp.abs(xr) * eps
    above_dry = xr >  rdry_b * (1.0 - eps)
    below_max = xr <= rdry_b * 1000.0
    finite    = ~jnp.isnan(xr)
    valid     = is_real & above_dry & below_max & finite

    masked_xr = jnp.where(valid, xr, jnp.inf)
    r_picked  = jnp.min(masked_xr, axis=-1)
    has_solution = jnp.any(valid, axis=-1)
    return r_picked, has_solution


def modal_aero_kohler(rdry_in, hygro, s):
    """Solve the Köhler-equilibrium wet radius for a batch of aerosols.

    Direct port of Fortran ``modal_aero_kohler`` (lines 488–680). Skips
    the ``#ifdef verify_wateruptake`` bisection branch (the reference
    build does not define that macro).

    Args:
        rdry_in: dry radii (m), shape ``(N,)``.
        hygro:   volume-mean hygroscopicities (dimensionless), shape ``(N,)``.
        s:       relative humidities (1.0 = saturated), shape ``(N,)``.

    Returns:
        Wet radii (m), shape ``(N,)``. Upper-bounded at 30 microns
        (Fortran line 675's "1-day lifetime" cap).
    """
    eps = _KH_EPS
    a   = _KH_A

    rdry_in = jnp.asarray(rdry_in, dtype=jnp.float64)
    hygro   = jnp.asarray(hygro,   dtype=jnp.float64)
    s       = jnp.asarray(s,       dtype=jnp.float64)

    # Convert dry radius from m to microns; "vol" is rdry**3 (NOT volume —
    # Fortran comment line 550: "vol is r**3, not volume").
    rdry = rdry_in * 1.0e6
    vol  = rdry ** 3
    b    = vol * hygro

    # RH clamping (Fortran lines 554-555).
    ss   = jnp.minimum(s, 1.0 - eps)
    ss   = jnp.maximum(ss, 1.0e-10)
    slog = jnp.log(ss)

    # Quartic coefficients (Fortran lines 557-560):
    #   x^4 + p43*x^3 + p42*x^2 + p41*x + p40 = 0
    p43 = -a / slog
    p42 = jnp.zeros_like(rdry)
    p41 = b / slog - vol
    p40 = a * vol / slog

    # Cubic coefficients (Fortran lines 562-564), used at s≈1:
    #   x^3 + p32*x^2 + p31*x + p30 = 0
    p32 = jnp.zeros_like(rdry)
    p31 = -b / a
    p30 = -vol

    # Solve both polynomial families for ALL points unconditionally —
    # cheaper than gathering. We discard the unused one per point below.
    cx4 = makoh_quartic(p40, p41, p42, p43)
    cx3 = makoh_cubic(p30, p31, p32)

    # Small-p test (Fortran line 576).
    p_param = jnp.abs(p31) / (rdry * rdry)
    small_p = p_param < eps

    # --- Quartic branch (sub-saturation) -----------------------------------

    # Small-p quartic approximation (Fortran line 579):
    #   r = rdry * (1 + p*third / (1 - slog*rdry/a))
    # Use the clamped slog (not raw log(s)).
    r_small_p_quartic = rdry * (1.0 + p_param * _KH_THIRD /
                                (1.0 - slog * rdry / a))

    # Generic quartic: pick smallest valid real root, fall back to rdry on
    # "no solution found" (Fortran line 612).
    r_q_picked, has_q = _pick_smallest_valid_real_root(cx4, rdry)
    r4 = jnp.where(has_q, r_q_picked, rdry)
    r4 = jnp.where(small_p, r_small_p_quartic, r4)

    # --- Cubic branch (near-saturation reference at s=1) -------------------

    # Small-p cubic approximation (Fortran line 626): r = rdry * (1 + p*third)
    r_small_p_cubic = rdry * (1.0 + p_param * _KH_THIRD)

    # Generic cubic: pick smallest valid real root, fall back to rdry.
    r_c_picked, has_c = _pick_smallest_valid_real_root(cx3, rdry)
    r3 = jnp.where(has_c, r_c_picked, rdry)
    r3 = jnp.where(small_p, r_small_p_cubic, r3)

    # Near-saturation interpolation (Fortran line 668):
    #   r = (r4*(1 - s) + r3*(s - 1 + eps)) / eps
    r_near_sat = (r4 * (1.0 - s) + r3 * (s - 1.0 + eps)) / eps

    near_sat = s > 1.0 - eps
    r = jnp.where(near_sat, r_near_sat, r4)

    # --- Very-small / insoluble particle override --------------------------
    # Fortran line 571: vol <= 1e-12 microns³ → r = rdry, skip all of the
    # above. We compute the rest unconditionally for vectorisation, then
    # mask back to rdry here.
    very_small = vol <= 1.0e-12
    r = jnp.where(very_small, rdry, r)

    # 30-micron upper bound (Fortran line 675).
    r = jnp.minimum(r, 30.0)

    # Microns → m (Fortran line 676).
    return r * 1.0e-6
