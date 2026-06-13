"""Validate the JAX port of :func:`mam4_jax.coag.getcoags`.

Reference: ``tests/reference/coag_coefficients/reference.npz`` — output
of ``scripts/reference_drivers/coag_coefficients_driver.F90`` swept
over (T, P, dgnumA, dgnumB) for fixed sigmag/density. 240 records,
covering Knudsen numbers from continuum to free-molecular and diameter
ratios spanning two and a half decades.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.coag import getcoags, getcoags_wrapper_f

REF = Path(__file__).resolve().parent / "reference" / "coag_coefficients" / "reference.npz"
RTOL = 1e-6
# `qv12` ranges down to ~1e-38 (third-moment intermodal transfer for the
# smallest Aitken diameters); atol absorbs noise on those near-zero values.
ATOL = 1e-40

SG_ATK = 1.6
SG_ACC = 1.8
PDENS_ATK = 1770.0
PDENS_ACC = 1770.0


def _load() -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in np.load(REF).items()}


def test_getcoags_matches_fortran() -> None:
    d = _load()
    lamda   = jnp.asarray(d["lamda"])
    knc     = jnp.asarray(d["knc"])
    kfmat   = jnp.asarray(d["kfmat"])
    kfmac   = jnp.asarray(d["kfmac"])
    kfmatac = jnp.asarray(d["kfmatac"])
    dgatk   = jnp.asarray(d["dgnumA"])
    dgacc   = jnp.asarray(d["dgnumB"])
    sgatk   = jnp.full_like(dgatk, SG_ATK)
    sgacc   = jnp.full_like(dgacc, SG_ACC)
    xxlsgat = jnp.log(sgatk)
    xxlsgac = jnp.log(sgacc)

    qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12 = getcoags(
        lamda, kfmatac, kfmat, kfmac, knc,
        dgatk, dgacc, sgatk, sgacc, xxlsgat, xxlsgac,
    )

    for name, jax_out, fort_out in (
        ("qs11", qs11, d["qs11"]),
        ("qn11", qn11, d["qn11"]),
        ("qs22", qs22, d["qs22"]),
        ("qn22", qn22, d["qn22"]),
        ("qs12", qs12, d["qs12"]),
        ("qs21", qs21, d["qs21"]),
        ("qn12", qn12, d["qn12"]),
        ("qv12", qv12, d["qv12"]),
    ):
        np.testing.assert_allclose(
            np.asarray(jax_out), fort_out,
            rtol=RTOL, atol=ATOL,
            err_msg=f"getcoags output {name!r} diverged from Fortran reference",
        )


def test_getcoags_wrapper_f_matches_fortran() -> None:
    """JAX `getcoags_wrapper_f` reproduces the 8 post-processed beta
    coefficients at `rtol=1e-6`. Validates the prep arithmetic
    (`amu`, `lamda`, `knc`, `kfm*`), the `getcoags` composition, and
    the CMAQ→MIRAGE2 division/clamp post-processing as one unit."""
    d = _load()
    airtemp = jnp.asarray(d["temp"])
    airprs  = jnp.asarray(d["press"])
    dgatk   = jnp.asarray(d["dgnumA"])
    dgacc   = jnp.asarray(d["dgnumB"])
    sgatk   = jnp.full_like(dgatk, SG_ATK)
    sgacc   = jnp.full_like(dgacc, SG_ACC)
    pdensat = jnp.full_like(dgatk, PDENS_ATK)
    pdensac = jnp.full_like(dgacc, PDENS_ACC)
    xxlsgat = jnp.log(sgatk)
    xxlsgac = jnp.log(sgacc)

    (bij0, bij2i, bij2j, bij3,
     bii0, bii2,  bjj0,  bjj2) = getcoags_wrapper_f(
        airtemp, airprs, dgatk, dgacc, sgatk, sgacc,
        xxlsgat, xxlsgac, pdensat, pdensac,
    )

    for name, jax_out, fort_out in (
        ("betaij0",  bij0,  d["betaij0"]),
        ("betaij2i", bij2i, d["betaij2i"]),
        ("betaij2j", bij2j, d["betaij2j"]),
        ("betaij3",  bij3,  d["betaij3"]),
        ("betaii0",  bii0,  d["betaii0"]),
        ("betaii2",  bii2,  d["betaii2"]),
        ("betajj0",  bjj0,  d["betajj0"]),
        ("betajj2",  bjj2,  d["betajj2"]),
    ):
        np.testing.assert_allclose(
            np.asarray(jax_out), fort_out,
            rtol=RTOL, atol=ATOL,
            err_msg=f"getcoags_wrapper_f output {name!r} diverged from Fortran reference",
        )


def test_getcoags_finite_in_float32() -> None:
    """Every getcoags output stays finite in true float32.

    ``qv12`` (third-moment intermodal transfer) is ``coagnc3*coagfm3 /
    (coagnc3+coagfm3)`` with both terms ``∝ dgat3 = d³``; in SI metres that is
    ~1e-38, so in float32 both terms underflow to 0 and the ratio becomes
    0/0 = NaN. The dgat3-factored harmonic mean keeps the two operands in
    normal range. Run the full reference sweep (Knudsen continuum →
    free-molecular, 2.5 decades of diameter ratio) in float32 and require
    finiteness, plus rough agreement with the f64 reference where ``qv12`` is
    non-negligible (loose tol: float32 has ~7 digits; atol absorbs the ~1e-38
    records that legitimately flush toward zero).
    """
    import jax

    d = _load()
    was = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", False)

        def f32(a):
            return jnp.asarray(a, jnp.float32)

        dgatk = f32(d["dgnumA"])
        dgacc = f32(d["dgnumB"])
        sgatk = jnp.full_like(dgatk, SG_ATK)
        sgacc = jnp.full_like(dgacc, SG_ACC)
        out = getcoags(
            f32(d["lamda"]), f32(d["kfmatac"]), f32(d["kfmat"]), f32(d["kfmac"]),
            f32(d["knc"]), dgatk, dgacc, sgatk, sgacc,
            jnp.log(sgatk), jnp.log(sgacc),
        )
        names = ("qs11", "qn11", "qs22", "qn22", "qs12", "qs21", "qn12", "qv12")
        for name, o in zip(names, out):
            assert np.all(np.isfinite(np.asarray(o))), \
                f"getcoags output {name!r} is non-finite in float32"
        # qv12 values above ~1e-33 (the ones that drive any non-negligible mass
        # transfer) track the f64 reference to float32 precision; the smaller
        # ones legitimately flush toward zero in float32 (a ~1e-35 coag volume
        # coefficient is a ~1e-26 mass-transfer fraction — physically nil).
        qv12 = np.asarray(out[7]).astype(np.float64)
        np.testing.assert_allclose(
            qv12, d["qv12"], rtol=5e-3, atol=1e-33,
            err_msg="float32 qv12 diverged from the f64 reference",
        )
    finally:
        jax.config.update("jax_enable_x64", was)
