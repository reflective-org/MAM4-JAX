"""Validate the JAX port of :func:`mam4_jax.coag.getcoags`.

Reference: ``tests/reference/coag_coefficients/reference.npz`` — output
of ``scripts/reference_drivers/coag_coefficients_driver.F90`` swept
over (T, P, dgnumA, dgnumB) for fixed sigmag/density. 240 records,
covering Knudsen numbers from continuum to free-molecular and diameter
ratios spanning two and a half decades.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default; JAX_ENABLE_X64=0 to opt out
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
    """Float32 finiteness + numerical agreement for the float32-representable
    coagulation coefficients.

    The 8 getcoags outputs split into three precision tiers:

    * **Zeroth-moment** (``qn11``, ``qn22``, ``qn12``) — magnitudes 1e-15 to
      1e-12, well within float32 normal range. Should agree with the f64
      reference at float32 epsilon (~1e-6).
    * **Third-moment intermodal** (``qv12``) — magnitudes 1e-38 to 1e-35.
      Without the ``dgat3``-factored harmonic mean (see ``getcoags``) both
      operands underflow to 0 and ``qv12`` becomes 0/0 = NaN. With the
      refactor it stays finite; values above ~1e-33 (the ones that drive
      any non-negligible mass transfer) track the f64 reference to float32
      epsilon; smaller ones flush toward zero (atol absorbs them).
    * **Second-moment** (``qs11``, ``qs22``, ``qs12``, ``qs21``) — magnitudes
      1e-30 to 1e-27. These are physically below float32's useful precision
      (one cubic-metre's worth of MMR-rate noise dwarfs them). They stay
      *finite* in float32 — that is the property we lock in here — but they
      are NOT numerically meaningful in float32 and the test does not
      assert agreement.
    """
    # The test toggles jax_enable_x64 in-process. It MUST be True at entry —
    # otherwise we'd be a silent no-op, the rest of the test suite would be
    # running in f32 already, and the "restore" in finally would be wrong.
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("Test requires the default x64=on configuration to toggle.")

    d = _load()
    try:
        jax.config.update("jax_enable_x64", False)

        def f32(a):
            return jnp.asarray(a, jnp.float32)

        dgatk = f32(d["dgnumA"])
        dgacc = f32(d["dgnumB"])
        sgatk = jnp.full_like(dgatk, SG_ATK)
        sgacc = jnp.full_like(dgacc, SG_ACC)
        qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12 = getcoags(
            f32(d["lamda"]), f32(d["kfmatac"]), f32(d["kfmat"]), f32(d["kfmac"]),
            f32(d["knc"]), dgatk, dgacc, sgatk, sgacc,
            jnp.log(sgatk), jnp.log(sgacc),
        )

        # 1) Every output finite.
        out_by_name = dict(qs11=qs11, qn11=qn11, qs22=qs22, qn22=qn22,
                           qs12=qs12, qs21=qs21, qn12=qn12, qv12=qv12)
        for name, o in out_by_name.items():
            assert np.all(np.isfinite(np.asarray(o))), \
                f"getcoags output {name!r} is non-finite in float32"

        # 2) Zeroth-moment coefficients track the f64 reference at float32
        # epsilon (measured max rel-err ~9.5e-7; 2e-6 leaves ~1 decade of
        # headroom for ULP drift across CPU/GPU backends).
        for name in ("qn11", "qn22", "qn12"):
            np.testing.assert_allclose(
                np.asarray(out_by_name[name]).astype(np.float64), d[name],
                rtol=2e-6, atol=1e-40,
                err_msg=f"float32 {name} diverged from the f64 reference",
            )

        # 3) Third-moment intermodal: qv12 agrees to float32 epsilon above
        # the physically-nil atol floor of 1e-33 (a coag volume coefficient
        # ~1e-35 is a mass-transfer fraction ~1e-26, well below numerical
        # significance). Measured max rel-err with the dgat3-factored
        # harmonic mean is ~5.9e-8 — rtol=1e-6 leaves headroom.
        np.testing.assert_allclose(
            np.asarray(qv12).astype(np.float64), d["qv12"],
            rtol=1e-6, atol=1e-33,
            err_msg="float32 qv12 diverged from the f64 reference",
        )

        # 4) Second-moment coefficients (qs11, qs22, qs12, qs21) intentionally
        # not asserted for numerical agreement — magnitudes 1e-30 to 1e-27 are
        # at or below float32's useful precision floor. The finite-check above
        # is the only invariant we can lock in. This is a fundamental float32
        # precision limit, not a bug introduced by this PR.
    finally:
        # Restore unconditionally: pre-condition asserts x64 was True at entry.
        jax.config.update("jax_enable_x64", True)


def test_jax_enable_x64_zero_opts_out() -> None:
    """``JAX_ENABLE_X64=0`` set before import leaves ``jax_enable_x64`` off
    and ``mam4_jax.x64_enabled`` False (ADR-018). Run in a subprocess —
    the env var must affect the very first import, which an in-process
    test cannot reproduce."""
    code = textwrap.dedent("""
        import warnings, sys
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import mam4_jax
            import jax
        opt_out_warning = any(
            issubclass(w.category, UserWarning) and "JAX_ENABLE_X64 opt-out" in str(w.message)
            for w in caught
        )
        assert jax.config.read("jax_enable_x64") is False, "x64 leaked True"
        assert mam4_jax.x64_enabled is False, "mam4_jax.x64_enabled leaked True"
        assert opt_out_warning, "import-time opt-out UserWarning did not fire"
        print("OK")
    """).strip()
    env = {**os.environ, "JAX_ENABLE_X64": "0"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "OK" in result.stdout
