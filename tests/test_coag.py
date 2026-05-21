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
from mam4_jax.coag import getcoags

REF = Path(__file__).resolve().parent / "reference" / "coag_coefficients" / "reference.npz"
RTOL = 1e-6
# `qv12` ranges down to ~1e-38 (third-moment intermodal transfer for the
# smallest Aitken diameters); atol absorbs noise on those near-zero values.
ATOL = 1e-40

SG_ATK = 1.6
SG_ACC = 1.8


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
