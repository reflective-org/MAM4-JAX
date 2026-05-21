"""Validate the JAX port of `binary_nuc_vehk2002` and `pbl_nuc_wang2008`.

Reference: `tests/reference/newnuc_helpers/reference.npz` — output of
`scripts/reference_drivers/newnuc_helpers_driver.F90` swept over a 3D
(T, RH, [H₂SO₄]) grid; both PBL flagaa branches captured (11 and 12).
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax.newnuc import binary_nuc_vehk2002, pbl_nuc_wang2008

REF = Path(__file__).resolve().parent / "reference" / "newnuc_helpers" / "reference.npz"
RTOL = 1e-6
# Some of the captured outputs include `1.79e308` (Fortran's huge(real8))
# alongside `0.0` (when nucleation rate underflows). Atol absorbs the
# zero-side noise.
ATOL = 1e-30


def _load() -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in np.load(REF).items()}


def test_binary_nuc_vehk2002_matches_fortran() -> None:
    d = _load()
    temp   = jnp.asarray(d["temp"])
    rh     = jnp.asarray(d["rh"])
    so4vol = jnp.asarray(d["so4vol"])

    ratenucl, rateloge, cnum_h2so4, cnum_tot, radius = binary_nuc_vehk2002(temp, rh, so4vol)

    np.testing.assert_allclose(np.asarray(ratenucl), d["binary_ratenucl"],
                               rtol=RTOL, atol=ATOL, err_msg="ratenucl mismatch")
    np.testing.assert_allclose(np.asarray(rateloge), d["binary_rateloge"],
                               rtol=RTOL, atol=ATOL, err_msg="rateloge mismatch")
    np.testing.assert_allclose(np.asarray(cnum_h2so4), d["binary_cnum_h2so4"],
                               rtol=RTOL, atol=ATOL, err_msg="cnum_h2so4 mismatch")
    np.testing.assert_allclose(np.asarray(cnum_tot),   d["binary_cnum_tot"],
                               rtol=RTOL, atol=ATOL, err_msg="cnum_tot mismatch")
    np.testing.assert_allclose(np.asarray(radius),     d["binary_radius"],
                               rtol=RTOL, atol=ATOL, err_msg="radius mismatch")


def _run_pbl(d: dict[str, np.ndarray], flagaa: int) -> tuple[np.ndarray, ...]:
    so4vol = jnp.asarray(d["so4vol"])
    rate_in = jnp.asarray(d["binary_ratenucl"])
    log_in  = jnp.asarray(d["binary_rateloge"])
    ch_in   = jnp.asarray(d["binary_cnum_h2so4"])
    ct_in   = jnp.asarray(d["binary_cnum_tot"])
    cn_in   = jnp.zeros_like(rate_in)
    rad_in  = jnp.asarray(d["binary_radius"])
    return pbl_nuc_wang2008(so4vol, flagaa, rate_in, log_in,
                             ct_in, ch_in, cn_in, rad_in)


def test_pbl_nuc_wang2008_flagaa11_matches_fortran() -> None:
    d = _load()
    (flagaa2, rate, log, cnum_tot, cnum_h2so4, cnum_nh3, radius) = _run_pbl(d, 11)

    np.testing.assert_array_equal(np.asarray(flagaa2),  d["pbl11_flagaa2"])
    np.testing.assert_allclose(np.asarray(rate),       d["pbl11_ratenucl"],
                               rtol=RTOL, atol=ATOL, err_msg="ratenucl mismatch")
    np.testing.assert_allclose(np.asarray(log),        d["pbl11_rateloge"],
                               rtol=RTOL, atol=ATOL, err_msg="rateloge mismatch")
    np.testing.assert_allclose(np.asarray(cnum_h2so4), d["pbl11_cnum_h2so4"],
                               rtol=RTOL, atol=ATOL, err_msg="cnum_h2so4 mismatch")
    np.testing.assert_allclose(np.asarray(cnum_tot),   d["pbl11_cnum_tot"],
                               rtol=RTOL, atol=ATOL, err_msg="cnum_tot mismatch")
    np.testing.assert_allclose(np.asarray(cnum_nh3),   d["pbl11_cnum_nh3"],
                               rtol=RTOL, atol=ATOL, err_msg="cnum_nh3 mismatch")
    np.testing.assert_allclose(np.asarray(radius),     d["pbl11_radius"],
                               rtol=RTOL, atol=ATOL, err_msg="radius mismatch")


def test_pbl_nuc_wang2008_flagaa12_matches_fortran() -> None:
    d = _load()
    (flagaa2, rate, log, cnum_tot, cnum_h2so4, cnum_nh3, radius) = _run_pbl(d, 12)

    np.testing.assert_array_equal(np.asarray(flagaa2),  d["pbl12_flagaa2"])
    np.testing.assert_allclose(np.asarray(rate),       d["pbl12_ratenucl"],
                               rtol=RTOL, atol=ATOL, err_msg="ratenucl mismatch")
    np.testing.assert_allclose(np.asarray(log),        d["pbl12_rateloge"],
                               rtol=RTOL, atol=ATOL, err_msg="rateloge mismatch")
    np.testing.assert_allclose(np.asarray(cnum_h2so4), d["pbl12_cnum_h2so4"],
                               rtol=RTOL, atol=ATOL, err_msg="cnum_h2so4 mismatch")
    np.testing.assert_allclose(np.asarray(cnum_tot),   d["pbl12_cnum_tot"],
                               rtol=RTOL, atol=ATOL, err_msg="cnum_tot mismatch")
    np.testing.assert_allclose(np.asarray(cnum_nh3),   d["pbl12_cnum_nh3"],
                               rtol=RTOL, atol=ATOL, err_msg="cnum_nh3 mismatch")
    np.testing.assert_allclose(np.asarray(radius),     d["pbl12_radius"],
                               rtol=RTOL, atol=ATOL, err_msg="radius mismatch")
