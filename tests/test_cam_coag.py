"""Validate the CAM coag port (coupling/cam_driver.py) — plan 025 G3.

Reference: ``tests/reference/cam_coag/{mam4,mam5}.json`` from
``mam-box-fortran/tools/capture_coag`` — the REAL ``modal_aero_coag_sub``
(cam6_4_187, ``pair_option_acoag = 3``) with prescribed diameters and wet
densities. 32 cases per topology: number-loading combinations that push
the three-branch closed forms through ``|tmpc| < 0.01``, ``|tmpa| < 0.001``
and the general branch; pcarbon core 0 (saturated aging) vs 1e-11 pom
(fractional aging); aitken diameter ×2.5; warm/high-p vs cold/low-p.

The getcoags kernel itself is byte-identical CAM-vs-E3SM and validated
elsewhere; what this pins is coag_sub's ORCHESTRATION (sequential number
solves consuming the prior mode's time-average, the combined aitken mass
transfer whose pcarbon-destined share ages straight through to accum
while accumulating shell volume, the legacy 8-monolayer aging fraction,
and the pcarbon direct+aging transfer).

Found by this capture: ``shr_const_rgas`` is the PRODUCT
``6.02214e26 * 1.38065e-23 = 8314.467591`` — the rounded ``8314.46``
shortcut is 9.1e-7 relative off, which the implicit number solves
amplified to ~2e-6 before the constant was corrected.

Sliver rule as in test_cam_gasaerexch: post-transfer remnants below
1e-10 of the slot's own input are compared with a per-slot atol
(1e-13 × q_in), everything else at rtol 5e-13 (measured worst 6e-16).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.core.cam_params import CAM_PARAMS
from mam4_jax.core.cam_topologies import CAM_MAM4, CAM_MAM5
from mam4_jax.coupling.cam_driver import modal_aero_coag_cam

REF_DIR = Path(__file__).resolve().parent / "reference" / "cam_coag"


def _load(tag):
    with open(REF_DIR / f"{tag}.json") as f:
        return json.load(f)


@pytest.mark.parametrize("tag,topo", [("mam4", CAM_MAM4), ("mam5", CAM_MAM5)])
def test_coag_matches_fortran(tag, topo) -> None:
    d = _load(tag)
    for i, c in enumerate(d["cases"]):
        t, p, _inum, _pcore, _scale = c["case"]
        got = np.asarray(modal_aero_coag_cam(
            jnp.asarray(c["q_in"]), jnp.asarray(t), jnp.asarray(p),
            d["deltat"], jnp.asarray(c["dgncur_a"]),
            jnp.asarray(c["dgncur_awet"]), jnp.asarray(c["wetdens"]),
            topology=topo))
        ref = np.asarray(c["q_out"])
        atol = 1e-13 * np.abs(np.asarray(c["q_in"]))
        err = np.abs(got - ref)
        bad = np.where(err > atol + 5e-13 * np.abs(ref))[0]
        assert bad.size == 0, (
            f"{tag} case {i} diverged at slots {bad.tolist()}: "
            f"got {got[bad]}, ref {ref[bad]}")


@pytest.mark.parametrize("tag,topo", [("mam4", CAM_MAM4), ("mam5", CAM_MAM5)])
def test_coag_conserves_mass_and_only_loses_number(tag, topo) -> None:
    """Coagulation conserves every species' total mass across modes,
    strictly loses aitken number, and leaves gases untouched. Also:
    coarse (and coarse_strat) modes are in NO pair — their slots must
    come back bit-identical."""
    d = _load(tag)
    p = CAM_PARAMS[topo.name]
    off = p["loffset"]
    names = p["cnst_names"]
    prefixes = ("so4", "pom", "soa", "bc", "dst", "ncl")
    untouched_modes = [m for m in range(topo.nmodes)
                       if topo.mode_names[m] not in
                       ("accum", "aitken", "primary_carbon")]
    for i, c in enumerate(d["cases"]):
        t, pp, _, _, _ = c["case"]
        got = np.asarray(modal_aero_coag_cam(
            jnp.asarray(c["q_in"]), jnp.asarray(t), jnp.asarray(pp),
            d["deltat"], jnp.asarray(c["dgncur_a"]),
            jnp.asarray(c["dgncur_awet"]), jnp.asarray(c["wetdens"]),
            topology=topo))
        qin = np.asarray(c["q_in"])
        for pre in prefixes:
            slots = [j for j, n in enumerate(names)
                     if n.rsplit("_", 1)[0] == pre and "_a" in n]
            if slots:
                np.testing.assert_allclose(
                    got[slots].sum(), qin[slots].sum(), rtol=1e-13,
                    err_msg=f"{tag} case {i}: {pre} mass not conserved")
        n_ait = topo.numptr_amode[topo.mode_index("aitken")] - off
        assert got[n_ait] <= qin[n_ait]
        assert got[names.index("H2SO4")] == qin[names.index("H2SO4")]
        for m in untouched_modes:
            slots = [topo.numptr_amode[m] - off] + [
                topo.lmassptr_amode[m][s] - off
                for s in range(topo.nspec_amode[m])]
            np.testing.assert_array_equal(
                got[slots], qin[slots],
                err_msg=f"{tag} case {i}: mode {topo.mode_names[m]} "
                        f"touched by coag")
