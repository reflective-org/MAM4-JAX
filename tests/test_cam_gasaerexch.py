"""Validate the CAM gasaerexch port (coupling/cam_driver.py) — plan 025 G1.

Reference: ``tests/reference/cam_gasaerexch/{mam4,mam5}.json``, captured by
``mam-box-fortran/tools/capture_gasaerexch`` calling the REAL
``modal_aero_gasaerexch_sub`` (cam6_4_187) with prescribed diameters — no
calcsize/wateruptake in the loop, so this isolates exactly what the port
implements: uptake rates → condensation (irreversible tropospheric AND
reversible sulfeq-limited) → the legacy 8-monolayer pcarbon aging → rename
A1 → tendency application.

The 24-case grid per topology crosses: sulfeq kind (trop / condensing strat
/ evaporating strat) × aitken diameter scale (rename off / firing) ×
pcarbon core (saturated aging / fractional aging) × H2SO4 amount. Because
the capture runs the FULL Fortran subroutine — soaexch and nh4 handling
included — parity here also confirms assumption A13 (zero SOA gas + zero
SOA aerosol is a fixed point of soaexch), not just the SO4 arithmetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.core.cam_topologies import CAM_MAM4, CAM_MAM5
from mam4_jax.coupling.cam_driver import (
    gas_aer_uptkrates_cam,
    modal_aero_gasaerexch_cam,
)

REF_DIR = Path(__file__).resolve().parent / "reference" / "cam_gasaerexch"


def _load(tag):
    with open(REF_DIR / f"{tag}.json") as f:
        return json.load(f)


@pytest.mark.parametrize("tag,topo", [("mam4", CAM_MAM4), ("mam5", CAM_MAM5)])
def test_gasaerexch_matches_fortran(tag, topo) -> None:
    """Every case, every tracer: rtol 5e-13, with a PER-SLOT atol of
    1e-13 x that tracer's own initial magnitude.

    The per-slot atol is not tolerance inflation — it defines what is
    comparable at all. The post-aging pcarbon number is a pure
    cancellation sliver, ``q*(1 - xferfrac_max) ~ q*10eps``: the
    reference binary (gfortran -O2 on arm64, -ffp-contract=fast)
    FMA-contracts ``q + dqdt*deltat`` into a single rounding, and on a
    10eps sliver that final-ulp difference is ~0.5% RELATIVE while being
    ~1e-15 of the tracer itself. Verified exactly: reproducing the
    two-rounding chain gives the JAX value bit-for-bit, and the
    correctly-rounded FMA gives the Fortran value bit-for-bit. Every
    quantity that is not a cancellation remnant of its own input passes
    the plain 5e-13 relative bar (atol 0 for slots that start at 0 —
    transfer DESTINATIONS are compared strictly).
    """
    d = _load(tag)
    assert d["nmodes"] == topo.nmodes
    deltat = d["deltat"]
    for i, c in enumerate(d["cases"]):
        t, pmid, kind, _scale, _pcore, _qgas = c["case"]
        sulfeq = (None if kind == 0
                  else jnp.asarray(c["sulfeq"]))
        q_out = modal_aero_gasaerexch_cam(
            jnp.asarray(c["q_in"]), jnp.asarray(t), jnp.asarray(pmid),
            deltat, jnp.asarray(c["dgncur_a"]),
            jnp.asarray(c["dgncur_awet"]),
            topology=topo, sulfeq=sulfeq)
        ref = np.asarray(c["q_out"])
        got = np.asarray(q_out)
        atol = 1e-13 * np.abs(np.asarray(c["q_in"]))
        # assert_allclose cannot format an array atol; same criterion by hand.
        err = np.abs(got - ref)
        bound = atol + 5e-13 * np.abs(ref)
        bad = np.where(err > bound)[0]
        assert bad.size == 0, (
            f"{tag} case {i} (sulfeq_kind={kind}) diverged at slots "
            f"{bad.tolist()}: got {got[bad]}, ref {ref[bad]}")


@pytest.mark.parametrize("tag,topo", [("mam4", CAM_MAM4), ("mam5", CAM_MAM5)])
def test_case_grid_genuinely_covers_the_branches(tag, topo) -> None:
    """The reference must exercise what it claims: rename firing (aitken
    number decreasing), aging firing (pcarbon number decreasing), and
    strat evaporation (so4 mass decreasing) each appear somewhere in the
    grid — otherwise the parity above proves less than advertised."""
    d = _load(tag)
    p = __import__("mam4_jax.core.cam_params", fromlist=["CAM_PARAMS"])
    params = p.CAM_PARAMS[topo.name]
    off = params["loffset"]
    names = params["cnst_names"]
    n_ait = topo.numptr_amode[topo.mode_index("aitken")] - off
    n_pca = topo.numptr_amode[topo.mode_index("primary_carbon")] - off
    i_so4a1 = names.index("so4_a1")
    rename_fired = aging_fired = evaporated = False
    for c in d["cases"]:
        dq = np.asarray(c["q_out"]) - np.asarray(c["q_in"])
        if dq[n_ait] < 0:
            rename_fired = True
        if dq[n_pca] < 0:
            aging_fired = True
        if c["case"][2] == 2 and dq[i_so4a1] < 0:
            evaporated = True
    assert rename_fired, "no case fired rename"
    assert aging_fired, "no case fired pcarbon aging"
    assert evaporated, "no strat case evaporated so4"


def test_uptkrates_positive_and_size_ordered() -> None:
    """Sanity on the third-variant uptake rates: positive, and at equal
    number a larger mode takes up more gas (rate grows with diameter in
    the continuum-corrected kernel)."""
    qnum = jnp.full((4,), 1.0e9 * 28.966)
    dg = jnp.asarray([1.1e-7, 2.6e-8, 2.0e-6, 5.0e-8]) * 1.2
    up = gas_aer_uptkrates_cam(qnum, 273.0, 1.0e5, dg,
                               CAM_MAM4.sigmag_amode)
    up = np.asarray(up)
    assert np.all(up > 0)
    assert up[2] > up[0] > up[3] > up[1]
