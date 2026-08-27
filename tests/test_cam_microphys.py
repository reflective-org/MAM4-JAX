"""Validate the CAM microphysics SEQUENCE (mam_microphysics_cam) — plan 025 G4a.

Reference: ``tests/reference/cam_microphys/{mam4,mam5}.json`` from
``mam-box-fortran/tools/capture_microphys`` — the sibling repo's own
``mam_coupling_cam`` (its transcription of ``aero_model.F90:1202-1247``),
chaining the real gasaerexch → newnuc → coag with the
``del_h2so4_aeruptk`` positive-down bookkeeping across the gasaerexch
call. The components are pinned individually by the G1-G3 tests; this
pins the CHAIN.

Bar: rtol 5e-12 with the per-slot sliver rule of the G1/G3 tests
(remnants below 1e-10 of the slot's own input carry a 1e-13·q_in atol).
Measured worst: 1.5e-12 — the G1-G3 ulp-level form differences compounded
through three chained stages.
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
from mam4_jax.coupling.cam_driver import mam_microphysics_cam

REF_DIR = Path(__file__).resolve().parent / "reference" / "cam_microphys"


def _load(tag):
    with open(REF_DIR / f"{tag}.json") as f:
        return json.load(f)


def _run_case(c, deltat, topo):
    t, p, qv, zm, pblh, kind, _qg, prod = c["case"]
    sulfeq = None if kind == 0 else jnp.asarray(c["sulfeq"])
    return np.asarray(mam_microphysics_cam(
        jnp.asarray(c["q_in"]), jnp.asarray(t), jnp.asarray(p), deltat,
        jnp.asarray(qv), jnp.asarray(zm), jnp.asarray(pblh),
        jnp.asarray(c["dgncur_a"]), jnp.asarray(c["dgncur_awet"]),
        jnp.asarray(c["wetdens"]), jnp.asarray(prod),
        topology=topo, sulfeq=sulfeq))


@pytest.mark.parametrize("tag,topo", [("mam4", CAM_MAM4), ("mam5", CAM_MAM5)])
def test_sequence_matches_fortran(tag, topo) -> None:
    d = _load(tag)
    for i, c in enumerate(d["cases"]):
        got = _run_case(c, d["deltat"], topo)
        ref = np.asarray(c["q_out"])
        atol = 1e-13 * np.abs(np.asarray(c["q_in"]))
        bad = np.where(np.abs(got - ref) > atol + 5e-12 * np.abs(ref))[0]
        assert bad.size == 0, (
            f"{tag} case {i} diverged at slots {bad.tolist()}: "
            f"got {got[bad]}, ref {ref[bad]}")


@pytest.mark.parametrize("tag,topo", [("mam4", CAM_MAM4), ("mam5", CAM_MAM5)])
def test_sequence_conserves_sulfur(tag, topo) -> None:
    """Gas + all so4 modes is conserved by the whole chain to machine
    precision, case by case (condensation, nucleation, coagulation and
    both aging paths only MOVE sulfur; production enters q before this
    call in the driver design)."""
    d = _load(tag)
    names = CAM_PARAMS[topo.name]["cnst_names"]
    s_slots = [j for j, n in enumerate(names)
               if n == "H2SO4" or n.rsplit("_", 1)[0] == "so4"]
    for i, c in enumerate(d["cases"]):
        got = _run_case(c, d["deltat"], topo)
        np.testing.assert_allclose(
            got[s_slots].sum(), np.asarray(c["q_in"])[s_slots].sum(),
            rtol=1e-13,
            err_msg=f"{tag} case {i}: sulfur not conserved by the chain")


def test_aeruptk_bookkeeping_feeds_newnuc() -> None:
    """The chain must hand newnuc the H2SO4 consumed by condensation:
    with strong condensation (trop, high gas) the chained result differs
    from running newnuc with aeruptk = 0 on the post-gasaerexch state —
    i.e. the bookkeeping is live, not decorative."""
    from mam4_jax.coupling.cam_driver import (
        modal_aero_gasaerexch_cam, modal_aero_newnuc_cam)
    d = _load("mam4")
    c = next(cc for cc in d["cases"]
             if cc["case"][5] == 0 and cc["case"][6] > 1e-10)
    t, p, qv, zm, pblh, _k, _qg, prod = c["case"]
    args = (jnp.asarray(t), jnp.asarray(p), d["deltat"])
    q0 = jnp.asarray(c["q_in"])
    chained = _run_case(c, d["deltat"], CAM_MAM4)
    # newnuc with the bookkeeping zeroed:
    q1 = modal_aero_gasaerexch_cam(
        q0, *args, jnp.asarray(c["dgncur_a"]),
        jnp.asarray(c["dgncur_awet"]), topology=CAM_MAM4, sulfeq=None)
    q2 = modal_aero_newnuc_cam(
        q1, *args, jnp.asarray(qv), jnp.asarray(zm), jnp.asarray(pblh),
        jnp.asarray(prod), jnp.asarray(0.0), topology=CAM_MAM4)
    assert not np.allclose(np.asarray(q2), chained, rtol=1e-6), (
        "zeroing del_h2so4_aeruptk changed nothing — the bookkeeping "
        "is not being exercised by this state")
