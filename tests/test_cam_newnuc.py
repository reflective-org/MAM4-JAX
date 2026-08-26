"""Validate the CAM newnuc wrapper (coupling/cam_driver.py) — plan 025 G2.

Reference: ``tests/reference/cam_newnuc/{mam4,mam5}.json`` from
``mam-box-fortran/tools/capture_newnuc`` — the REAL ``modal_aero_newnuc_sub``
(cam6_4_187) over 96 cases per topology crossing warm/cold (the table
qsat's ice branch), in-PBL vs free troposphere (Wang 2008 first-order PBL
adjustment on/off), H2SO4 from just-above-cutoff to strongly nucleating,
production and prior-uptake on/off, and two humidities.

Bar: rtol 1e-10. The leaf parameterizations are the already-validated
ports (0-diff CAM vs E3SM); what this wrapper adds — the step-average
H2SO4 reconstruction (log/exp chain, ``expm1`` vs the Fortran's
``exp()-1``) and the mixed-phase TABLE qsat — carries ~1-ulp form
differences that the nucleation rate amplifies through its ~10th-power
H2SO4 (and comparable RH) sensitivity: measured worst 2.0e-11 (mam4).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.core.cam_topologies import CAM_MAM4, CAM_MAM5
from mam4_jax.coupling.cam_driver import modal_aero_newnuc_cam
from mam4_jax.physics.cam_saturation import qsat_cam

REF_DIR = Path(__file__).resolve().parent / "reference" / "cam_newnuc"


def _load(tag):
    with open(REF_DIR / f"{tag}.json") as f:
        return json.load(f)


@pytest.mark.parametrize("tag,topo", [("mam4", CAM_MAM4), ("mam5", CAM_MAM5)])
def test_newnuc_matches_fortran(tag, topo) -> None:
    d = _load(tag)
    for i, c in enumerate(d["cases"]):
        t, p, qv, zm, pblh, _qg, prod, uptk = c["case"]
        out = modal_aero_newnuc_cam(
            jnp.asarray(c["q_in"]), jnp.asarray(t), jnp.asarray(p),
            d["deltat"], jnp.asarray(qv), jnp.asarray(zm),
            jnp.asarray(pblh), jnp.asarray(prod), jnp.asarray(uptk),
            topology=topo)
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(c["q_out"]), rtol=1e-10, atol=0.0,
            err_msg=f"{tag} case {i} diverged")


@pytest.mark.parametrize("tag", ["mam4", "mam5"])
def test_grid_covers_the_gates(tag) -> None:
    """The grid must contain BOTH nucleating and fully-gated cases (the
    4e-16 cutoffs / 100 #/kmol/s floor), or the cutoff logic is untested."""
    d = _load(tag)
    fired = sum(1 for c in d["cases"]
                if any(o != n for o, n in zip(c["q_out"], c["q_in"])))
    assert 0 < fired < len(d["cases"]), (
        f"{fired}/{len(d['cases'])} fired — gates not exercised")


def test_newnuc_conserves_sulfur_exactly() -> None:
    """so4_ait gain == H2SO4 loss, case by case (both are mol/mol with
    one S each; the wrapper moves mass between exactly these two)."""
    d = _load("mam4")
    p = __import__("mam4_jax.core.cam_params", fromlist=["CAM_PARAMS"])
    names = p.CAM_PARAMS["cam_mam4"]["cnst_names"]
    lg, la = names.index("H2SO4"), names.index("so4_a2")
    for c in d["cases"]:
        dq = np.asarray(c["q_out"]) - np.asarray(c["q_in"])
        np.testing.assert_allclose(dq[la], -dq[lg], rtol=1e-12, atol=1e-30)


def test_table_qsat_differs_from_direct_water_below_freezing() -> None:
    """The wrapper must use the mixed-phase TABLE qsat: at 232 K the
    ice-blended SVP is materially below the over-water value (that ratio
    is why the direct qsat_water would be the wrong reference here)."""
    from mam4_jax.physics.strat_sulfate import _qsat_water_cam
    t, p = 232.0, 5.0e3
    _es, qs_table = qsat_cam(jnp.asarray(t), jnp.asarray(p))
    qs_water = _qsat_water_cam(jnp.asarray(t), jnp.asarray(p))
    assert float(qs_table) < 0.75 * float(qs_water)
