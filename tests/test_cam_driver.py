"""End-to-end CAM driver vs the Fortran box model — plan 025 G4b/G5.

Reference: ``tests/reference/cam_box/*.out`` — four 120-step trajectories
from the ``mam-box-fortran`` **fixdumfac** builds (see the README there):
{cam_mam4, cam_mam5} × {tropospheric, stratospheric}, all-default
namelist. This is the first time MAM5 physics runs against an independent
reference.

Bars: the reference prints tracers at SEVEN significant digits
(``es14.6``), so per-tracer parity bottoms out at ~5e-7 relative — gated
at 2e-6. The one full-precision column, total sulfur ``totS_mol``
(``es24.16``), is the machine-precision handle: measured 4.5e-15 on all
four runs, gated at 1e-13.

Reference-faithful settings baked into the driver defaults:

* ``n_substeps = 1`` — the reference box is CAM-faithful (no
  sub-stepping). The A6 measurement (this file's convergence test, and
  plan 025 §7) shows first-order splitting error: ~60-78% from converged
  at dt=30 s, halving per doubling of substeps. Defaults reproduce the
  reference (the #75-review convention); hosts should pass
  ``n_substeps >= 8`` for accuracy.
* ``reseed_dgnwet_each_step = True`` — the box time-manager shim's
  ``is_first_step()`` is true EVERY step, so the reference recomputes
  sulfeq from fresh post-calcsize dry diameters each step; production
  CAM's lagged behaviour is ``False``.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.core.cam_params import CAM_PARAMS
from mam4_jax.core.cam_topologies import CAM_MAM4, CAM_MAM5
import mam4_jax.coupling.cam_driver as cd
from mam4_jax.physics.cam_saturation import qsat_cam

REF_DIR = Path(__file__).resolve().parent / "reference" / "cam_box"

NL = dict(temp=273.0, press=1.0e5, rh=0.9, dt=30.0, nstep=120,
          numc=[1.0e8, 1.0e9, 1.0e5, 1.0e0], numc5=1.0e4,
          so4frac=[1.0, 1.0, 1.0, 0.0], qso2=1.0e-7, qh2so4=1.0e-11)


def build_ic(topo):
    """The box driver's initial state (mam_box_driver_cam.F90:279-345):
    number from the namelist, so4 mass CONSISTENT with each mode's
    prescribed size (calcsize silently rescales inconsistent pairs),
    gases vmr→mmr, qv from RH via the table qsat."""
    tb = cd._cam_tables(topo)
    p = CAM_PARAMS[topo.name]
    rho = NL["press"] / (cd._RAIR * NL["temp"])
    q = np.zeros(p["gas_pcnst"])
    for m in range(topo.nmodes):
        numkg = (NL["numc"][m] if m < 4 else NL["numc5"]) / rho
        q[tb.num_ptr[m]] = numkg
        dg, sg = topo.dgnum_amode[m], topo.sigmag_amode[m]
        tmpvol = numkg * (np.pi / 6.0) * dg ** 3 * np.exp(4.5 * np.log(sg) ** 2)
        if tb.lptr_so4[m] >= 0:
            frac = 1.0 if m == 4 else NL["so4frac"][m]
            dens = topo.specdens_amode[topo.specname_amode.index("so4")]
            q[tb.lptr_so4[m]] = frac * tmpvol * dens
    names = p["cnst_names"]
    q[names.index("H2SO4")] = (NL["qh2so4"]
                               * p["adv_mass"][names.index("H2SO4")]
                               / p["mwdry"])
    q[names.index("SO2")] = (NL["qso2"]
                             * p["adv_mass"][names.index("SO2")]
                             / p["mwdry"])
    _es, qs = qsat_cam(jnp.asarray(NL["temp"]), jnp.asarray(NL["press"]))
    nm = topo.nmodes
    return {
        "q": jnp.asarray(q), "qv": jnp.asarray(NL["rh"] * float(qs)),
        "t": jnp.asarray(NL["temp"]), "pmid": jnp.asarray(NL["press"]),
        "zm": jnp.asarray(500.0), "pblh": jnp.asarray(1000.0),
        "dgncur_a": jnp.zeros(nm), "dgncur_awet": jnp.zeros(nm),
        "wetdens": jnp.zeros(nm), "deltat": jnp.asarray(NL["dt"]),
    }


def load_ref(tag):
    rows = []
    with open(REF_DIR / f"{tag}.out") as f:
        for line in f:
            if not line.startswith("#"):
                rows.append([float(x) for x in line.split()[1:]])
    return np.asarray(rows)


CASES = [("mam4_nostrat", CAM_MAM4, False), ("mam4_strat", CAM_MAM4, True),
         ("mam5_nostrat", CAM_MAM5, False), ("mam5_strat", CAM_MAM5, True)]


@pytest.mark.parametrize("tag,topo,strat", CASES)
def test_box_trajectory_matches_fortran(tag, topo, strat) -> None:
    tb = cd._cam_tables(topo)
    names = CAM_PARAMS[topo.name]["cnst_names"]
    adv = np.asarray(tb.adv_mass)
    ref = load_ref(tag)
    _, traj = cd.cam_run_timesteps(build_ic(topo), NL["nstep"],
                                   topology=topo, strat=strat)
    q = np.asarray(traj["q"])
    cols = [
        ("num_a1", q[:, tb.num_ptr[0]]), ("num_a2", q[:, tb.num_ptr[1]]),
        ("num_a3", q[:, tb.num_ptr[2]]),
        ("so4_a1", q[:, tb.lptr_so4[0]]), ("so4_a2", q[:, tb.lptr_so4[1]]),
        ("so4_a3", q[:, tb.lptr_so4[2]]),
        ("h2so4", q[:, names.index("H2SO4")]),
        ("dgn_a1", np.asarray(traj["dgncur_a"])[:, 0]),
        ("dgnwet_a1", np.asarray(traj["dgncur_awet"])[:, 0]),
        ("wetdens_a1", np.asarray(traj["wetdens"])[:, 0]),
        ("so2", q[:, names.index("SO2")]),
    ]
    for j, (name, mine) in enumerate(cols):
        np.testing.assert_allclose(
            mine, ref[:, j], rtol=2e-6, atol=0.0,
            err_msg=f"{tag}: {name} left the reference's 7-digit print "
                    f"floor (~5e-7)")
    if topo.nmodes >= 5:
        np.testing.assert_allclose(q[:, tb.num_ptr[4]], ref[:, 12], rtol=2e-6)
        np.testing.assert_allclose(q[:, tb.lptr_so4[4]], ref[:, 13], rtol=2e-6)

    # Full-precision handle: total sulfur, machine epsilon over 120 steps.
    s_slots = [j for j, n in enumerate(names)
               if n in ("H2SO4", "SO2") or n.rsplit("_", 1)[0] == "so4"]
    totS = (q[:, s_slots] / adv[s_slots]).sum(axis=1)
    np.testing.assert_allclose(
        totS, ref[:, 11], rtol=1e-13,
        err_msg=f"{tag}: full-precision total sulfur diverged")


def test_substepping_moves_toward_convergence() -> None:
    """A6: the splitting error is first-order — more substeps must move
    the answer monotonically toward the converged limit (measured at
    dt=30 s: ~60% from converged at n=1 on accum number, halving per
    doubling). 20 steps and n ∈ {1, 2, 4} keep this cheap while pinning
    both the direction and the rough factor-two reduction."""
    topo = CAM_MAM4
    tb = cd._cam_tables(topo)
    finals = {}
    for n in (1, 2, 4, 8):
        st, _ = cd.cam_run_timesteps(build_ic(topo), 20, topology=topo,
                                     strat=False, n_substeps=n)
        finals[n] = float(st["q"][tb.num_ptr[0]])
    err = {n: abs(finals[n] - finals[8]) for n in (1, 2, 4)}
    assert err[1] > err[2] > err[4] > 0.0
    assert err[2] < 0.75 * err[1] and err[4] < 0.75 * err[2]


def test_reference_defaults_are_reference_faithful() -> None:
    """The #75-review convention: defaults reproduce the reference.
    n_substeps defaults to 1 (the box is un-substepped CAM) and the
    wateruptake reseed defaults to the box shim's every-step behaviour."""
    import inspect
    sig = inspect.signature(cd.cam_run_step)
    assert sig.parameters["n_substeps"].default == 1
    assert sig.parameters["reseed_dgnwet_each_step"].default is True
