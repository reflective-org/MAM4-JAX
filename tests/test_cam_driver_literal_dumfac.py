"""CAM driver with ``bug_compat_stale_dumfac=True`` vs LITERAL upstream CAM.

``tests/test_cam_driver.py`` validates the default (fixed per-mode
``dumfac``) against the fixdumfac Fortran builds, at a loading where no
aitken<->accum transfer fires. This file covers the other half: at
``qso2 = 1e-5`` the transfer fires (from step ~50-60), and the CAM driver
with ``bug_compat_stale_dumfac=True`` must reproduce the unpatched builds
(``tests/reference/cam_box_literal``), while the default must NOT (it
differs there by up to 21 % / 34 % in accum dgn for MAM4 / MAM5).

The strat cases also pin the surface-tension default: the literal builds
carry the F90:1005 abscissa mismatch, so ``fix_surftens_interp=False``
(the default) is part of the literal behaviour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.core.cam_params import CAM_PARAMS
from mam4_jax.core.cam_topologies import CAM_MAM4, CAM_MAM5
import mam4_jax.coupling.cam_driver as cd

import test_cam_driver as base

REF_DIR = Path(__file__).resolve().parent / "reference" / "cam_box_literal"
CASES = [("mam4_nostrat", CAM_MAM4, False), ("mam4_strat", CAM_MAM4, True),
         ("mam5_nostrat", CAM_MAM5, False), ("mam5_strat", CAM_MAM5, True)]
H2SO4_COL = 6


def _ic(topo):
    saved = dict(base.NL)
    base.NL.update(qso2=1.0e-5)
    try:
        return base.build_ic(topo)
    finally:
        base.NL.clear()
        base.NL.update(saved)


def _columns(topo, traj):
    tb = cd._cam_tables(topo)
    q = np.asarray(traj["q"])
    names = CAM_PARAMS[topo.name]["cnst_names"]
    return np.stack([
        q[:, tb.num_ptr[0]], q[:, tb.num_ptr[1]], q[:, tb.num_ptr[2]],
        q[:, tb.lptr_so4[0]], q[:, tb.lptr_so4[1]], q[:, tb.lptr_so4[2]],
        q[:, names.index("H2SO4")],
        np.asarray(traj["dgncur_a"])[:, 0],
        np.asarray(traj["dgncur_awet"])[:, 0],
        np.asarray(traj["wetdens"])[:, 0],
        q[:, names.index("SO2")],
    ], axis=1)


def _load(tag):
    rows = [[float(x) for x in line.split()[1:]]
            for line in open(REF_DIR / f"{tag}.out") if not line.startswith("#")]
    return np.asarray(rows)


@pytest.mark.parametrize("tag,topo,strat", CASES)
def test_stale_dumfac_reproduces_literal_cam(tag, topo, strat) -> None:
    ref = _load(tag)
    _, traj = cd.cam_run_timesteps(_ic(topo), len(ref), topology=topo,
                                   strat=strat, n_substeps=1,
                                   bug_compat_stale_dumfac=True)
    mine = _columns(topo, traj)
    sizes = [j for j in range(11) if j != H2SO4_COL]
    np.testing.assert_allclose(mine[:, sizes], ref[:, sizes], rtol=2e-6,
                               atol=0.0, err_msg=f"{tag}: literal CAM")
    # Residual gas after near-complete uptake falls to ~1e-24 (twelve decades
    # below its peak), a cancellation product: gated relative to its peak.
    np.testing.assert_allclose(mine[:, H2SO4_COL], ref[:, H2SO4_COL], rtol=2e-6,
                               atol=2e-6 * ref[:, H2SO4_COL].max(),
                               err_msg=f"{tag}: literal CAM H2SO4")


@pytest.mark.parametrize("tag,topo,strat", CASES[2:3])
def test_fixed_dumfac_departs_from_literal_cam(tag, topo, strat) -> None:
    """Guard: the fixture must actually discriminate (transfer fired)."""
    ref = _load(tag)
    _, traj = cd.cam_run_timesteps(_ic(topo), len(ref), topology=topo,
                                   strat=strat, n_substeps=1)
    rel = np.abs(_columns(topo, traj)[:, 7] - ref[:, 7]) / ref[:, 7]
    assert rel.max() > 0.2, "no transfer fired; the literal test is vacuous"


def test_fix_surftens_interp_default_and_effect() -> None:
    """Default off (signature), and on the strat path it changes the
    trajectory (sulfeq and Tabazadeh water) while off is bit-identical."""
    import inspect
    for fn in (cd.cam_run_step, cd.cam_run_timesteps):
        assert inspect.signature(fn).parameters["fix_surftens_interp"].default is False
    topo = CAM_MAM5
    kw = dict(topology=topo, strat=True, n_substeps=1)
    _, a = cd.cam_run_timesteps(_ic(topo), 5, **kw)
    _, b = cd.cam_run_timesteps(_ic(topo), 5, fix_surftens_interp=False, **kw)
    _, c = cd.cam_run_timesteps(_ic(topo), 5, fix_surftens_interp=True, **kw)
    np.testing.assert_array_equal(np.asarray(a["q"]), np.asarray(b["q"]))
    assert not np.array_equal(np.asarray(a["wetdens"]), np.asarray(c["wetdens"]))
