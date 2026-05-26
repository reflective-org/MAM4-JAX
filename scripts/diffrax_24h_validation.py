"""Run the JAX driver with diffrax for 24h at each dt in
(300, 30, 5, 1) and compare per-mode trajectories against the
Fortran NetCDF references in tests/reference/sweep_24h_no_pcarbon_aging/.

Outputs:
  - scripts/_artifacts/diffrax_24h_dt{dt}.npz  (JAX trajectory cache)
  - prints max per-field per-mode rel-err per dt

Plots are produced separately by diffrax_24h_plot.py (runs against
the cached .npz files; no JAX re-execution).
"""
from __future__ import annotations

import time
from pathlib import Path

import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

import mam4_jax  # noqa: F401
from mam4_jax import data
from mam4_jax.driver import run_timesteps


ROOT = Path(__file__).resolve().parent.parent
REF_24H = ROOT / "tests" / "reference" / "sweep_24h_no_pcarbon_aging"
REF_IC = ROOT / "tests" / "reference" / "per_process_full_minus_pcarbon_aging"
ART = Path(__file__).resolve().parent / "_artifacts"
ART.mkdir(exist_ok=True)

TOTAL_24H_S = 86400
DT_LIST = (300, 30, 5, 1)  # ordered fastest-running first

T_BOX, P_BOX, ZMID, PBLH, RH = 273.0, 1.0e5, 3.0e3, 1.1e3, 0.9


def _so4_idx() -> list[int]:
    out: list[int] = []
    for m in range(data.NTOT_AMODE):
        type_row = data.LSPECTYPE_AMODE[m]
        mass_row = data.LMASSPTR_AMODE[m]
        slot = next((s for s, t in enumerate(type_row) if t == 0), -1)
        out.append(int(mass_row[slot]) if slot >= 0 else -1)
    return out


def _soa_idx() -> list[int]:
    out: list[int] = []
    for m in range(data.NTOT_AMODE):
        type_row = data.LSPECTYPE_AMODE[m]
        mass_row = data.LMASSPTR_AMODE[m]
        slot = next((s for s, t in enumerate(type_row) if t == 4), -1)
        out.append(int(mass_row[slot]) if slot >= 0 else -1)
    return out


SO4_IDX = _so4_idx()
SOA_IDX = _soa_idx()
H2SO4_PCNST = int(data.LMAP_GAS[1])
SOAG_PCNST = int(data.LMAP_GAS[0])


def _build_state(snapshot, dt: float):
    ncol, pver = snapshot["q"].shape[1], snapshot["q"].shape[2]
    return {
        "q": jnp.asarray(snapshot["q"][0]),
        "qqcw": jnp.asarray(snapshot["qqcw"][0]),
        "dgncur_a": jnp.asarray(snapshot["dgncur_a"][0]),
        "dgncur_awet": jnp.asarray(snapshot["dgncur_awet"][0]),
        "qaerwat": jnp.asarray(snapshot["qaerwat"][0]),
        "wetdens": jnp.asarray(snapshot["wetdens"][0]),
        "t": jnp.asarray(np.full((ncol, pver), T_BOX)),
        "pmid": jnp.asarray(np.full((ncol, pver), P_BOX)),
        "cldn": jnp.asarray(np.full((ncol, pver), 0.0)),
        "zmid": jnp.asarray(np.full((ncol, pver), ZMID)),
        "pblh": jnp.asarray(np.full((ncol, pver), PBLH)),
        "relhum": jnp.asarray(np.full((ncol, pver), RH)),
        "deltat": jnp.asarray(float(dt)),
    }


def _extract_jax(traj, nstep: int) -> dict[str, np.ndarray]:
    j_q = np.asarray(traj["q"])
    return {
        "num_aer": np.stack(
            [j_q[:, 0, 0, int(data.NUMPTR_AMODE[m])] for m in range(4)], axis=0
        ),
        "so4_aer": np.stack(
            [j_q[:, 0, 0, SO4_IDX[m]] if SO4_IDX[m] >= 0 else np.zeros(nstep)
             for m in range(4)], axis=0
        ),
        "soa_aer": np.stack(
            [j_q[:, 0, 0, SOA_IDX[m]] if SOA_IDX[m] >= 0 else np.zeros(nstep)
             for m in range(4)], axis=0
        ),
        "h2so4_gas": j_q[:, 0, 0, H2SO4_PCNST],
        "soag_gas": j_q[:, 0, 0, SOAG_PCNST],
    }


def _load_fortran(dt: int, nstep: int) -> dict[str, np.ndarray]:
    ds = nc.Dataset(REF_24H / f"mam_dt{dt}_ndt{nstep}.nc", "r")
    try:
        return {
            "num_aer": np.asarray(ds.variables["num_aer"][:]),
            "so4_aer": np.asarray(ds.variables["so4_aer"][:]),
            "soa_aer": np.asarray(ds.variables["soa_aer"][:]),
            "h2so4_gas": np.asarray(ds.variables["h2so4_gas"][:]),
            "soag_gas": np.asarray(ds.variables["soag_gas"][:]),
        }
    finally:
        ds.close()


def _per_mode_max_relerr(j: dict, f: dict) -> dict:
    """Returns max rel-err per (field, mode) plus overall max."""
    out = {}
    for key in ("num_aer", "so4_aer", "soa_aer"):
        for m in range(4):
            rel = np.abs(j[key][m] - f[key][m]) / np.maximum(
                np.abs(f[key][m]), 1e-300
            )
            out[f"{key}_mode{m}"] = float(rel.max())
    for key in ("h2so4_gas", "soag_gas"):
        rel = np.abs(j[key] - f[key]) / np.maximum(np.abs(f[key]), 1e-300)
        out[key] = float(rel.max())
    out["MAX"] = max(out.values())
    return out


def main() -> None:
    snapshot = {
        k: np.asarray(v)
        for k, v in np.load(REF_IC / "calcsize_before.npz").items()
    }
    print(f"24h JAX/diffrax validation against Fortran")
    print(f"{'dt':>5} {'nstep':>7} {'JAX wall (s)':>14} "
          f"{'overall max':>12} {'worst field':>28}")
    print("-" * 80)

    summary = []
    for dt in DT_LIST:
        nstep = TOTAL_24H_S // dt
        state = _build_state(snapshot, dt)
        t_start = time.time()
        traj = run_timesteps(state, n_steps=nstep)
        t_wall = time.time() - t_start

        j = _extract_jax(traj, nstep)
        f = _load_fortran(dt, nstep)
        rel = _per_mode_max_relerr(j, f)

        worst_field = max(
            (k for k in rel if k != "MAX"), key=lambda k: rel[k]
        )
        print(f"{dt:>5d} {nstep:>7d} {t_wall:>14.1f} "
              f"{rel['MAX']:>12.2e} {worst_field:>28}")

        # Save JAX trajectory + per-mode rel-err for plotting later.
        np.savez(
            ART / f"diffrax_24h_dt{dt}.npz",
            dt=dt, nstep=nstep, t_wall=t_wall,
            j_num=j["num_aer"], j_so4=j["so4_aer"], j_soa=j["soa_aer"],
            j_h2so4=j["h2so4_gas"], j_soag=j["soag_gas"],
            f_num=f["num_aer"], f_so4=f["so4_aer"], f_soa=f["soa_aer"],
            f_h2so4=f["h2so4_gas"], f_soag=f["soag_gas"],
        )
        summary.append((dt, nstep, rel))

    print()
    print("Per-mode max rel-err breakdown:")
    fields = (
        "num_aer_mode0", "num_aer_mode1", "num_aer_mode2", "num_aer_mode3",
        "so4_aer_mode0", "so4_aer_mode1", "so4_aer_mode2", "so4_aer_mode3",
        "soa_aer_mode0", "soa_aer_mode1", "soa_aer_mode2", "soa_aer_mode3",
        "h2so4_gas", "soag_gas",
    )
    hdr = f"{'field':>18}  " + "  ".join(f"dt={dt:>4d}" for dt, _, _ in summary)
    print(hdr)
    for fld in fields:
        vals = "  ".join(f"{rel[fld]:>7.2e}" for _, _, rel in summary)
        print(f"{fld:>18}  {vals}")


if __name__ == "__main__":
    main()
