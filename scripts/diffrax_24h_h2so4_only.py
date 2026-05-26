"""H₂SO₄-only 24h experiment (soaexch off, ablation).

Compares JAX with `_mam_soaexch_1subarea` patched to passthrough
against the matching Fortran 24h reference built with
`gasaerexch_skip_soaexch.patch` + `skip_pcarbon_aging.patch`.

Both sides have the soaexch path disabled. The H₂SO₄ analytical /
diffrax integration is the only gas-aerosol exchange running. Per
the PR-D2 diagnosis, h2so4_gas rel-err is expected to drop from
the full-physics 0.31% floor (PR-D2 baseline) to ~machine
precision, confirming the soaexch propagation hypothesis.

Caches per-dt outputs to scripts/_artifacts/h2so4_only_24h_dt*.npz
to avoid clobbering PR-D2's diffrax_24h_dt*.npz files.
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
REF_24H = ROOT / "tests" / "reference" / "sweep_24h_skip_soaexch_no_pcarbon_aging"
REF_IC = ROOT / "tests" / "reference" / "per_process_full_minus_pcarbon_aging"
ART = Path(__file__).resolve().parent / "_artifacts"
ART.mkdir(exist_ok=True)

TOTAL_24H_S = 86400
DT_LIST = (300, 30, 5, 1)

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


def main() -> None:
    snapshot = {
        k: np.asarray(v)
        for k, v in np.load(REF_IC / "calcsize_before.npz").items()
    }
    print("H2SO4-only 24h ablation (soaexch off in both JAX and Fortran)")
    print(f"{'dt':>4} {'nstep':>6} {'wall(s)':>10} "
          f"{'h2so4_gas max':>15} {'so4_aer accum max':>20}")
    print("-" * 80)

    for dt in DT_LIST:
        nstep = TOTAL_24H_S // dt
        state = _build_state(snapshot, dt)
        t0 = time.time()
        traj = run_timesteps(state, n_steps=nstep)
        t_wall = time.time() - t0

        j = _extract_jax(traj, nstep)
        f = _load_fortran(dt, nstep)
        rel_h = np.abs(j["h2so4_gas"] - f["h2so4_gas"]) / np.maximum(
            np.abs(f["h2so4_gas"]), 1e-300
        )
        rel_s = np.abs(j["so4_aer"][1] - f["so4_aer"][1]) / np.maximum(
            np.abs(f["so4_aer"][1]), 1e-300
        )
        print(
            f"{dt:>4d} {nstep:>6d} {t_wall:>10.1f} "
            f"{float(rel_h.max()):>15.3e} {float(rel_s.max()):>20.3e}"
        )

        np.savez(
            ART / f"h2so4_only_24h_dt{dt}.npz",
            dt=dt, nstep=nstep, t_wall=t_wall,
            j_num=j["num_aer"], j_so4=j["so4_aer"], j_soa=j["soa_aer"],
            j_h2so4=j["h2so4_gas"], j_soag=j["soag_gas"],
            f_num=f["num_aer"], f_so4=f["so4_aer"], f_soa=f["soa_aer"],
            f_h2so4=f["h2so4_gas"], f_soag=f["soag_gas"],
        )


if __name__ == "__main__":
    main()
