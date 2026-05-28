"""1000-simulation wall-time benchmark: Fortran vs JAX (diffrax + JIT + scan).

Both implementations are run at the canonical MAM4 box-model
configuration: dt=30 s, nstep=60, 1800 s total trajectory window.
Each "simulation" runs the full operator-splitting sequence for
60 substeps.

Fortran side:
- Uses the existing mam_box_test.exe built with skip_pcarbon_aging
  (matches the sweep_no_pcarbon_aging fixture).
- 1000 subprocess invocations; each call writes mam_output.nc and
  is overwritten by the next. The final NetCDF is saved as the
  rel-err reference.
- Wall time includes subprocess startup (~50 ms) — this is the
  honest "cost of one Fortran simulation" from a user's POV.

JAX side:
- Uses run_timesteps (scan + JIT'd run_step from M6 PR-J1/J2).
- One warmup call (excluded from stats) so the JIT cache is hot.
- 1000 in-process calls; wall time per call measured with
  time.time() and block_until_ready().

Outputs:
- scripts/_artifacts/benchmark_1000sims.npz with timings + Fortran
  reference + one JAX trajectory.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from textwrap import dedent

import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

import mam4_jax  # noqa: F401
from mam4_jax import data
from mam4_jax.driver import run_timesteps


ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "mam4-original-src-code" / "run"
EXE = RUN_DIR / "mam_box_test.exe"
REF_IC = ROOT / "tests" / "reference" / "per_process_full_minus_pcarbon_aging"
ART = Path(__file__).resolve().parent / "_artifacts"
ART.mkdir(exist_ok=True)

# Canonical default-MAM4-timestep box-model run.
DT = 30
NSTEP = 60
N_TRIALS = 1000

T_BOX, P_BOX, ZMID, PBLH, RH = 273.0, 1.0e5, 3.0e3, 1.1e3, 0.9

NAMELIST = dedent(f"""\
    &time_input
    mam_dt    = {DT},
    mam_nstep = {NSTEP},
    /
    &cntl_input
    mdo_gaschem    = 0,
    mdo_gasaerexch = 1,
    mdo_rename     = 1,
    mdo_newnuc     = 1,
    mdo_coag       = 1,
    /
    &met_input
    temp    = 273.,
    press   = 1.e5,
    RH_CLEA = 0.9,
    /
    &chem_input
    numc1=1.e8, numc2=1.e9, numc3=1.e5, numc4=2.e8,
    mfso41=0.3, mfpom1=0., mfsoa1=0.3, mfbc1=0., mfdst1=0., mfncl1=0.4,
    mfso42=0.3, mfsoa2=0.3, mfncl2=0.4,
    mfdst3=0., mfncl3=0.4, mfso43=0.3, mfbc3=0., mfpom3=0., mfsoa3=0.3,
    mfpom4=0., mfbc4=1.,
    qso2=1.e-4, qh2so4=1.e-13, qsoag=5.e-10,
    /
    &size_parameters
    dgnum1=0.1100e-6, dgnum2=0.0260e-6, dgnum3=2.000e-6, dgnum4=0.050e-6,
    sigmag1=1.800, sigmag2=1.600, sigmag3=1.800, sigmag4=1.600,
    /
""")


def _build_jax_state():
    snap = {
        k: np.asarray(v)
        for k, v in np.load(REF_IC / "calcsize_before.npz").items()
    }
    ncol, pver = snap["q"].shape[1], snap["q"].shape[2]
    return {
        "q": jnp.asarray(snap["q"][0]),
        "qqcw": jnp.asarray(snap["qqcw"][0]),
        "dgncur_a": jnp.asarray(snap["dgncur_a"][0]),
        "dgncur_awet": jnp.asarray(snap["dgncur_awet"][0]),
        "qaerwat": jnp.asarray(snap["qaerwat"][0]),
        "wetdens": jnp.asarray(snap["wetdens"][0]),
        "t": jnp.asarray(np.full((ncol, pver), T_BOX)),
        "pmid": jnp.asarray(np.full((ncol, pver), P_BOX)),
        "cldn": jnp.asarray(np.full((ncol, pver), 0.0)),
        "zmid": jnp.asarray(np.full((ncol, pver), ZMID)),
        "pblh": jnp.asarray(np.full((ncol, pver), PBLH)),
        "relhum": jnp.asarray(np.full((ncol, pver), RH)),
        "deltat": jnp.asarray(float(DT)),
    }


def _parse_fortran_nc(path: Path) -> dict[str, np.ndarray]:
    ds = nc.Dataset(path, "r")
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


def _extract_jax(traj) -> dict[str, np.ndarray]:
    # Lookup indices for mass tracers
    def _idx(species_type):
        out = []
        for m in range(data.NTOT_AMODE):
            type_row = data.LSPECTYPE_AMODE[m]
            mass_row = data.LMASSPTR_AMODE[m]
            slot = next(
                (s for s, t in enumerate(type_row) if t == species_type), -1
            )
            out.append(int(mass_row[slot]) if slot >= 0 else -1)
        return out

    so4_idx = _idx(0)
    soa_idx = _idx(4)
    h2so4_pcnst = int(data.LMAP_GAS[1])
    soag_pcnst = int(data.LMAP_GAS[0])

    j_q = np.asarray(traj["q"])
    return {
        "num_aer": np.stack(
            [j_q[:, 0, 0, int(data.NUMPTR_AMODE[m])] for m in range(4)], axis=0
        ),
        "so4_aer": np.stack(
            [j_q[:, 0, 0, so4_idx[m]] if so4_idx[m] >= 0 else np.zeros(NSTEP)
             for m in range(4)], axis=0
        ),
        "soa_aer": np.stack(
            [j_q[:, 0, 0, soa_idx[m]] if soa_idx[m] >= 0 else np.zeros(NSTEP)
             for m in range(4)], axis=0
        ),
        "h2so4_gas": j_q[:, 0, 0, h2so4_pcnst],
        "soag_gas": j_q[:, 0, 0, soag_pcnst],
    }


def benchmark_fortran() -> tuple[np.ndarray, Path]:
    """Run mam_box_test.exe N_TRIALS times; return per-call wall times
    and the final NetCDF path."""
    assert EXE.exists(), f"Fortran executable not found at {EXE}"
    (RUN_DIR / "namelist").write_text(NAMELIST)
    nc_out = RUN_DIR / "mam_output.nc"

    print(f"Fortran: running {N_TRIALS} subprocess invocations of "
          f"{EXE.name} at dt={DT}, nstep={NSTEP} ...")
    timings = np.empty(N_TRIALS, dtype=np.float64)
    for i in range(N_TRIALS):
        t0 = time.perf_counter()
        subprocess.run(
            ["./mam_box_test.exe"], cwd=RUN_DIR, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        timings[i] = time.perf_counter() - t0
        if (i + 1) % 100 == 0:
            print(f"  fortran {i+1}/{N_TRIALS}: {timings[i]:.3f}s "
                  f"(running mean {timings[:i+1].mean():.3f}s)")
    return timings, nc_out


def benchmark_jax() -> tuple[np.ndarray, dict]:
    """Run run_timesteps N_TRIALS times; return per-call wall times and
    one trajectory."""
    state = _build_jax_state()
    print(f"JAX: warmup (compile) at dt={DT}, nstep={NSTEP} ...")
    t0 = time.perf_counter()
    traj_warmup = run_timesteps(state, n_steps=NSTEP)
    traj_warmup["q"].block_until_ready()
    print(f"  warmup wall: {time.perf_counter()-t0:.3f}s")

    print(f"JAX: running {N_TRIALS} cached calls of run_timesteps ...")
    timings = np.empty(N_TRIALS, dtype=np.float64)
    saved_traj = None
    for i in range(N_TRIALS):
        t0 = time.perf_counter()
        traj = run_timesteps(state, n_steps=NSTEP)
        traj["q"].block_until_ready()
        timings[i] = time.perf_counter() - t0
        if i == 0:
            saved_traj = {k: np.asarray(v) for k, v in traj.items()}
        if (i + 1) % 100 == 0:
            print(f"  jax {i+1}/{N_TRIALS}: {timings[i]*1000:.2f}ms "
                  f"(running mean {timings[:i+1].mean()*1000:.2f}ms)")

    return timings, saved_traj


def main() -> None:
    print(f"Benchmark: {N_TRIALS} simulations at dt={DT}s, "
          f"nstep={NSTEP} (1800 s window)")
    print("=" * 70)

    jax_times, jax_traj = benchmark_jax()
    fortran_times, fortran_nc = benchmark_fortran()

    # Parse the final Fortran NetCDF for rel-err.
    fortran_state = _parse_fortran_nc(fortran_nc)
    jax_state = _extract_jax(jax_traj)

    out = ART / "benchmark_1000sims.npz"
    np.savez(
        out,
        dt=DT, nstep=NSTEP, n_trials=N_TRIALS,
        jax_times=jax_times,
        fortran_times=fortran_times,
        j_num=jax_state["num_aer"], j_so4=jax_state["so4_aer"],
        j_soa=jax_state["soa_aer"], j_h2so4=jax_state["h2so4_gas"],
        j_soag=jax_state["soag_gas"],
        f_num=fortran_state["num_aer"], f_so4=fortran_state["so4_aer"],
        f_soa=fortran_state["soa_aer"], f_h2so4=fortran_state["h2so4_gas"],
        f_soag=fortran_state["soag_gas"],
    )
    print()
    print(f"Saved {out}")
    print()
    print("Summary (wall time per simulation):")
    print(f"  JAX     median {np.median(jax_times)*1000:.2f}ms, "
          f"P5 {np.percentile(jax_times, 5)*1000:.2f}ms, "
          f"P95 {np.percentile(jax_times, 95)*1000:.2f}ms, "
          f"max {jax_times.max()*1000:.2f}ms")
    print(f"  Fortran median {np.median(fortran_times)*1000:.2f}ms, "
          f"P5 {np.percentile(fortran_times, 5)*1000:.2f}ms, "
          f"P95 {np.percentile(fortran_times, 95)*1000:.2f}ms, "
          f"max {fortran_times.max()*1000:.2f}ms")
    print(f"  Median speedup JAX vs Fortran: "
          f"{np.median(fortran_times) / np.median(jax_times):.2f}x")


if __name__ == "__main__":
    main()
