"""Render the M5 12-point convergence-sweep JAX-vs-Fortran figure.

For each ``nstep`` in the canonical sweep ``(1, 2, 4, 9, 18, 30, 60,
120, 180, 360, 900, 1800)``:

1. Drive ``mam4_jax.driver.run_timesteps(ic, nstep)`` with
   ``state["deltat"] = 1800/nstep``.
2. Load the matching Fortran NetCDF from
   ``tests/reference/sweep_no_pcarbon_aging/``.
3. Compare the final-state and per-step rel-err.

Writes ``docs/figures/sweep_convergence.png``:

    Top-left  — per-mode final-step number density (Fortran solid /
                JAX dashed) vs nstep (log x). 4 mode colors.
    Top-right — final-step H₂SO₄ gas (Fortran solid / JAX dashed)
                vs nstep.
    Bottom    — worst |JAX − Fortran|/|Fortran| per nstep (semilog y)
                across all tracers and timesteps, with ADR-003 1e-6
                reference. The ``nstep <= 30`` half is shaded as
                "PR-E2 deferred (adaptive SOA substepping)" — JAX
                diverges there until ``mam_soaexch_1subarea``'s
                ``dtcur = alpha_astem/tmpa`` adaptive substepping is
                ported.

Usage:
    python scripts/plot_sweep_convergence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default; JAX_ENABLE_X64=0 to opt out
from mam4_jax import data
from mam4_jax.driver import run_timesteps

REF_SWEEP_DIR = REPO_ROOT / "tests" / "reference" / "sweep_no_pcarbon_aging"
REF_IC_DIR    = REPO_ROOT / "tests" / "reference" / "per_process_full_minus_pcarbon_aging"
FIG_PATH      = REPO_ROOT / "docs" / "figures" / "sweep_convergence.png"

NSTEP_SWEEP = (1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800)
NSTEP_PR_E2_BOUNDARY = 60  # nstep < 60 requires PR-E2
TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps
TOTAL_DURATION_S = 1800

T_BOX_MODEL    = 273.0
PMID_BOX_MODEL = 1.0e5
ZMID_BOX_MODEL = 3.0e3
PBLH_BOX_MODEL = 1.1e3
RH_BOX_MODEL   = 0.9
H2SO4_PCNST_IDX = int(data.LMAP_GAS[1])


def _build_state(snapshot: dict[str, np.ndarray], deltat: float):
    ncol, pver = snapshot["q"].shape[1], snapshot["q"].shape[2]
    return {
        "q":           jnp.asarray(snapshot["q"][0]),
        "qqcw":        jnp.asarray(snapshot["qqcw"][0]),
        "dgncur_a":    jnp.asarray(snapshot["dgncur_a"][0]),
        "dgncur_awet": jnp.asarray(snapshot["dgncur_awet"][0]),
        "qaerwat":     jnp.asarray(snapshot["qaerwat"][0]),
        "wetdens":     jnp.asarray(snapshot["wetdens"][0]),
        "t":           jnp.asarray(np.full((ncol, pver), T_BOX_MODEL)),
        "pmid":        jnp.asarray(np.full((ncol, pver), PMID_BOX_MODEL)),
        "cldn":        jnp.asarray(np.full((ncol, pver), 0.0)),
        "zmid":        jnp.asarray(np.full((ncol, pver), ZMID_BOX_MODEL)),
        "pblh":        jnp.asarray(np.full((ncol, pver), PBLH_BOX_MODEL)),
        "relhum":      jnp.asarray(np.full((ncol, pver), RH_BOX_MODEL)),
        "deltat":      jnp.asarray(deltat),
    }


def main() -> int:
    cb = {k: np.asarray(v)
          for k, v in np.load(REF_IC_DIR / "calcsize_before.npz").items()}

    # Collect final-step values + per-nstep worst rel-err.
    f_final_num    = np.zeros((len(NSTEP_SWEEP), 4))    # (n_sweep, mode)
    j_final_num    = np.zeros_like(f_final_num)
    f_final_h2so4  = np.zeros(len(NSTEP_SWEEP))
    j_final_h2so4  = np.zeros_like(f_final_h2so4)
    worst_per_step = np.zeros(len(NSTEP_SWEEP))

    for i, nstep in enumerate(NSTEP_SWEEP):
        deltat = TOTAL_DURATION_S // nstep
        print(f"  nstep={nstep:5d} dt={deltat:5d}s ...", flush=True)
        state = _build_state(cb, deltat=float(deltat))
        traj = run_timesteps(state, n_steps=nstep)

        ds = nc.Dataset(REF_SWEEP_DIR / f"mam_dt{deltat}_ndt{nstep}.nc", "r")
        try:
            f_num   = np.asarray(ds.variables["num_aer"][:])    # (4, nstep)
            f_h2so4 = np.asarray(ds.variables["h2so4_gas"][:])  # (nstep,)
        finally:
            ds.close()

        j_q = np.asarray(traj["q"])                                # (nstep, 1, 1, 35)
        j_num = np.stack(
            [j_q[:, 0, 0, int(data.NUMPTR_AMODE[m])] for m in range(4)],
            axis=0)
        j_h2so4 = j_q[:, 0, 0, H2SO4_PCNST_IDX]

        f_final_num[i, :]   = f_num[:, -1]
        j_final_num[i, :]   = j_num[:, -1]
        f_final_h2so4[i]    = f_h2so4[-1]
        j_final_h2so4[i]    = j_h2so4[-1]

        rel_num = (np.abs(j_num - f_num)
                   / np.maximum(np.abs(f_num), 1e-300))
        rel_h2  = (np.abs(j_h2so4 - f_h2so4)
                   / np.maximum(np.abs(f_h2so4), 1e-300))
        worst_per_step[i] = float(max(rel_num.max(), rel_h2.max()))

    # --- Figure layout: 2 top panels + 1 bottom panel ---------------------
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_num   = fig.add_subplot(gs[0, 0])
    ax_h2so4 = fig.add_subplot(gs[0, 1])
    ax_rel   = fig.add_subplot(gs[1, :])

    nsteps = np.asarray(NSTEP_SWEEP)
    cmap = plt.get_cmap("tab10")
    mode_names = ("accum", "Aitken", "coarse", "pcarbon")

    # Top-left: per-mode number-density at final step vs nstep.
    for m in range(4):
        color = cmap(m)
        ax_num.loglog(nsteps, f_final_num[:, m], color=color, lw=2.0,
                      marker="o", markersize=4,
                      label=f"Fortran {mode_names[m]}")
        ax_num.loglog(nsteps, j_final_num[:, m], color=color, lw=0.9,
                      ls="--", marker="x", markersize=5,
                      label=f"JAX     {mode_names[m]}")
    ax_num.set_xlabel("nstep")
    ax_num.set_ylabel("final-step number-density (#/kmol-air)")
    ax_num.set_title("Per-mode number-density convergence")
    ax_num.grid(True, which="both", alpha=0.3)
    ax_num.legend(fontsize=7, ncol=2, loc="best")
    ax_num.set_xticks(nsteps)
    ax_num.set_xticklabels([str(n) for n in nsteps], rotation=45, fontsize=8)

    # Top-right: H2SO4 gas at final step vs nstep.
    ax_h2so4.loglog(nsteps, f_final_h2so4, color="C0", lw=2.0,
                    marker="o", markersize=5, label="Fortran")
    ax_h2so4.loglog(nsteps, j_final_h2so4, color="C0", lw=0.9,
                    ls="--", marker="x", markersize=6, label="JAX")
    ax_h2so4.set_xlabel("nstep")
    ax_h2so4.set_ylabel(r"final-step $q_{\rm H_2SO_4}$  (mol/mol-air)")
    ax_h2so4.set_title("H₂SO₄ gas convergence")
    ax_h2so4.grid(True, which="both", alpha=0.3)
    ax_h2so4.legend(fontsize=9, loc="best")
    ax_h2so4.set_xticks(nsteps)
    ax_h2so4.set_xticklabels([str(n) for n in nsteps], rotation=45, fontsize=8)

    # Bottom: worst rel-err per nstep, semilog y.
    ax_rel.semilogy(nsteps, np.maximum(worst_per_step, EPS),
                    color="C3", lw=1.5, marker="o", markersize=6,
                    label="worst rel-err (num_aer + H₂SO₄)")
    ax_rel.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                   label=f"ADR-003 tol ({TOLERANCE:.0e})")
    ax_rel.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                   label=f"float64 ε ({EPS:.0e})")
    # Shade the PR-E2-deferred region.
    ax_rel.axvspan(0, NSTEP_PR_E2_BOUNDARY - 1, alpha=0.15, color="orange",
                   label="PR-E2 deferred (adaptive SOA substepping)")
    ax_rel.set_xlabel("nstep")
    ax_rel.set_ylabel(r"max $|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    ax_rel.set_title("Per-nstep worst rel-err across all timesteps")
    ax_rel.set_xscale("log")
    ax_rel.set_xticks(nsteps)
    ax_rel.set_xticklabels([str(n) for n in nsteps], rotation=45, fontsize=8)
    ax_rel.grid(True, which="both", alpha=0.3)
    ax_rel.legend(fontsize=9, loc="best")

    # Determine M5-pass region.
    nstep_ok = nsteps[nsteps >= NSTEP_PR_E2_BOUNDARY]
    worst_ok = worst_per_step[nsteps >= NSTEP_PR_E2_BOUNDARY]
    worst_ok_max = float(worst_ok.max()) if worst_ok.size else float("nan")

    fig.suptitle(
        f"M5 12-point convergence sweep — nstep ≥ 60: worst rel-err "
        f"{worst_ok_max:.2e}  (ADR-003 tol = {TOLERANCE:.0e})\n"
        f"nstep ≤ 30 deferred to PR-E2 (adaptive SOA substepping)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_sweep_convergence] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  nstep>=60 worst rel-err: {worst_ok_max:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
