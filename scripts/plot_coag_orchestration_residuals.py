"""Render the coag orchestration JAX-vs-Fortran residual figure.

Loads the 60-step coag-only fixture (``tests/reference/per_process_coag/``),
runs the JAX orchestration with ``mdo_coag=1, others=0``, and writes
``docs/figures/coag_orchestration_residuals.png``:

    Top:    Per-mode number-density time series — Aitken (q[17]),
            pcarbon (q[22]), accum (q[34]) — over 60 timesteps, JAX
            (dashed) over Fortran (solid). These are the three modes
            directly affected by coag's number-loss cascade.
    Bottom: Per-(timestep, tracer-index) |rel-err| for all 33 aerosol
            slots (gas slots excluded — see test_amicphys.py for the
            rationale). Reference lines at ADR-003 1e-6 and float64 ε.

Usage:
    python scripts/plot_coag_orchestration_residuals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mam4_jax  # noqa: F401  - enables jax_enable_x64
from mam4_jax import data
from mam4_jax.processes.amicphys import amicphys

REF_DIR  = REPO_ROOT / "tests" / "reference" / "per_process_coag"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "coag_orchestration_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


def _rel_err(jax_arr: np.ndarray, ref_arr: np.ndarray) -> np.ndarray:
    return np.abs(jax_arr - ref_arr) / np.maximum(np.abs(ref_arr), 1e-300)


def main() -> int:
    before = {k: np.asarray(v) for k, v in
              np.load(REF_DIR / "amicphys_before.npz").items()}
    aw     = {k: np.asarray(v) for k, v in
              np.load(REF_DIR / "amicphys_after_writeback.npz").items()}

    nstep, ncol, pver, _ = before["q"].shape
    state = {
        "q":           jnp.asarray(before["q"]),
        "qqcw":        jnp.asarray(before["qqcw"]),
        "dgncur_a":    jnp.asarray(before["dgncur_a"]),
        "dgncur_awet": jnp.asarray(before["dgncur_awet"]),
        "qaerwat":     jnp.asarray(before["qaerwat"]),
        "wetdens":     jnp.asarray(before["wetdens"]),
        "t":           jnp.asarray(np.full((nstep, ncol, pver), 273.0)),
        "pmid":        jnp.asarray(np.full((nstep, ncol, pver), 1.0e5)),
        "cldn":        jnp.asarray(np.full((nstep, ncol, pver), 0.0)),
        "zmid":        jnp.asarray(np.full((nstep, ncol, pver), 3.0e3)),
        "pblh":        jnp.asarray(np.full((nstep, ncol, pver), 1.1e3)),
        "relhum":      jnp.asarray(np.full((nstep, ncol, pver), 0.9)),
        "deltat":      jnp.asarray(30.0),
    }
    new_state = amicphys(state, mdo_gasaerexch=0, mdo_rename=0,
                         mdo_newnuc=0, mdo_coag=1)

    jq_full = np.asarray(new_state["q"])
    fq_full = aw["q"]

    # Per-mode number tracer indices (NUMPTR_AMODE).
    # MAM4-MOM order: accum=17, aitken=22, coarse=30, primary_carbon=34
    # ...wait, that's wrong. Let's pull from data.
    nait_idx = int(data.NUMPTR_AMODE[data.AITKEN_MODE_IDX])
    nacc_idx = int(data.NUMPTR_AMODE[data.ACCUM_MODE_IDX])
    npca_idx = int(data.NUMPTR_AMODE[data.PCARBON_MODE_IDX])

    steps = np.arange(nstep)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8))

    # --- Top: per-mode number time series -----------------------------------
    cmap = plt.get_cmap("viridis")
    series = [
        ("Aitken  number  (q[%d])" % nait_idx,  nait_idx, cmap(0.10)),
        ("pcarbon number  (q[%d])" % npca_idx,  npca_idx, cmap(0.50)),
        ("accum   number  (q[%d])" % nacc_idx,  nacc_idx, cmap(0.85)),
    ]
    for label, idx, color in series:
        ref = fq_full[:, 0, 0, idx]
        jxr = jq_full[:, 0, 0, idx]
        top.plot(steps, ref, color=color, lw=2.0,
                 label=f"Fortran {label}")
        top.plot(steps, jxr, color=color, lw=0.9, ls="--",
                 label=f"JAX     {label}")
    top.set_xlabel("timestep index")
    top.set_ylabel("number mixing ratio (#/kmol-air)")
    top.set_yscale("log")
    top.set_title("coag orchestration: per-mode number-density trajectories")
    top.grid(True, which="both", alpha=0.3)
    top.legend(fontsize=8, ncol=2, loc="best")

    # --- Bottom: per-record rel-err for all 33 aerosol slots ---------------
    gas_slots = set(int(i) for i in data.LMAP_GAS)
    pcnst = before["q"].shape[-1]
    aerosol_slots = [i for i in range(pcnst) if i not in gas_slots]

    worst = 0.0
    for itr in aerosol_slots:
        rel = _rel_err(jq_full[:, 0, 0, itr], fq_full[:, 0, 0, itr])
        worst = max(worst, float(rel.max()))
        bot.semilogy(steps, np.maximum(rel, EPS),
                     lw=0.6, marker=",", alpha=0.55)
    bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                label=f"ADR-003 tol ({TOLERANCE:.0e})")
    bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                label=f"float64 ε ({EPS:.0e})")
    bot.set_xlabel(f"timestep index ({nstep} steps)")
    bot.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    bot.set_title(
        f"Per-(step, tracer) rel-err across {len(aerosol_slots)} aerosol slots "
        f"(gas slots {sorted(gas_slots)} excluded — see test docstring)"
    )
    bot.grid(True, which="both", alpha=0.3)
    bot.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"coag orchestration JAX vs Fortran — worst aerosol-slot rel-err: "
        f"{worst:.2e}    (ADR-003 tol = {TOLERANCE:.0e})",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_coag_orchestration_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  worst aerosol-slot rel-err: {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
