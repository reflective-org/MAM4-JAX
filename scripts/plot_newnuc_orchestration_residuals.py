"""Render the newnuc orchestration JAX-vs-Fortran residual figure.

Loads the 60-step gasaerexch+newnuc fixture
(``tests/reference/per_process_gasaerexch_and_newnuc/``), runs the JAX
orchestration with ``mdo_gasaerexch=1, mdo_newnuc=1, others=0``, and
writes ``docs/figures/newnuc_orchestration_residuals.png``:

    Top:    H₂SO₄ gas + Aitken-mode number + Aitken-mode so4 mass time
            series over 60 timesteps, JAX (dashed) over Fortran (solid).
            These are the tracers newnuc directly modifies, layered on
            top of gasaerexch's H₂SO₄ depletion and so4 deposition.
    Bottom: Per-(timestep, tracer-index) |rel-err| for the newnuc-affected
            tracers vs ADR-003 1e-6 and float64 ε reference lines.

Usage:
    python scripts/plot_newnuc_orchestration_residuals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default; JAX_ENABLE_X64=0 to opt out
from mam4_jax import data
from mam4_jax.processes.amicphys import amicphys

REF_DIR  = REPO_ROOT / "tests" / "reference" / "per_process_gasaerexch_and_newnuc"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "newnuc_orchestration_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


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
    new = amicphys(state, mdo_gasaerexch=1, mdo_rename=0,
                   mdo_newnuc=1, mdo_coag=0)
    q_jax = np.asarray(new["q"])
    q_ref = aw["q"]
    steps = np.arange(1, nstep + 1)

    # Tracers newnuc directly modifies (plus the H2SO4 gas which both
    # gasaerexch and newnuc touch).
    h2so4_idx = int(data.LMAP_GAS[1])
    nait_num_idx = int(data.LMAP_NUM[data.AITKEN_MODE_IDX])
    nait_so4_idx = int(data.LMAP_AER[data.AITKEN_MODE_IDX,
                                      data.AMICPHYS_IAER_SOA + 1])  # iaer_so4 = SOA + 1

    label_for = {
        h2so4_idx:    "H₂SO₄ gas",
        nait_num_idx: "Aitken number",
        nait_so4_idx: "Aitken so4 mass",
    }
    plotted_indices = [h2so4_idx, nait_num_idx, nait_so4_idx]

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8))

    # Top: time series. Each tracer has its own y-scale magnitude, so
    # show as semilog with thin overlay lines. Twin axes would be tidier
    # but harder to read with three series.
    cmap = plt.get_cmap("viridis")
    for j, idx in enumerate(plotted_indices):
        color = cmap(j / max(len(plotted_indices) - 1, 1))
        label = label_for[idx]
        top.semilogy(steps, q_ref[:, 0, 0, idx], color=color, lw=2.0,
                     label=f"Fortran {label}")
        top.semilogy(steps, q_jax[:, 0, 0, idx], color=color, lw=0.9, ls="--",
                     label=f"JAX     {label}")
    top.set_xlabel("timestep")
    top.set_ylabel("tracer value")
    top.set_title("newnuc orchestration: H₂SO₄ + Aitken-mode number + so4 mass, 60 steps")
    top.grid(True, which="both", alpha=0.3)
    top.legend(fontsize=8, ncol=2, loc="best")

    # Bottom: per-(timestep, tracer) rel-err for the plotted tracers.
    rel_err = np.abs(q_jax - q_ref) / np.maximum(np.abs(q_ref), 1e-30)
    palette = [cmap(j / max(len(plotted_indices) - 1, 1))
               for j in range(len(plotted_indices))]
    for j, idx in enumerate(plotted_indices):
        bot.semilogy(steps, np.maximum(rel_err[:, 0, 0, idx], EPS),
                     color=palette[j], lw=1.5, marker="o", ms=3,
                     label=label_for[idx])
    bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                label=f"ADR-003 tol ({TOLERANCE:.0e})")
    bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                label=f"float64 ε ({EPS:.0e})")
    bot.set_xlabel("timestep")
    bot.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    bot.set_title("per-(timestep, tracer) relative error — newnuc-affected tracers")
    bot.grid(True, which="both", alpha=0.3)
    bot.legend(fontsize=8, loc="best", ncol=2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_newnuc_orchestration_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  newnuc tracer indices: {plotted_indices}")
    print(f"  max rel-err: {rel_err[..., plotted_indices].max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
