"""Render the soaexch JAX-vs-Fortran residual figure.

Loads the 60-step gasaerexch+soaexch fixture
(``tests/reference/per_process_gasaerexch/``), runs the JAX
orchestration with ``mdo_gasaerexch=1, others=0``, and writes
``docs/figures/soaexch_residuals.png``:

    Top:    SOA gas + SOA aerosol mass per active mode (accum,
            aitken, coarse) over the 60 timesteps, JAX (dashed)
            overlaying Fortran (solid).
    Bottom: |rel-err| per (timestep, SOA-modified tracer) vs. ADR-003's
            1e-6 tolerance + float64 epsilon reference lines.

The fixture is the same one as the gasaerexch figure but here we
filter the bottom panel to highlight the *SOA-specific* tracers — the
H₂SO₄ + so4-mass divergences are already covered by
``gasaerexch_residuals.png``.

Usage:
    python scripts/plot_soaexch_residuals.py
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

REF_DIR  = REPO_ROOT / "tests" / "reference" / "per_process_gasaerexch"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "soaexch_residuals.png"

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
        "deltat":      jnp.asarray(30.0),
    }
    new = amicphys(state, mdo_gasaerexch=1, mdo_rename=0,
                   mdo_newnuc=0, mdo_coag=0)
    q_jax = np.asarray(new["q"])
    q_ref = aw["q"]
    steps = np.arange(1, nstep + 1)

    # SOA tracer indices: SOA gas + SOA aerosol mass in each mode that
    # actually carries SOA (excluding the pcarbon sentinel).
    soa_gas_idx = int(data.LMAP_GAS[data.AMICPHYS_IAER_SOA])
    soa_aer_indices = {}
    for m in range(data.NTOT_AMODE):
        idx = int(data.LMAP_AER[m, data.AMICPHYS_IAER_SOA])
        if idx >= 0:
            soa_aer_indices[idx] = f"SOA aer ({data.MODE_NAMES[m]})"

    label_for = {soa_gas_idx: "SOA gas"}
    label_for.update(soa_aer_indices)
    plotted_indices = [soa_gas_idx] + sorted(soa_aer_indices)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8))

    # Top: time-series of SOA tracers.
    cmap = plt.get_cmap("viridis")
    for i, idx in enumerate(plotted_indices):
        color = cmap(i / max(len(plotted_indices) - 1, 1))
        label = label_for[idx]
        top.plot(steps, q_ref[:, 0, 0, idx], color=color, lw=2.0,
                 label=f"Fortran {label}")
        top.plot(steps, q_jax[:, 0, 0, idx], color=color, lw=0.9, ls="--",
                 label=f"JAX     {label}")
    top.set_yscale("log")
    top.set_xlabel("timestep")
    top.set_ylabel("q  (mass mixing ratio)")
    top.set_title("soaexch: SOA gas + per-mode SOA aerosol mass, 60 steps")
    top.grid(True, which="both", alpha=0.3)
    top.legend(fontsize=8, ncol=2, loc="best")

    # Bottom: per-(timestep, SOA-tracer) rel-err.
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
    bot.set_title("per-(timestep, tracer) relative error — SOA-specific tracers")
    bot.grid(True, which="both", alpha=0.3)
    bot.legend(fontsize=8, loc="best", ncol=2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_soaexch_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  SOA tracer indices plotted: {plotted_indices}")
    print(f"  max rel-err: {rel_err[..., plotted_indices].max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
