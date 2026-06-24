"""Render the gasaerexch JAX-vs-Fortran residual figure.

Loads the 60-step single-toggle capture
(``tests/reference/per_process_gasaerexch_only/amicphys_{before,after_writeback}.npz``),
runs the JAX orchestration with ``mdo_gasaerexch=1, others=0``, and
writes ``docs/figures/gasaerexch_residuals.png``:

    Top:    H₂SO₄ gas + so4 mass per active mode (accum, aitken, coarse)
            over the 60 timesteps, JAX (dashed) overlaying Fortran (solid).
    Bottom: |rel-err| per (timestep, modified-tracer) vs. ADR-003's 1e-6
            tolerance + float64 epsilon reference lines.

Usage:
    python scripts/plot_gasaerexch_residuals.py
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

REF_DIR  = REPO_ROOT / "tests" / "reference" / "per_process_gasaerexch_only"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "gasaerexch_residuals.png"

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

    # Identify which pcnst indices gasaerexch *meaningfully* modifies.
    # We include tracers whose Fortran-side change is above 1e-20
    # (excludes FP-noise updates on tracers that should be exactly zero).
    diff_amplitude = np.abs(q_ref - before["q"]).max(axis=(0, 1, 2))
    ref_amplitude  = np.abs(q_ref).max(axis=(0, 1, 2))
    modified_idx = np.where((diff_amplitude > 1e-20) & (ref_amplitude > 1e-25))[0]

    # Friendly tracer labels keyed by pcnst index.
    so4_modes = {int(data.LMAP_AER[m, 1]): f"so4 ({data.MODE_NAMES[m]})"
                 for m in range(data.NTOT_AMODE) if data.LMAP_AER[m, 1] >= 0}
    gas_h2so4 = int(data.LMAP_GAS[1])
    label_for = {gas_h2so4: "H₂SO₄ gas"}
    label_for.update(so4_modes)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8))

    # Top: time series of the principal tracers (H2SO4 gas + so4 per mode).
    cmap = plt.get_cmap("viridis")
    principal = [gas_h2so4] + sorted(so4_modes)
    for i, idx in enumerate(principal):
        color = cmap(i / max(len(principal) - 1, 1))
        label = label_for.get(idx, f"q[{idx}]")
        top.plot(steps, q_ref[:, 0, 0, idx], color=color, lw=2.0,
                 label=f"Fortran {label}")
        top.plot(steps, q_jax[:, 0, 0, idx], color=color, lw=0.9, ls="--",
                 label=f"JAX     {label}")
    top.set_yscale("log")
    top.set_xlabel("timestep")
    top.set_ylabel("q  (mass / number mixing ratio)")
    top.set_title("gasaerexch (no SOA): principal tracer evolution, 60 steps")
    top.grid(True, which="both", alpha=0.3)
    top.legend(fontsize=8, ncol=2, loc="best")

    # Bottom: per-(timestep, tracer) rel-err for the modified-tracer set.
    rel_err = np.abs(q_jax - q_ref) / np.maximum(np.abs(q_ref), 1e-30)
    palette = [cmap(j / max(len(modified_idx) - 1, 1)) for j in range(len(modified_idx))]
    for j, idx in enumerate(modified_idx):
        bot.semilogy(steps, np.maximum(rel_err[:, 0, 0, idx], EPS),
                     color=palette[j], lw=1.5, marker="o", ms=3,
                     label=label_for.get(idx, f"q[{idx}]"))
    bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                label=f"ADR-003 tol ({TOLERANCE:.0e})")
    bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                label=f"float64 ε ({EPS:.0e})")
    bot.set_xlabel("timestep")
    bot.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    bot.set_title("per-(timestep, tracer) relative error — gasaerexch-modified pcnst slots")
    bot.grid(True, which="both", alpha=0.3)
    bot.legend(fontsize=8, loc="best", ncol=2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_gasaerexch_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  modified pcnst indices: {modified_idx.tolist()}")
    print(f"  max rel-err across modified: {rel_err[..., modified_idx].max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
