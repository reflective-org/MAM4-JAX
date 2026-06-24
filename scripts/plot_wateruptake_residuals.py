"""Render the wateruptake JAX-vs-Fortran residual figure.

Loads tests/reference/per_process/wateruptake_{before,after}.npz, runs
the JAX port on the captured inputs, and writes
docs/figures/wateruptake_residuals.png:

    Top:    Bar chart of dgncur_a (input dry diameter) and dgncur_awet
            (Fortran solid + JAX hatched) per mode.
    Middle: log-scale qaerwat per mode (Fortran + JAX).
    Bottom: |rel-err| per output variable per mode, with the ADR-003
            1e-6 tolerance and float64 epsilon reference lines.

Usage:
    python scripts/plot_wateruptake_residuals.py
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
from mam4_jax.data import MODE_NAMES
from mam4_jax.processes.wateruptake import wateruptake

REF_DIR = REPO_ROOT / "tests" / "reference" / "per_process"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "wateruptake_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


def main() -> int:
    before = {k: np.asarray(v) for k, v in np.load(REF_DIR / "wateruptake_before.npz").items()}
    after  = {k: np.asarray(v) for k, v in np.load(REF_DIR / "wateruptake_after.npz").items()}

    state = {
        "q":        jnp.asarray(before["q"][0]),
        "dgncur_a": jnp.asarray(before["dgncur_a"][0]),
        "t":        jnp.asarray(np.full((1, 1), 273.0)),
        "pmid":     jnp.asarray(np.full((1, 1), 1.0e5)),
        "cldn":     jnp.asarray(np.full((1, 1), 0.0)),
    }
    new = wateruptake(state)

    dgncur_a    = np.asarray(state["dgncur_a"]).ravel()
    awet_jax    = np.asarray(new["dgncur_awet"]).ravel()
    awet_fort   = after["dgncur_awet"][0].ravel()
    qaer_jax    = np.asarray(new["qaerwat"]).ravel()
    qaer_fort   = after["qaerwat"][0].ravel()
    wdens_jax   = np.asarray(new["wetdens"]).ravel()
    wdens_fort  = after["wetdens"][0].ravel()

    mode_idx = np.arange(4)
    bar_w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Top-left: dry vs wet diameters per mode.
    ax = axes[0, 0]
    ax.bar(mode_idx - bar_w / 2, dgncur_a, bar_w, label="dry (input)", color="C0", alpha=0.6)
    ax.bar(mode_idx + bar_w / 2, awet_fort, bar_w, label="wet (Fortran)", color="C1")
    ax.bar(mode_idx + bar_w / 2, awet_jax, bar_w, label="wet (JAX)",
           edgecolor="black", hatch="//", fill=False, lw=1.5)
    ax.set_yscale("log")
    ax.set_xticks(mode_idx)
    ax.set_xticklabels(MODE_NAMES, rotation=20)
    ax.set_ylabel("mode diameter (m)")
    ax.set_title("dgncur_a (dry) vs dgncur_awet (wet)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", axis="y", alpha=0.3)

    # Top-right: aerosol water content.
    ax = axes[0, 1]
    qaer_jax_p   = np.maximum(qaer_jax, 1e-30)
    qaer_fort_p  = np.maximum(qaer_fort, 1e-30)
    ax.bar(mode_idx - bar_w / 2, qaer_fort_p, bar_w, label="Fortran", color="C2")
    ax.bar(mode_idx + bar_w / 2, qaer_jax_p, bar_w, label="JAX",
           edgecolor="black", hatch="//", fill=False, lw=1.5)
    ax.set_yscale("log")
    ax.set_xticks(mode_idx)
    ax.set_xticklabels(MODE_NAMES, rotation=20)
    ax.set_ylabel("aerosol water content (kg/kg)")
    ax.set_title("qaerwat per mode")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", axis="y", alpha=0.3)

    # Bottom-left: wet density per mode.
    ax = axes[1, 0]
    ax.bar(mode_idx - bar_w / 2, wdens_fort, bar_w, label="Fortran", color="C3")
    ax.bar(mode_idx + bar_w / 2, wdens_jax, bar_w, label="JAX",
           edgecolor="black", hatch="//", fill=False, lw=1.5)
    ax.set_xticks(mode_idx)
    ax.set_xticklabels(MODE_NAMES, rotation=20)
    ax.set_ylabel("wet density (kg/m³)")
    ax.set_title("wetdens per mode")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Bottom-right: relative errors per mode per output variable.
    ax = axes[1, 1]
    def rel(j, f):
        return np.abs(j - f) / np.maximum(np.abs(f), 1e-30)
    r_awet  = rel(awet_jax,  awet_fort)
    r_qaer  = rel(qaer_jax,  qaer_fort)
    r_wdens = rel(wdens_jax, wdens_fort)
    width = 0.25
    ax.bar(mode_idx - width, np.maximum(r_awet, EPS),  width, label="dgncur_awet", color="C0")
    ax.bar(mode_idx,         np.maximum(r_qaer, EPS),  width, label="qaerwat",    color="C1")
    ax.bar(mode_idx + width, np.maximum(r_wdens, EPS), width, label="wetdens",    color="C2")
    ax.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
               label=f"ADR-003 tol ({TOLERANCE:.0e})")
    ax.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
               label=f"float64 ε ({EPS:.0e})")
    ax.set_yscale("log")
    ax.set_xticks(mode_idx)
    ax.set_xticklabels(MODE_NAMES, rotation=20)
    ax.set_ylabel("|JAX − Fortran| / |Fortran|")
    ax.set_title("Per-mode relative error")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, which="both", axis="y", alpha=0.3)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_wateruptake_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  max rel-err dgncur_awet = {r_awet.max():.3e}")
    print(f"  max rel-err qaerwat     = {r_qaer.max():.3e}")
    print(f"  max rel-err wetdens     = {r_wdens.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
