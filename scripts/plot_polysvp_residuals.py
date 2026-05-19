"""Render the polysvp JAX-vs-Fortran residual figure.

Reads tests/reference/polysvp/reference.npz, runs the JAX port over the
same T sweep, and writes a two-panel PNG to docs/figures/polysvp_residuals.png:

    Top:    e_sat(T) for water and ice, on log scale, both JAX and Fortran.
    Bottom: |rel-err| vs T for both branches, with the 1e-6 ADR-003 line.

Usage:
    python scripts/plot_polysvp_residuals.py
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
from mam4_jax.saturation import polysvp_ice, polysvp_water
REFERENCE_NPZ = REPO_ROOT / "tests" / "reference" / "polysvp" / "reference.npz"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "polysvp_residuals.png"

TOLERANCE = 1e-6  # ADR-003

EPS = np.finfo(np.float64).eps


def main() -> int:
    ref = np.load(REFERENCE_NPZ)
    T = ref["T"]
    fortran_water = ref["esat_water"]
    fortran_ice = ref["esat_ice"]

    jax_water = np.asarray(polysvp_water(jnp.asarray(T)), dtype=np.float64)
    jax_ice = np.asarray(polysvp_ice(jnp.asarray(T)), dtype=np.float64)

    rel_water = np.abs(jax_water - fortran_water) / np.abs(fortran_water)
    rel_ice = np.abs(jax_ice - fortran_ice) / np.abs(fortran_ice)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    top.semilogy(T, fortran_water, color="C0", lw=2.5, label="Fortran  water")
    top.semilogy(T, fortran_ice, color="C1", lw=2.5, label="Fortran  ice")
    top.semilogy(T, jax_water, color="C0", lw=1, ls="--", label="JAX      water")
    top.semilogy(T, jax_ice, color="C1", lw=1, ls="--", label="JAX      ice")
    top.set_ylabel("saturation vapor pressure  $e_\\mathrm{sat}$  [Pa]")
    top.set_title("polysvp: JAX port vs Fortran reference")
    top.legend(loc="lower right", fontsize=9)
    top.grid(True, which="both", alpha=0.3)

    # Floor rel-err at machine epsilon so log-scale doesn't lose zeros.
    bot.semilogy(T, np.maximum(rel_water, EPS),
                 color="C0", lw=1.5, label="water")
    bot.semilogy(T, np.maximum(rel_ice, EPS),
                 color="C1", lw=1.5, label="ice")
    bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                label=f"ADR-003 tolerance ({TOLERANCE:.0e})")
    bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                label=f"float64 $\\varepsilon$ ({EPS:.0e})")
    bot.set_xlabel("temperature  T  [K]")
    bot.set_ylabel("relative error  |JAX - Fortran| / |Fortran|")
    bot.legend(loc="upper right", fontsize=9)
    bot.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_polysvp_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  max rel-err water = {rel_water.max():.3e}")
    print(f"  max rel-err ice   = {rel_ice.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
