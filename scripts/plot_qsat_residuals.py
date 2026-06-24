"""Render the qsat JAX-vs-Fortran residual figure.

Reads tests/reference/qsat/reference.npz, runs the JAX port over the
same (T, p) grid, and writes a four-panel PNG to
docs/figures/qsat_residuals.png:

    Top row:    qs_water(T) and qs_ice(T), one curve per pressure level,
                JAX overlaying Fortran.
    Bottom row: |rel-err| vs T per pressure level, with the 1e-6
                ADR-003 line and the float64 epsilon floor.

Usage:
    python scripts/plot_qsat_residuals.py
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
from mam4_jax.saturation import qsat_ice, qsat_water

REFERENCE_NPZ = REPO_ROOT / "tests" / "reference" / "qsat" / "reference.npz"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "qsat_residuals.png"

TOLERANCE = 1e-6   # ADR-003
EPS = np.finfo(np.float64).eps


def main() -> int:
    ref = np.load(REFERENCE_NPZ)
    T = ref["T"]
    p = ref["p"]
    fw, fi = ref["qs_water"], ref["qs_ice"]

    jw = np.asarray(qsat_water(jnp.asarray(T), jnp.asarray(p)), dtype=np.float64)
    ji = np.asarray(qsat_ice(jnp.asarray(T), jnp.asarray(p)), dtype=np.float64)

    rel_w = np.abs(jw - fw) / np.abs(fw)
    rel_i = np.abs(ji - fi) / np.abs(fi)

    p_levels = np.unique(p)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for col, (name, fortran, jax_val, rel, label) in enumerate([
        ("water", fw, jw, rel_w, "qsat_water"),
        ("ice",   fi, ji, rel_i, "qsat_ice"),
    ]):
        top, bot = axes[0, col], axes[1, col]
        for pl in p_levels:
            mask = p == pl
            top.semilogy(T[mask], fortran[mask], lw=2.0,
                         label=f"Fortran  p={pl:.0e} Pa")
            top.semilogy(T[mask], jax_val[mask], lw=0.9, ls="--",
                         label=f"JAX      p={pl:.0e} Pa")
            bot.semilogy(T[mask], np.maximum(rel[mask], EPS), lw=1.2,
                         label=f"p={pl:.0e} Pa")
        top.set_title(f"{label}: JAX vs Fortran reference")
        top.set_ylabel("saturation specific humidity  qs  [kg/kg]")
        top.legend(loc="lower right", fontsize=7, ncol=2)
        top.grid(True, which="both", alpha=0.3)

        bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                    label=f"ADR-003 tolerance ({TOLERANCE:.0e})")
        bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                    label=f"float64 ε ({EPS:.0e})")
        bot.set_xlabel("temperature  T  [K]")
        bot.set_ylabel("relative error  |JAX − Fortran| / |Fortran|")
        bot.legend(loc="upper right", fontsize=7)
        bot.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_qsat_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  max rel-err water = {rel_w.max():.3e}")
    print(f"  max rel-err ice   = {rel_i.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
