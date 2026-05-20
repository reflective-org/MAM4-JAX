"""Render the kohler JAX-vs-Fortran residual figure.

Reads tests/reference/kohler/reference.npz, runs the JAX port over the
same (rdry, hygro, s) grid, and writes a four-panel PNG to
docs/figures/kohler_residuals.png:

    Top row:    growth factor rwet/rdry vs RH for each hygroscopicity,
                with one curve per dry-radius value, JAX (dashed) over
                Fortran (solid).
    Bottom panel: |rel-err| vs case index for all 168 points, with the
                ADR-003 1e-6 line and the float64 epsilon floor.

Usage:
    python scripts/plot_kohler_residuals.py
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
from mam4_jax.kohler import modal_aero_kohler

REFERENCE_NPZ = REPO_ROOT / "tests" / "reference" / "kohler" / "reference.npz"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "kohler_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


def main() -> int:
    ref = np.load(REFERENCE_NPZ)
    rdry = ref["rdry_in"]
    hygro = ref["hygro"]
    s = ref["s"]
    rwet_f = ref["rwet"]

    rwet_j = np.asarray(
        modal_aero_kohler(jnp.asarray(rdry), jnp.asarray(hygro), jnp.asarray(s)),
        dtype=np.float64,
    )

    rel_err = np.abs(rwet_j - rwet_f) / np.abs(rwet_f)

    hygro_levels = sorted(set(hygro.tolist()))
    rdry_levels = sorted(set(rdry.tolist()))

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    cmap = plt.get_cmap("viridis")

    for i, h in enumerate(hygro_levels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        for j, rd in enumerate(rdry_levels):
            color = cmap(j / max(1, len(rdry_levels) - 1))
            mask = (hygro == h) & (rdry == rd)
            if not np.any(mask):
                continue
            s_pts = s[mask]
            gf_f = rwet_f[mask] / rd
            gf_j = rwet_j[mask] / rd
            order = np.argsort(s_pts)
            ax.semilogy(s_pts[order], gf_f[order], color=color, lw=2.0,
                        label=f"rdry={rd:.0e} m")
            ax.semilogy(s_pts[order], gf_j[order], color=color, lw=0.9, ls="--")
        ax.set_title(f"hygro = {h:.2f}")
        ax.set_xlabel("relative humidity  s")
        ax.set_ylabel("growth factor  rwet / rdry")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", ncol=2)

    ax = fig.add_subplot(gs[2, :])
    idx = np.arange(len(rel_err))
    ax.semilogy(idx, np.maximum(rel_err, EPS), "o", ms=4, color="C0")
    ax.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
               label=f"ADR-003 tolerance ({TOLERANCE:.0e})")
    ax.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
               label=f"float64 ε ({EPS:.0e})")
    ax.set_title("|rel-err| per (rdry, hygro, s) test point")
    ax.set_xlabel("test case index")
    ax.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| \, / \, |{\rm Fortran}|$")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_kohler_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  max rel-err = {rel_err.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
