"""Render the makoh JAX-vs-Fortran residual figure.

Reads tests/reference/makoh/reference.npz, runs the JAX port on the same
polynomial test cases, and writes a two-panel PNG to
docs/figures/makoh_residuals.png:

    Top:    log10 |JAX - Fortran| (absolute error of the complex root)
            vs test-case index, one marker series per root branch.
    Bottom: same, but relative error |Δ| / |Fortran root|.

Each test case appears at integer x; root branches are offset within the
case for legibility.

Usage:
    python scripts/plot_makoh_residuals.py
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
from mam4_jax.kohler import makoh_cubic, makoh_quartic

REFERENCE_NPZ = REPO_ROOT / "tests" / "reference" / "makoh" / "reference.npz"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "makoh_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


def _plot_panel(ax, jax_roots, fortran_roots, kind: str, n_branches: int) -> None:
    n_cases = jax_roots.shape[0]
    cases = np.arange(1, n_cases + 1)
    width = 0.8 / n_branches
    for b in range(n_branches):
        jx = jax_roots[:, b]
        fr = fortran_roots[:, b]
        if kind == "abs":
            err = np.abs(jx - fr)
        else:
            mag = np.abs(fr)
            err = np.where(mag > 0, np.abs(jx - fr) / np.maximum(mag, 1e-300),
                           np.abs(jx - fr))
        xs = cases + (b - (n_branches - 1) / 2) * width
        ax.semilogy(xs, np.maximum(err, EPS), "o", ms=8,
                    label=f"root {b + 1}")
    ax.axhline(TOLERANCE, color="red", ls=":", lw=1.2,
               label=f"ADR-003 tol ({TOLERANCE:.0e})")
    ax.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
               label=f"float64 ε ({EPS:.0e})")
    ax.set_xticks(cases)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)


def main() -> int:
    ref = np.load(REFERENCE_NPZ)

    ci, qi = ref["cubic_inputs"], ref["quartic_inputs"]
    jax_c = np.asarray(
        makoh_cubic(jnp.asarray(ci[:, 0]), jnp.asarray(ci[:, 1]), jnp.asarray(ci[:, 2])),
        dtype=np.complex128,
    )
    jax_q = np.asarray(
        makoh_quartic(jnp.asarray(qi[:, 0]), jnp.asarray(qi[:, 1]),
                      jnp.asarray(qi[:, 2]), jnp.asarray(qi[:, 3])),
        dtype=np.complex128,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    _plot_panel(axes[0, 0], jax_c, ref["cubic_roots"], "abs", 3)
    axes[0, 0].set_title("makoh_cubic — absolute complex error")
    axes[0, 0].set_ylabel(r"$|{\rm JAX} - {\rm Fortran}|$")

    _plot_panel(axes[1, 0], jax_c, ref["cubic_roots"], "rel", 3)
    axes[1, 0].set_title("makoh_cubic — relative error")
    axes[1, 0].set_ylabel(r"$|\Delta| \, / \, |{\rm Fortran}|$")
    axes[1, 0].set_xlabel("cubic test case index")

    _plot_panel(axes[0, 1], jax_q, ref["quartic_roots"], "abs", 4)
    axes[0, 1].set_title("makoh_quartic — absolute complex error")
    axes[0, 1].set_ylabel(r"$|{\rm JAX} - {\rm Fortran}|$")

    _plot_panel(axes[1, 1], jax_q, ref["quartic_roots"], "rel", 4)
    axes[1, 1].set_title("makoh_quartic — relative error")
    axes[1, 1].set_ylabel(r"$|\Delta| \, / \, |{\rm Fortran}|$")
    axes[1, 1].set_xlabel("quartic test case index")

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_makoh_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")

    for name, jx, fr in [("cubic", jax_c, ref["cubic_roots"]),
                         ("quartic", jax_q, ref["quartic_roots"])]:
        diff = np.abs(jx - fr)
        mag = np.abs(fr)
        rel = np.where(mag > 0, diff / np.maximum(mag, 1e-300), diff)
        print(f"  makoh_{name:7s}  max rel-err = {rel.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
