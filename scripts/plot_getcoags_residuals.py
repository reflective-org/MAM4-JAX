"""Render the getcoags leaf-function JAX-vs-Fortran residual figure.

Loads ``tests/reference/coag_coefficients/reference.npz`` (output of
``scripts/reference_drivers/coag_coefficients_driver.F90``), runs the
JAX port :func:`mam4_jax.coag.getcoags`, and writes
``docs/figures/getcoags_residuals.png``:

Eight panels (4×2) — one per coagulation coefficient
(``qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12``). Each panel is a
log-log JAX-vs-Fortran scatter (240 points), colored by the
diameter-ratio index ``n1`` (Whitby table bucket); the panel title
quotes that coefficient's worst |rel-err|.

Usage:
    python scripts/plot_getcoags_residuals.py
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
from mam4_jax.coag import getcoags

REF_PATH = REPO_ROOT / "tests" / "reference" / "coag_coefficients" / "reference.npz"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "getcoags_residuals.png"

TOLERANCE = 1e-6
SG_ATK = 1.6
SG_ACC = 1.8
DLGSQT2 = 1.0 / np.log(np.sqrt(2.0))


def _rel_err(jax_arr: np.ndarray, ref_arr: np.ndarray) -> np.ndarray:
    return np.abs(jax_arr - ref_arr) / np.maximum(np.abs(ref_arr), 1e-300)


def main() -> int:
    d = {k: np.asarray(v) for k, v in np.load(REF_PATH).items()}

    dgnumA = d["dgnumA"]
    dgnumB = d["dgnumB"]
    sgatk = np.full_like(dgnumA, SG_ATK)
    sgacc = np.full_like(dgnumA, SG_ACC)

    qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12 = getcoags(
        jnp.asarray(d["lamda"]),
        jnp.asarray(d["kfmatac"]),
        jnp.asarray(d["kfmat"]),
        jnp.asarray(d["kfmac"]),
        jnp.asarray(d["knc"]),
        jnp.asarray(dgnumA),
        jnp.asarray(dgnumB),
        jnp.asarray(sgatk),
        jnp.asarray(sgacc),
        jnp.log(jnp.asarray(sgatk)),
        jnp.log(jnp.asarray(sgacc)),
    )

    # n1 = Whitby bucket on the diameter ratio sqrt(dgnumB/dgnumA).
    # The function uses dgacc/dgatk directly; reconstruct so the colour
    # mapping matches the table-lookup regime visually.
    rat = dgnumB / dgnumA
    n1 = np.clip(np.rint(1.0 + DLGSQT2 * np.log(rat)).astype(int), 1, 10)

    series = [
        ("qs11", np.asarray(qs11), d["qs11"]),
        ("qn11", np.asarray(qn11), d["qn11"]),
        ("qs22", np.asarray(qs22), d["qs22"]),
        ("qn22", np.asarray(qn22), d["qn22"]),
        ("qs12", np.asarray(qs12), d["qs12"]),
        ("qs21", np.asarray(qs21), d["qs21"]),
        ("qn12", np.asarray(qn12), d["qn12"]),
        ("qv12", np.asarray(qv12), d["qv12"]),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(11, 13))
    cmap = plt.get_cmap("viridis")
    worst_overall = 0.0
    for ax, (name, jax_v, ref_v) in zip(axes.ravel(), series):
        rel = _rel_err(jax_v, ref_v)
        worst = float(rel.max())
        worst_overall = max(worst_overall, worst)

        floor = max(np.min(np.abs(ref_v[ref_v != 0])) / 10.0, 1e-300)
        sc = ax.scatter(
            np.maximum(np.abs(ref_v), floor),
            np.maximum(np.abs(jax_v), floor),
            c=n1, cmap=cmap, vmin=1, vmax=10,
            s=14, alpha=0.85, edgecolor="none",
        )
        lo = min(np.abs(ref_v[ref_v != 0]).min(),
                 np.abs(jax_v[jax_v != 0]).min())
        hi = max(np.abs(ref_v).max(), np.abs(jax_v).max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"Fortran {name}")
        ax.set_ylabel(f"JAX {name}")
        ax.set_title(f"{name}    max |rel-err| = {worst:.2e}",
                     fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"getcoags JAX vs Fortran — worst rel-err across 240 records × 8 outputs: "
        f"{worst_overall:.2e}    (ADR-003 tol = {TOLERANCE:.0e})",
        fontsize=11, y=1.00,
    )
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6,
                         pad=0.02, ticks=range(1, 11))
    cbar.set_label("Whitby table index n1 (diameter-ratio bucket)")

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_getcoags_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  worst rel-err overall: {worst_overall:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
