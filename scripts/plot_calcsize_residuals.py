"""Render the calcsize JAX-vs-Fortran residual figure.

Loads the 60-step full-transfer capture
(``tests/reference/per_process/``, which is what the canonical Fortran
box-model produces — ``do_aitacc_transfer=True``), runs the JAX port
with ``do_aitacc_transfer=True`` on each calcsize_before snapshot, and
writes ``docs/figures/calcsize_residuals.png``:

    Top:    dgncur_a evolution per mode across the 60 timesteps,
            JAX (dashed) overlaying Fortran (solid).
    Bottom: |rel-err| per (timestep, mode) with the ADR-003 1e-6
            tolerance and float64 epsilon reference lines.

The Aitken ↔ accum transfer never triggers in this fixture (see
``docs/DEFERRED.md``), so the figure visually matches what the
no-aitacc fixture would produce.

Usage:
    python scripts/plot_calcsize_residuals.py
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
from mam4_jax.processes.calcsize import calcsize

REF_DIR  = REPO_ROOT / "tests" / "reference" / "per_process"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "calcsize_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


def main() -> int:
    before = {k: np.asarray(v) for k, v in np.load(REF_DIR / "calcsize_before.npz").items()}
    after  = {k: np.asarray(v) for k, v in np.load(REF_DIR / "calcsize_after.npz").items()}

    state = {
        "q":        jnp.asarray(before["q"]),
        "qqcw":     jnp.asarray(before["qqcw"]),
        "dgncur_a": jnp.asarray(before["dgncur_a"]),
        "deltat":   jnp.asarray(30.0),
    }
    new = calcsize(state)

    dgn_jax = np.asarray(new["dgncur_a"], dtype=np.float64)[:, 0, 0, :]   # (nstep, m)
    dgn_ref = after["dgncur_a"][:, 0, 0, :]
    rel_err = np.abs(dgn_jax - dgn_ref) / np.maximum(np.abs(dgn_ref), 1e-30)

    nstep = dgn_ref.shape[0]
    steps = np.arange(1, nstep + 1)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8))

    cmap = plt.get_cmap("viridis")
    for m in range(4):
        color = cmap(m / 3)
        top.plot(steps, dgn_ref[:, m] * 1e6, color=color, lw=2.0,
                 label=f"Fortran {MODE_NAMES[m]}")
        top.plot(steps, dgn_jax[:, m] * 1e6, color=color, lw=0.9, ls="--",
                 label=f"JAX     {MODE_NAMES[m]}")
    top.set_yscale("log")
    top.set_xlabel("timestep")
    top.set_ylabel("dgncur_a  (µm)")
    top.set_title("calcsize: dgncur_a evolution over 60 timesteps")
    top.grid(True, which="both", alpha=0.3)
    top.legend(fontsize=8, ncol=2, loc="upper right")

    for m in range(4):
        color = cmap(m / 3)
        bot.semilogy(steps, np.maximum(rel_err[:, m], EPS),
                     color=color, lw=1.5, marker="o", ms=3,
                     label=MODE_NAMES[m])
    bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                label=f"ADR-003 tol ({TOLERANCE:.0e})")
    bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                label=f"float64 ε ({EPS:.0e})")
    bot.set_xlabel("timestep")
    bot.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    bot.set_title("per-(timestep, mode) relative error")
    bot.grid(True, which="both", alpha=0.3)
    bot.legend(fontsize=8, loc="upper right", ncol=2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_calcsize_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  max rel-err = {rel_err.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
