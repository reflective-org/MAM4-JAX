"""Render the rename JAX-vs-Fortran residual figure.

Loads the 60-step full-physics capture
(``tests/reference/per_process/rename_{before,after}.npz``), runs the
JAX port on each "before" snapshot, and writes
``docs/figures/rename_residuals.png``:

    Top:    Aitken-mode number evolution across the 60 timesteps,
            JAX (dashed) overlaying Fortran (solid). Rename's
            characteristic signature is the per-step drop in the
            Aitken-mode number (transferred to accum).
    Bottom: |rel-err| per (timestep, mode) for qnum_cur with the
            ADR-003 1e-6 tolerance and float64 epsilon reference lines.

Usage:
    python scripts/plot_rename_residuals.py
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
from mam4_jax.data import ACCUM_MODE_IDX, AITKEN_MODE_IDX, MODE_NAMES
from mam4_jax.processes.amicphys import _mam_rename_1subarea

REF_DIR  = REPO_ROOT / "tests" / "reference" / "per_process"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "rename_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


def main() -> int:
    before = np.load(REF_DIR / "rename_before.npz", allow_pickle=False)
    after  = np.load(REF_DIR / "rename_after.npz",  allow_pickle=False)

    nstep = int(before["istep"].shape[0])
    qnum_jax = np.zeros_like(after["qnum_cur"])
    for t in range(nstep):
        qnum_out, _, _ = _mam_rename_1subarea(
            jnp.asarray(before["qnum_cur"][t]),
            jnp.asarray(before["qaer_cur"][t]),
            jnp.asarray(before["qaer_delsub_grow4rnam"][t]),
            jnp.asarray(before["qwtr_cur"][t]),
            jnp.asarray(before["fac_m2v_aer"][t]),
        )
        qnum_jax[t] = np.asarray(qnum_out)

    qnum_ref = after["qnum_cur"]
    rel_err = np.abs(qnum_jax - qnum_ref) / np.maximum(np.abs(qnum_ref), 1e-25)
    steps = np.arange(1, nstep + 1)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8))

    # Top: per-step before → after for the two active modes (Aitken loses,
    # accum gains). The other modes are unchanged.
    cmap = plt.get_cmap("viridis")
    for label, mode_idx, color_frac in (
        ("Aitken (mfrm)", AITKEN_MODE_IDX, 0.25),
        ("accum (mtoo)",  ACCUM_MODE_IDX,  0.75),
    ):
        color = cmap(color_frac)
        top.plot(steps, qnum_ref[:, mode_idx], color=color, lw=2.0,
                 label=f"Fortran {MODE_NAMES[mode_idx]}")
        top.plot(steps, qnum_jax[:, mode_idx], color=color, lw=0.9, ls="--",
                 label=f"JAX     {MODE_NAMES[mode_idx]}")
    top.set_xlabel("timestep")
    top.set_ylabel("qnum_cur  (particles / kmol-air)")
    top.set_title("rename: per-mode number after rename, 60 timesteps")
    top.grid(True, which="both", alpha=0.3)
    top.legend(fontsize=9, loc="best")

    # Bottom: rel-err per (timestep, mode). Loop over the modes that
    # actually carry number — primary_carbon, coarse don't participate in
    # this rename pair so the rel-err there is 0 / max(0,1e-25) = 0 → eps.
    n_modes = qnum_ref.shape[1]
    for m in range(min(n_modes, 4)):
        color = cmap(m / 3)
        bot.semilogy(steps, np.maximum(rel_err[:, m], EPS),
                     color=color, lw=1.5, marker="o", ms=3,
                     label=MODE_NAMES[m] if m < len(MODE_NAMES) else f"mode {m}")
    bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                label=f"ADR-003 tol ({TOLERANCE:.0e})")
    bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                label=f"float64 ε ({EPS:.0e})")
    bot.set_xlabel("timestep")
    bot.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    bot.set_title("per-(timestep, mode) relative error on qnum_cur")
    bot.grid(True, which="both", alpha=0.3)
    bot.legend(fontsize=8, loc="upper right", ncol=2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_rename_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  max rel-err (qnum) = {rel_err.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
