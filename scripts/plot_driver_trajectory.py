"""Render the M4 driver 60-step JAX-vs-Fortran trajectory figure.

Loads the M4 PR-A fixture (``tests/reference/per_process_full_minus_pcarbon_aging/``),
drives ``mam4_jax.driver.run_timesteps`` for 60 timesteps starting from
``calcsize_before[step=0]``, and writes
``docs/figures/driver_60step_trajectory.png``:

    Top — 4 panels (one per MAM4-MOM mode: accum / Aitken / coarse /
          primary_carbon). Each panel has dual y-axes:
            left  (log scale)  : number-density q[..., NUMPTR_AMODE[mode]]
                                 in #/kmol-air.
            right (linear)     : dry diameter dgncur_a[..., mode]
                                 in m.
          x-axis: timestep index (0..59).
          Fortran solid (lw 2.0), JAX dashed (lw 0.9, ls "--").
    Bottom — per-(step, tracer) |rel-err| for all 35 tracers,
          semilog y, with the ADR-003 1e-6 tolerance and float64 ε
          reference lines.

This is the mode-by-mode size-distribution comparison the owner asked
about before M4 landed. It is a *self-driven JAX trajectory* against
the Fortran capture (per ``feedback-validation-must-be-driven``), not
per-step JAX on captured before-states.

Usage:
    python scripts/plot_driver_trajectory.py
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
from mam4_jax import data
from mam4_jax.driver import run_timesteps

REF_DIR  = REPO_ROOT / "tests" / "reference" / "per_process_full_minus_pcarbon_aging"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "driver_60step_trajectory.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps

# Box-model namelist constants (same as tests/test_driver.py).
T_BOX_MODEL    = 273.0
PMID_BOX_MODEL = 1.0e5
DELTAT_60      = 30.0
ZMID_BOX_MODEL = 3.0e3
PBLH_BOX_MODEL = 1.1e3
RH_BOX_MODEL   = 0.9


def _build_state(snapshot: dict[str, np.ndarray], step: int):
    ncol, pver = snapshot["q"].shape[1], snapshot["q"].shape[2]
    return {
        "q":           jnp.asarray(snapshot["q"][step]),
        "qqcw":        jnp.asarray(snapshot["qqcw"][step]),
        "dgncur_a":    jnp.asarray(snapshot["dgncur_a"][step]),
        "dgncur_awet": jnp.asarray(snapshot["dgncur_awet"][step]),
        "qaerwat":     jnp.asarray(snapshot["qaerwat"][step]),
        "wetdens":     jnp.asarray(snapshot["wetdens"][step]),
        "t":           jnp.asarray(np.full((ncol, pver), T_BOX_MODEL)),
        "pmid":        jnp.asarray(np.full((ncol, pver), PMID_BOX_MODEL)),
        "cldn":        jnp.asarray(np.full((ncol, pver), 0.0)),
        "zmid":        jnp.asarray(np.full((ncol, pver), ZMID_BOX_MODEL)),
        "pblh":        jnp.asarray(np.full((ncol, pver), PBLH_BOX_MODEL)),
        "relhum":      jnp.asarray(np.full((ncol, pver), RH_BOX_MODEL)),
        "deltat":      jnp.asarray(DELTAT_60),
    }


def main() -> int:
    cb = {k: np.asarray(v) for k, v in np.load(REF_DIR / "calcsize_before.npz").items()}
    aw = {k: np.asarray(v) for k, v in np.load(REF_DIR / "amicphys_after_writeback.npz").items()}

    n_steps = 60
    steps = np.arange(n_steps)
    ic = _build_state(cb, step=0)

    print(f"Driving mam4_jax.driver.run_timesteps for {n_steps} steps ...")
    traj = run_timesteps(ic, n_steps=n_steps)

    # Trajectories: traj["q"] shape (n_steps, 1, 1, 35);
    # traj["dgncur_a"] shape (n_steps, 1, 1, 4).
    jq    = np.asarray(traj["q"])
    jdgn  = np.asarray(traj["dgncur_a"])
    fq    = aw["q"]
    fdgn  = aw["dgncur_a"]

    # --- Figure layout: 5 rows (4 mode panels + 1 rel-err panel) ----------
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(13, 11),
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )
    # Place modes in 2×2 grid, then use the 3rd row (both columns merged)
    # for the rel-err panel.
    mode_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    # Merge the bottom row for the rel-err panel.
    gs = axes[2, 0].get_gridspec()
    axes[2, 0].remove()
    axes[2, 1].remove()
    rel_ax = fig.add_subplot(gs[2, :])

    cmap = plt.get_cmap("tab10")
    color_num = cmap(0)   # blue for number
    color_dgn = cmap(3)   # red for dry diameter

    for mode_idx in range(4):
        ax_num = mode_axes[mode_idx]
        ax_dgn = ax_num.twinx()
        mode_name = data.MODE_NAMES[mode_idx]
        num_pcnst = int(data.NUMPTR_AMODE[mode_idx])

        f_num = fq[:, 0, 0, num_pcnst]
        j_num = jq[:, 0, 0, num_pcnst]
        f_dg  = fdgn[:, 0, 0, mode_idx]
        j_dg  = jdgn[:, 0, 0, mode_idx]

        ax_num.semilogy(steps, f_num, color=color_num, lw=2.0,
                        label=f"Fortran #/kmol-air")
        ax_num.semilogy(steps, j_num, color=color_num, lw=0.9, ls="--",
                        label=f"JAX     #/kmol-air")
        ax_num.set_ylabel("number  (#/kmol-air)", color=color_num)
        ax_num.tick_params(axis="y", labelcolor=color_num)

        ax_dgn.plot(steps, f_dg * 1.0e9, color=color_dgn, lw=2.0,
                    label=f"Fortran dgncur_a")
        ax_dgn.plot(steps, j_dg * 1.0e9, color=color_dgn, lw=0.9, ls="--",
                    label=f"JAX     dgncur_a")
        ax_dgn.set_ylabel("dry diameter  (nm)", color=color_dgn)
        ax_dgn.tick_params(axis="y", labelcolor=color_dgn)

        ax_num.set_xlabel("timestep index")
        ax_num.set_title(f"mode {mode_idx}: {mode_name}", fontsize=11)
        ax_num.grid(True, which="both", alpha=0.3)

    # --- Bottom: rel-err for all 35 tracers, semilog y ---------------------
    abs_diff = np.abs(jq - fq)
    ref_mag = np.maximum(np.abs(fq), 1e-300)
    rel = abs_diff / ref_mag                              # (n_steps, 1, 1, 35)
    rel_flat = rel[:, 0, 0, :]                            # (n_steps, 35)
    worst = float(rel_flat.max())
    worst_step, worst_tr = np.unravel_index(rel_flat.argmax(), rel_flat.shape)

    for itr in range(rel_flat.shape[-1]):
        rel_ax.semilogy(steps, np.maximum(rel_flat[:, itr], EPS),
                        lw=0.6, marker=",", alpha=0.55)
    rel_ax.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                    label=f"ADR-003 tol ({TOLERANCE:.0e})")
    rel_ax.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                    label=f"float64 ε ({EPS:.0e})")
    rel_ax.set_xlabel(f"timestep index ({n_steps} steps)")
    rel_ax.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    rel_ax.set_title(
        f"Per-(step, tracer) rel-err for all {rel_flat.shape[-1]} tracers — "
        f"worst {worst:.2e} at step {worst_step}, tracer {worst_tr}"
    )
    rel_ax.grid(True, which="both", alpha=0.3)
    rel_ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"M4 driver 60-step JAX-vs-Fortran trajectory "
        f"(per_process_full_minus_pcarbon_aging fixture)\n"
        f"worst trajectory rel-err: {worst:.2e}   "
        f"(ADR-003 tol = {TOLERANCE:.0e})",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_driver_trajectory] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  worst rel-err: {worst:.3e} at step {worst_step}, tracer {worst_tr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
