"""Plot M8 PR-K2 cloudchem residuals — JAX vs Fortran on the
per-process cloudchem fixture.

2×3 grid per the PR-K2 framing discussion:

  Row 1 (gases, vmr-space):
    H2SO4         | SO2             | SOAG (negative control)
    Fortran solid + JAX dashed on each.

  Row 2:
    per-mode SO4_cw     | JAX-vs-Fortran  | max rel-err per step,
    (accum / aitken /   | scatter for     | log y, with reference
    coarse — pcarbon    | all 7 tracers   | lines at ADR-003 1e-6
    has no sulfate)     | (diagonal check)| and ADR-015 3 %.

Run:
    python scripts/plot_cloudchem_residuals.py
Output:
    docs/figures/cloudchem_residuals.png
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mam4_jax import data
from mam4_jax.processes.cloudchem import cloudchem_simple_sub


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "reference" / "per_process_cloudchem"
FIG = ROOT / "docs" / "figures" / "cloudchem_residuals.png"

CLDN_FIXTURE = 0.5
DT_FIXTURE   = 30.0
N_STEPS      = 60


def main() -> None:
    before = np.load(FIXTURE_DIR / "cloudchem_before.npz")
    after  = np.load(FIXTURE_DIR / "cloudchem_after.npz")
    cldn = jnp.full((1, 1), CLDN_FIXTURE)

    # Run JAX cloudchem on all 60 fixture steps.
    jax_vmr_list   = []
    jax_vmrcw_list = []
    for step in range(N_STEPS):
        vmr_out, vmrcw_out = cloudchem_simple_sub(
            jnp.asarray(before["vmr"][step]),
            jnp.asarray(before["vmrcw"][step]),
            cldn, DT_FIXTURE,
        )
        jax_vmr_list.append(np.asarray(vmr_out))
        jax_vmrcw_list.append(np.asarray(vmrcw_out))
    jax_vmr   = np.stack(jax_vmr_list)
    jax_vmrcw = np.stack(jax_vmrcw_list)
    fort_vmr   = after["vmr"]
    fort_vmrcw = after["vmrcw"]

    t_min = np.arange(N_STEPS) * DT_FIXTURE / 60.0  # 0, 0.5, 1, ..., 29.5 min

    # 7 tracers for the scatter + rel-err panels:
    # 3 gases + 2 written cloud-borne SO4 (accum, aitken)
    # + 1 unwritten cloud-borne SO4 (coarse — negative control)
    # + 1 read-only cloud-borne number (accum, used in tmpd fraction).
    tracers = [
        ("H2SO4 (gas)",     fort_vmr[...,   data.VMR_H2SO4],
                            jax_vmr[...,    data.VMR_H2SO4]),
        ("SO2 (gas)",       fort_vmr[...,   data.VMR_SO2],
                            jax_vmr[...,    data.VMR_SO2]),
        ("SOAG (gas)",      fort_vmr[...,   data.VMR_SOAG],
                            jax_vmr[...,    data.VMR_SOAG]),
        ("SO4_cw accum",    fort_vmrcw[..., data.VMRCW_SO4[data.ACCUM_MODE_IDX]],
                            jax_vmrcw[...,  data.VMRCW_SO4[data.ACCUM_MODE_IDX]]),
        ("SO4_cw aitken",   fort_vmrcw[..., data.VMRCW_SO4[data.AITKEN_MODE_IDX]],
                            jax_vmrcw[...,  data.VMRCW_SO4[data.AITKEN_MODE_IDX]]),
        ("SO4_cw coarse",   fort_vmrcw[..., data.VMRCW_SO4[data.COARSE_MODE_IDX]],
                            jax_vmrcw[...,  data.VMRCW_SO4[data.COARSE_MODE_IDX]]),
        ("num_cw accum",    fort_vmrcw[..., data.VMRCW_NUM[data.ACCUM_MODE_IDX]],
                            jax_vmrcw[...,  data.VMRCW_NUM[data.ACCUM_MODE_IDX]]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # ----- Row 1: gases (vmr-space, mol/mol)
    for ax, (name, f_arr, j_arr) in zip(
        axes[0], tracers[:3]
    ):
        ax.plot(t_min, f_arr[:, 0, 0], "k-",  label="Fortran", lw=2.0)
        ax.plot(t_min, j_arr[:, 0, 0], "r--", label="JAX",     lw=1.2)
        ax.set_title(name)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("vmr (mol/mol)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    # ----- Row 2 col 0: per-mode SO4_cw (accum / aitken / coarse)
    ax = axes[1, 0]
    mode_colors = {
        data.ACCUM_MODE_IDX:  ("tab:blue",   "accum"),
        data.AITKEN_MODE_IDX: ("tab:green",  "aitken"),
        data.COARSE_MODE_IDX: ("tab:red",    "coarse (untouched)"),
    }
    for mode_idx, (col, label) in mode_colors.items():
        slot = data.VMRCW_SO4[mode_idx]
        if slot < 0:
            continue
        ax.plot(t_min, fort_vmrcw[:, 0, 0, slot], "-",  color=col,
                label=f"Fortran {label}", lw=2.0)
        ax.plot(t_min, jax_vmrcw[:,  0, 0, slot], "--", color=col,
                label=f"JAX {label}",     lw=1.2)
    ax.set_title("SO4 cloud-borne, per mode")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("vmrcw (mol/mol)")
    ax.set_yscale("symlog", linthresh=1e-30)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # ----- Row 2 col 1: scatter, all 7 tracers (JAX y vs Fortran x)
    # Note: zero-valued points are masked from the axis-range computation
    # (a value of 0 can't be plotted on a log-log axis). Most notably,
    # vmrcw[SO4_cw_accum / aitken] is 0 at the first fixture step (before
    # cloudchem fires); those points are dropped from the diagonal bounds
    # but still scattered (matplotlib silently drops the log-of-zero ones).
    # Expected on-screen point count ≈ 60 × 7 − (zero entries) < 420.
    ax = axes[1, 1]
    cmap = plt.cm.tab10
    all_min, all_max = np.inf, -np.inf
    for i, (name, f_arr, j_arr) in enumerate(tracers):
        flat_f = f_arr.flatten()
        flat_j = j_arr.flatten()
        ax.scatter(flat_f, flat_j, s=12, alpha=0.6,
                   color=cmap(i), label=name)
        mask = (flat_f > 0) & (flat_j > 0)
        if mask.any():
            all_min = min(all_min, float(np.min(flat_f[mask])),
                                   float(np.min(flat_j[mask])))
            all_max = max(all_max, float(np.max(flat_f[mask])),
                                   float(np.max(flat_j[mask])))
    if np.isfinite(all_min) and np.isfinite(all_max):
        diag = np.array([all_min, all_max])
        ax.plot(diag, diag, "k:", lw=0.8, label="y = x")
    ax.set_title("JAX vs Fortran (sit-on-diagonal = bit-clean)")
    ax.set_xlabel("Fortran")
    ax.set_ylabel("JAX")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    # ----- Row 2 col 2: max rel-err per step, log y, with reference lines
    ax = axes[1, 2]
    # Compute relative error per tracer per step.
    rel_err_per_step = np.zeros(N_STEPS)
    for step in range(N_STEPS):
        worst = 0.0
        for name, f_arr, j_arr in tracers:
            ref = max(abs(float(f_arr[step, 0, 0])), 1e-30)
            err = abs(float(j_arr[step, 0, 0]) - float(f_arr[step, 0, 0])) / ref
            worst = max(worst, err)
        # Replace exact zero by a very small positive value so log plotting works.
        rel_err_per_step[step] = max(worst, 1e-18)
    ax.plot(np.arange(1, N_STEPS + 1), rel_err_per_step,
            "o-", color="tab:purple", markersize=4, lw=1.2)
    ax.axhline(1e-6, color="goldenrod", ls=":",
               label="ADR-003 (1e-6)")
    ax.axhline(3e-2, color="crimson",   ls=":",
               label="ADR-015 (3%)")
    ax.set_title("max rel-err vs Fortran (all 7 tracers)")
    ax.set_xlabel("step")
    ax.set_ylabel("max rel-err (log)")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"M8 PR-K2: cloudchem_simple_sub — per-process JAX vs Fortran "
        f"(cldn = {CLDN_FIXTURE}, dt = {DT_FIXTURE:.0f}s, "
        f"{N_STEPS} per-step comparisons, vmr-space). "
        f"NOT a JAX-driven trajectory — each point is JAX cloudchem "
        f"applied to a Fortran-captured before-state.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120, bbox_inches="tight")
    print(f"Saved {FIG}")

    # One-line summary so the CLI invocation surfaces the actual bar.
    print(f"\nmax rel-err across 60 steps × 7 tracers: "
          f"{rel_err_per_step.max():.3e}")


if __name__ == "__main__":
    main()
