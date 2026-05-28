"""Plot the 1000-simulation benchmark: wall-time boxplot + rel-err
boxplots per the layout chosen for PR-J2 follow-up.

Reads scripts/_artifacts/benchmark_1000sims.npz produced by
scripts/benchmark_1000_sims.py.

Outputs:
- docs/figures/benchmark_walltime_1000sims.png
    Single boxplot, 2 boxes (Fortran vs JAX). Log y. 1000 trials each.
- docs/figures/benchmark_relerr_aerosols.png
    3-panel figure: num_aer / so4_aer / soa_aer, each with 4 per-mode
    box-and-whiskers. Distribution per box = rel-err at each of the 60
    timesteps of one canonical simulation.
- docs/figures/benchmark_relerr_gas.png
    2-panel figure: h2so4_gas / soag_gas, one box each.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = Path(__file__).resolve().parent / "_artifacts"
FIG_DIR = ROOT / "docs" / "figures"
CACHE = ART / "benchmark_1000sims.npz"

MODE_NAMES = ("Aitken", "accumulation", "primary-carbon", "coarse")


def _relerr(j: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.abs(j - f) / np.maximum(np.abs(f), 1e-300)


def _walltime_plot(d: dict) -> Path:
    jax_t = d["jax_times"] * 1000.0       # ms
    f_t = d["fortran_times"] * 1000.0     # ms
    n = int(d["n_trials"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bp = ax.boxplot(
        [f_t, jax_t], tick_labels=["Fortran", "JAX (diffrax + jit + scan)"],
        widths=0.5, patch_artist=True, showfliers=True, whis=(5, 95),
    )
    for patch, color in zip(bp["boxes"], ["tab:blue", "tab:red"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_ylabel("wall time per simulation (ms)")
    ax.set_title(
        f"Wall-time per 1800 s simulation (dt={int(d['dt'])} s, "
        f"nstep={int(d['nstep'])}) — {n} trials each\n"
        f"Median: Fortran {np.median(f_t):.1f} ms, JAX {np.median(jax_t):.2f} ms "
        f"(JAX {np.median(f_t)/np.median(jax_t):.1f}× faster)"
    )
    ax.grid(True, which="both", alpha=0.3, axis="y")
    ax.text(
        0.02, 0.02,
        "Note: each Fortran trial is a fresh subprocess; the wall time\n"
        "includes ~50–100 ms of `mam_box_test.exe` startup per invocation.\n"
        "JAX trial timings are in-process after a one-time JIT compile.",
        transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom",
        bbox={"facecolor": "lightyellow", "alpha": 0.7, "edgecolor": "gray"},
    )
    out = FIG_DIR / "benchmark_walltime_1000sims.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _aerosols_relerr_plot(d: dict) -> Path:
    """3 panels: num_aer / so4_aer / soa_aer. 4 box-and-whiskers per
    panel (one per mode). Distribution per box: rel-err at each of
    the 60 timesteps."""
    fields = (
        ("num_aer", d["j_num"], d["f_num"], "rel-err"),
        ("so4_aer", d["j_so4"], d["f_so4"], "rel-err"),
        ("soa_aer", d["j_soa"], d["f_soa"], "rel-err"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, j, f, ylabel) in zip(axes, fields):
        boxes = []
        labels = []
        for m in range(4):
            if not np.any(f[m]):
                # All zeros (e.g. coarse mode so4/soa) — skip with a
                # blank slot so x-axis still labels mode positions.
                boxes.append(np.array([np.nan]))
            else:
                rel = _relerr(j[m], f[m])
                boxes.append(np.maximum(rel, 1e-20))
            labels.append(MODE_NAMES[m])
        bp = ax.boxplot(
            boxes, tick_labels=labels, widths=0.6, patch_artist=True,
            whis=(5, 95), showfliers=True,
        )
        colors = ("tab:blue", "tab:orange", "tab:green", "tab:gray")
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.axhline(1e-2, color="black", linestyle=":", lw=0.8,
                   label="1% bar")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(
        f"Per-mode rel-err over a single dt={int(d['dt'])} s, "
        f"nstep={int(d['nstep'])} simulation (60 timesteps per box)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "benchmark_relerr_aerosols.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _gas_relerr_plot(d: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    fields = (
        ("h2so4_gas", d["j_h2so4"], d["f_h2so4"]),
        ("soag_gas", d["j_soag"], d["f_soag"]),
    )
    for ax, (name, j, f) in zip(axes, fields):
        rel = _relerr(j, f)
        bp = ax.boxplot(
            [np.maximum(rel, 1e-20)], tick_labels=[name],
            widths=0.5, patch_artist=True, whis=(5, 95), showfliers=True,
        )
        bp["boxes"][0].set_facecolor("tab:purple")
        bp["boxes"][0].set_alpha(0.6)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_ylabel("rel-err")
        ax.axhline(1e-2, color="black", linestyle=":", lw=0.8,
                   label="1% bar")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"Gas-phase rel-err over a single dt={int(d['dt'])} s, "
        f"nstep={int(d['nstep'])} simulation (60 timesteps per box)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "benchmark_relerr_gas.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    if not CACHE.exists():
        print(f"No cache found at {CACHE}. Run "
              f"scripts/benchmark_1000_sims.py first.")
        return
    d = dict(np.load(CACHE))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    wt = _walltime_plot(d)
    print(f"saved {wt.name}")
    ar = _aerosols_relerr_plot(d)
    print(f"saved {ar.name}")
    gs = _gas_relerr_plot(d)
    print(f"saved {gs.name}")


if __name__ == "__main__":
    main()
