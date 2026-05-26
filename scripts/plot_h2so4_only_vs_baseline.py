"""Plot h2so4_gas rel-err: PR-D2 baseline (soaexch on) vs ablation
experiment (soaexch off, both Fortran and JAX), per dt.

Reads cached .npz files produced by:
  - scripts/diffrax_24h_validation.py     (PR-D2 baseline, soaexch on)
  - scripts/diffrax_24h_h2so4_only.py     (ablation, soaexch off)

Renders one panel per available dt; soaexch-on dashed red,
soaexch-off solid blue. Confirms the hypothesis that the 0.31%
h2so4_gas rel-err under PR-D2 is propagation from soaexch drift,
not anything intrinsic to H2SO4.

Output: docs/figures/h2so4_only_vs_baseline.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = Path(__file__).resolve().parent / "_artifacts"
FIG = ROOT / "docs" / "figures" / "h2so4_only_vs_baseline.png"
DT_LIST = (300, 30, 5, 1)


def _relerr(j: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.abs(j - f) / np.maximum(np.abs(f), 1e-300)


def main() -> None:
    panels = []
    for dt in DT_LIST:
        on_path = ART / f"diffrax_24h_dt{dt}.npz"
        off_path = ART / f"h2so4_only_24h_dt{dt}.npz"
        if not (on_path.exists() and off_path.exists()):
            print(f"  skipping dt={dt}: missing cache")
            continue
        panels.append(dt)
    if not panels:
        print("no cached results found; nothing to plot")
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n + 0.5, 4.5),
                             sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "h2so4_gas rel-err vs Fortran — "
        "soaexch ON (PR-D2 baseline) vs soaexch OFF (ablation)",
        fontsize=12,
    )
    for ax, dt in zip(axes, panels):
        d_on = dict(np.load(ART / f"diffrax_24h_dt{dt}.npz"))
        d_off = dict(np.load(ART / f"h2so4_only_24h_dt{dt}.npz"))
        nstep = int(d_on["nstep"])
        t_h = np.arange(1, nstep + 1) * dt / 3600.0
        rel_on = _relerr(d_on["j_h2so4"], d_on["f_h2so4"])
        rel_off = _relerr(d_off["j_h2so4"], d_off["f_h2so4"])

        ax.semilogy(t_h, np.maximum(rel_on, 1e-20),
                    "--", color="tab:red", lw=1.3,
                    label=f"soaexch ON (peak {rel_on.max():.2e})")
        ax.semilogy(t_h, np.maximum(rel_off, 1e-20),
                    "-", color="tab:blue", lw=1.3,
                    label=f"soaexch OFF (peak {rel_off.max():.2e})")
        ax.axhline(1e-2, color="black", linestyle=":", lw=0.8,
                   label="1% bar")
        ax.set_title(f"dt = {dt} s  (nstep = {nstep})")
        ax.set_xlabel("time (h)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("h2so4_gas rel-err")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
