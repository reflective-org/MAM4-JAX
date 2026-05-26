"""Per-mode trajectory plots from the cached 24h JAX validation runs.

Reads scripts/_artifacts/diffrax_24h_dt{dt}.npz produced by
diffrax_24h_validation.py and renders, per dt:
  - 4-column × 2-row figure for each per-mode field
    (num_aer, so4_aer, soa_aer): value top, rel-err bottom.
  - 2-column × 2-row figure for the gas fields
    (h2so4_gas, soag_gas).

Output: docs/figures/traj_<field>_24h_dt{dt}.png and
        docs/figures/traj_gas_24h_dt{dt}.png.

Also emits a single summary figure docs/figures/summary_24h_per_field.png
showing max per-mode rel-err vs dt for every field.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = Path(__file__).resolve().parent / "_artifacts"
FIG = ROOT / "docs" / "figures"

DT_LIST = (1, 5, 30, 300)
MODE_NAMES = ("Aitken", "accumulation", "primary-carbon", "coarse")


def _relerr(j: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.abs(j - f) / np.maximum(np.abs(f), 1e-300)


def _plot_per_mode(name: str, dt: int, j_pm: np.ndarray, f_pm: np.ndarray,
                   t: np.ndarray, ylabel: str) -> None:
    """j_pm, f_pm: shape (4 modes, ntime). `t` is seconds; we plot in
    hours on a shared x-axis across both rows."""
    t_h = t / 3600.0
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    fig.suptitle(
        f"{name} trajectory — 24h, dt={dt}s — Fortran solid / diffrax dashed",
        fontsize=12,
    )
    for m in range(4):
        ax_v, ax_e = axes[0, m], axes[1, m]
        # Skip mode if reference is all zero (e.g., coarse mode SO4/SOA empty).
        if not np.any(f_pm[m]):
            ax_v.text(0.5, 0.5, "field is identically zero",
                      ha="center", va="center", transform=ax_v.transAxes,
                      fontsize=9, color="gray")
            ax_v.set_title(f"mode {m}: {MODE_NAMES[m]}")
            ax_v.set_xticks([])
            ax_v.set_yticks([])
            ax_e.set_xticks([])
            ax_e.set_yticks([])
            continue
        ax_v.plot(t_h, f_pm[m], "-", color="tab:blue", lw=1.2, label="Fortran")
        ax_v.plot(t_h, j_pm[m], "--", color="tab:red", lw=1.1, label="diffrax")
        ax_v.set_yscale("log")
        ax_v.set_title(f"mode {m}: {MODE_NAMES[m]}")
        if m == 0:
            ax_v.set_ylabel(ylabel)
            ax_v.legend(loc="best", fontsize=9)
        ax_v.grid(True, which="both", alpha=0.3)

        rel = _relerr(j_pm[m], f_pm[m])
        ax_e.semilogy(t_h, np.maximum(rel, 1e-20),
                      color="tab:purple", lw=1.0)
        ax_e.axhline(1e-2, color="black", linestyle=":", lw=0.8,
                     label="1% bar")
        if m == 0:
            ax_e.set_ylabel("rel-err")
            ax_e.legend(loc="best", fontsize=9)
        ax_e.set_xlabel("time (h)")
        ax_e.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIG / f"traj_{name}_24h_dt{dt}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def _plot_gas(dt: int, j_h: np.ndarray, f_h: np.ndarray,
              j_s: np.ndarray, f_s: np.ndarray, t: np.ndarray) -> None:
    t_h = t / 3600.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(f"Gas-phase trajectory — 24h, dt={dt}s — "
                 f"Fortran solid / diffrax dashed", fontsize=12)
    for col, (name, j, f, ylabel) in enumerate((
        ("h2so4_gas", j_h, f_h, "kg H2SO4 / kg air"),
        ("soag_gas", j_s, f_s, "kg SOAG / kg air"),
    )):
        ax_v, ax_e = axes[0, col], axes[1, col]
        ax_v.plot(t_h, f, "-", color="tab:blue", lw=1.2, label="Fortran")
        ax_v.plot(t_h, j, "--", color="tab:red", lw=1.1, label="diffrax")
        ax_v.set_yscale("log")
        ax_v.set_title(name)
        ax_v.set_ylabel(ylabel)
        ax_v.grid(True, which="both", alpha=0.3)
        ax_v.legend(loc="best", fontsize=9)

        rel = _relerr(j, f)
        ax_e.semilogy(t_h, np.maximum(rel, 1e-20),
                      color="tab:purple", lw=1.0)
        ax_e.axhline(1e-2, color="black", linestyle=":", lw=0.8,
                     label="1% bar")
        ax_e.set_ylabel("rel-err")
        ax_e.set_xlabel("time (h)")
        ax_e.grid(True, which="both", alpha=0.3)
        ax_e.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = FIG / f"traj_gas_24h_dt{dt}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def _summary_plot(per_dt: dict) -> None:
    """One log-log plot per field; x=dt, y=max rel-err per mode."""
    fields = ("num_aer", "so4_aer", "soa_aer")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    dts = sorted(per_dt.keys())

    for col, fld in enumerate(fields):
        ax = axes[col]
        for m in range(4):
            vals = [per_dt[dt][f"{fld}_mode{m}"] for dt in dts]
            ax.loglog(dts, vals, "o-",
                      label=MODE_NAMES[m], lw=1.2, markersize=5)
        ax.axhline(1e-2, color="black", linestyle=":", lw=0.8)
        ax.set_xlabel("dt (s)")
        ax.set_ylabel("max rel-err" if col == 0 else "")
        ax.set_title(fld)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.invert_xaxis()

    for col, fld in enumerate(("h2so4_gas", "soag_gas")):
        ax = axes[3 + col]
        vals = [per_dt[dt][fld] for dt in dts]
        ax.loglog(dts, vals, "s-", color="tab:purple",
                  lw=1.3, markersize=6)
        ax.axhline(1e-2, color="black", linestyle=":", lw=0.8,
                   label="1% bar")
        ax.set_xlabel("dt (s)")
        ax.set_title(fld)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.invert_xaxis()

    fig.suptitle("24h validation: max per-mode rel-err vs dt", fontsize=12)
    fig.tight_layout()
    out = FIG / "summary_24h_per_field.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def _per_dt_relerrs(d: dict) -> dict:
    out = {}
    for key, j_key, f_key in (
        ("num_aer", "j_num", "f_num"),
        ("so4_aer", "j_so4", "f_so4"),
        ("soa_aer", "j_soa", "f_soa"),
    ):
        for m in range(4):
            r = _relerr(d[j_key][m], d[f_key][m])
            out[f"{key}_mode{m}"] = float(r.max())
    for key, j_key, f_key in (
        ("h2so4_gas", "j_h2so4", "f_h2so4"),
        ("soag_gas", "j_soag", "f_soag"),
    ):
        r = _relerr(d[j_key], d[f_key])
        out[key] = float(r.max())
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    per_dt = {}
    for dt in DT_LIST:
        path = ART / f"diffrax_24h_dt{dt}.npz"
        if not path.exists():
            print(f"  skipping dt={dt}: {path.name} not found")
            continue
        d = dict(np.load(path))
        nstep = int(d["nstep"])
        t = np.arange(1, nstep + 1, dtype=np.float64) * dt
        print(f"Plotting dt={dt}s (nstep={nstep}):")
        _plot_per_mode("num_aer", dt, d["j_num"], d["f_num"], t,
                       "num_aer (#/kg)")
        _plot_per_mode("so4_aer", dt, d["j_so4"], d["f_so4"], t,
                       "so4_aer (kg/kg)")
        _plot_per_mode("soa_aer", dt, d["j_soa"], d["f_soa"], t,
                       "soa_aer (kg/kg)")
        _plot_gas(dt, d["j_h2so4"], d["f_h2so4"],
                  d["j_soag"], d["f_soag"], t)
        per_dt[dt] = _per_dt_relerrs(d)

    if per_dt:
        print("Summary figure:")
        _summary_plot(per_dt)


if __name__ == "__main__":
    main()
