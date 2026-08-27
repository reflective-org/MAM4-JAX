"""Stratospheric RH scenario studies on the CAM driver.

Regenerates ``docs/figures/cam_strat_rh_*.png``. Two studies, both
``cam_mam5``, ``strat=True``, T=232 K, p=50 hPa, dt=30 s, and the ADR-021
default ``n_substeps=16``:

* **6-h sweep, strong forcing** (SO2 = 1e-7 vmr = 0.1 ppmv, volcanic-plume
  scale): the classic nucleation–growth banana. Nearly RH-insensitive in
  number and mass because the system is PRODUCTION-limited — every H2SO4
  molecule produced (1.6e6 cm^-3 s^-1) is consumed at any RH.
* **12-h sweep × three experiments** at 1e7 molec/cm3 initial H2SO4,
  isolating the RH → water uptake → wet diameter → condensation-sink
  chain: (1) *burst* (no SO2) — RH-ordered gas-depletion floors spanning
  3.5 decades; (2) *background* (SO2 = 1e-10 vmr) — a ~12% number spread
  with MORE particles at LOW RH (the weaker sink leaves gas standing for
  nucleation); (3) *condensation-only* (nucleation off) — the wet aitken
  diameter fans 31.5 → 36.2 nm at identical dry size, and the steady gas
  sits 35% lower at RH 50%.

Exploratory scenario study, not validation — no Fortran reference is
attached. Figure conventions: per-mode categorical colors in fixed order
across every figure; ordered dimensions (time, RH) use sequential ramps.

Usage:  python scripts/cam_rh_scenarios.py [outdir]   (default docs/figures)
Runtime: ~2 min on CPU (compile-dominated; the runs themselves are ~0.2 s
each on the jitted driver).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import jax.numpy as jnp  # noqa: E402

import mam4_jax  # noqa: F401,E402  (enables x64)
import mam4_jax.coupling.cam_driver as cd  # noqa: E402
from mam4_jax.core.cam_params import CAM_PARAMS  # noqa: E402
from mam4_jax.core.cam_topologies import CAM_MAM5  # noqa: E402
from mam4_jax.physics.cam_saturation import qsat_cam  # noqa: E402

TOPO = CAM_MAM5
T, P, DT = 232.0, 5.0e3, 30.0
N_AIR = P / (1.380649e-23 * T) * 1e-6          # molec/cm3
RHO = P / (287.0423 * T)                       # kg/m3
RHS = [0.05, 0.10, 0.20, 0.50]
RH_TAGS = [f"rh{int(r * 100):02d}" for r in RHS]
RH_LBL = {t: f"RH {int(t[2:])}%" for t in RH_TAGS}
MODES = ["accum", "aitken", "coarse", "prim. carbon", "coarse_strat"]
MODE_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order
RH_C = ["#b8d3f2", "#7fb0e6", "#4a90dc", "#1c5cab"]               # sequential
GREYS = ["#c9c9c9", "#a0a0a0", "#747474", "#424242", "#000000"]
NUMC = (1.0e8, 1.0e9, 1.0e5, 1.0e0)            # #/m3 per mode 1..4 (box namelist)
NUMC5 = 1.0e4
SO4FRAC = (1.0, 1.0, 1.0, 0.0)
GRID = dict(color="#e3e3e3", lw=0.6)

_tb = cd._cam_tables(TOPO)
_names = CAM_PARAMS[TOPO.name]["cnst_names"]
_iH = _names.index("H2SO4")
SIG = np.asarray(TOPO.sigmag_amode)
LNS = np.log(SIG)
DP = np.logspace(np.log10(1e-9), np.log10(3e-6), 240)
LNDP = np.log(DP)


def build_ic(rh, qso2, qh2so4):
    """The box driver's IC construction (mam_box_driver_cam.F90:279-345)."""
    p = CAM_PARAMS[TOPO.name]
    q = np.zeros(p["gas_pcnst"])
    for m in range(TOPO.nmodes):
        numkg = (NUMC[m] if m < 4 else NUMC5) / RHO
        q[_tb.num_ptr[m]] = numkg
        dg, sg = TOPO.dgnum_amode[m], TOPO.sigmag_amode[m]
        tmpvol = numkg * (np.pi / 6) * dg ** 3 * np.exp(4.5 * np.log(sg) ** 2)
        if _tb.lptr_so4[m] >= 0:
            frac = 1.0 if m == 4 else SO4FRAC[m]
            q[_tb.lptr_so4[m]] = frac * tmpvol * 1770.0
    q[_iH] = qh2so4 * p["adv_mass"][_iH] / p["mwdry"]
    q[_names.index("SO2")] = qso2 * p["adv_mass"][_names.index("SO2")] / p["mwdry"]
    _es, qs = qsat_cam(jnp.asarray(T), jnp.asarray(P))
    nm = TOPO.nmodes
    return {"q": jnp.asarray(q), "qv": jnp.asarray(rh * float(qs)),
            "t": jnp.asarray(T), "pmid": jnp.asarray(P),
            "zm": jnp.asarray(500.0), "pblh": jnp.asarray(1000.0),
            "dgncur_a": jnp.zeros(nm), "dgncur_awet": jnp.zeros(nm),
            "wetdens": jnp.zeros(nm), "deltat": jnp.asarray(DT)}


def run(rh, nstep, qso2, qh2so4, do_newnuc=True):
    _, traj = cd.cam_run_timesteps(build_ic(rh, qso2, qh2so4), nstep,
                                   topology=TOPO, strat=True, n_substeps=16,
                                   do_newnuc=do_newnuc)
    q = np.asarray(traj["q"])
    return dict(
        num=q[:, _tb.num_ptr],
        so4=np.stack([q[:, _tb.lptr_so4[m]] if _tb.lptr_so4[m] >= 0
                      else np.zeros(nstep) for m in range(TOPO.nmodes)], 1),
        dgn=np.asarray(traj["dgncur_a"]),
        dgnwet=np.asarray(traj["dgncur_awet"]),
        h2so4=q[:, _iH] * _tb.mwdry / _tb.adv_mass[_iH] * N_AIR,  # molec/cm3
    )


def init_dist_inputs():
    num0 = np.asarray([(NUMC[m] if m < 4 else NUMC5) / RHO
                       for m in range(TOPO.nmodes)])
    return num0, np.asarray(TOPO.dgnum_amode)


def dist(num, dgn):
    """dN/dlnD (#/mg-air) on the DP grid; also per-mode components."""
    tot = np.zeros_like(DP)
    pm = []
    for m in range(len(num)):
        c = (num[m] * 1e-6) / (np.sqrt(2 * np.pi) * LNS[m]) * np.exp(
            -((LNDP - np.log(dgn[m])) ** 2) / (2 * LNS[m] ** 2))
        pm.append(c)
        tot += c
    return tot, pm


def _style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
                         "axes.edgecolor": "#999999", "axes.linewidth": 0.8})


def banana_panel(ax, run_d, t_hr, first):
    num, dgn = run_d["num"], run_d["dgn"]
    Z = np.zeros((len(DP), len(t_hr)))
    for it in range(len(t_hr)):
        Z[:, it], _ = dist(num[it], dgn[it])
    pm = ax.pcolormesh(t_hr, DP * 1e9, np.maximum(Z, 1e-2),
                       norm=LogNorm(vmin=1e0, vmax=None),
                       cmap="Blues", rasterized=True, shading="auto")
    for m in range(TOPO.nmodes):
        if num[:, m].max() * 1e-6 > 1e-3:
            ax.plot(t_hr, dgn[:, m] * 1e9, color=MODE_C[m], lw=1.4,
                    label=MODES[m] if first else None)
    ax.set_yscale("log")
    ax.set_ylim(1, 3000)
    return pm


def fig_sweep_6h(out: Path):
    """Strong-forcing 6-h sweep: banana + snapshots + budgets."""
    nstep = 720
    t_hr = (np.arange(nstep) + 1) * DT / 3600.0
    runs = {tag: run(rh, nstep, qso2=1.0e-7, qh2so4=1.0e-11)
            for tag, rh in zip(RH_TAGS, RHS)}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True, sharey=True)
    for i, (ax, tag) in enumerate(zip(axes.ravel(), RH_TAGS)):
        pm = banana_panel(ax, runs[tag], t_hr, first=(i == 0))
        pm.set_norm(LogNorm(vmin=1e2, vmax=5e6))
        ax.set_title(RH_LBL[tag], loc="left")
    for ax in axes[1]:
        ax.set_xlabel("time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("dry diameter $D_p$ (nm)")
    fig.colorbar(pm, ax=axes, label=r"dN/dln$D_p$  (# mg$^{-1}$)", pad=0.02)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.45, -0.035),
               title="mode $d_{gn}$ overlays")
    fig.suptitle("Stratospheric nucleation–growth (banana) — CAM driver, cam_mam5, "
                 "T=232 K, p=50 hPa, SO$_2$=1e-7 vmr", y=0.995, fontsize=11)
    fig.savefig(out / "cam_strat_rh_banana_6h.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    num0, dgn0 = init_dist_inputs()
    snap_h = [0.0, 0.5, 1.0, 3.0, 6.0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey=True)
    for i, (ax, tag) in enumerate(zip(axes.ravel(), RH_TAGS)):
        num, dgn = runs[tag]["num"], runs[tag]["dgn"]
        for h, c in zip(snap_h, GREYS):
            if h == 0.0:
                tot, _ = dist(num0, dgn0)
                lbl = "initial"
            else:
                it = int(h * 3600 / DT) - 1
                tot, _ = dist(num[it], dgn[it])
                lbl = f"{h:g} h"
            ax.plot(DP * 1e9, tot, color=c, lw=1.7, label=lbl)
        _, pm6 = dist(num[-1], dgn[-1])
        for m in range(TOPO.nmodes):
            if num[-1, m] > 1e-3:
                ax.plot(DP * 1e9, pm6[m], color=MODE_C[m], lw=1.1, ls="--",
                        label=MODES[m] if i == 0 else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, 3000); ax.set_ylim(1e-2, 3e7)
        ax.grid(True, which="major", **GRID)
        ax.set_title(RH_LBL[tag], loc="left")
    for ax in axes[1]:
        ax.set_xlabel("dry diameter $D_p$ (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"dN/dln$D_p$  (# mg$^{-1}$)")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(h_, l_, loc="upper right", fontsize=7.6, ncol=2,
                      framealpha=0.95)
    fig.suptitle("Number-distribution snapshots (grey = time; dashed = per-mode at 6 h)",
                 y=0.995, fontsize=11)
    fig.savefig(out / "cam_strat_rh_snapshots_6h.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axN, axM, ax5, ax50 = axes.ravel()
    for tag, c in zip(RH_TAGS, RH_C):
        axN.plot(t_hr, runs[tag]["num"].sum(1) * 1e-6, color=c, lw=1.8,
                 label=RH_LBL[tag])
        axM.plot(t_hr, runs[tag]["so4"].sum(1) * 1e9, color=c, lw=1.8)
    axN.set_yscale("log"); axN.set_ylabel(r"total number (# mg$^{-1}$)")
    axN.set_title("(a) total number concentration", loc="left")
    axM.set_ylabel(r"total SO$_4$ aerosol mass (µg kg$^{-1}$)")
    axM.set_title("(b) total sulfate aerosol mass", loc="left")
    axN.legend(fontsize=8, framealpha=0.95)
    for ax, tag in ((ax5, "rh05"), (ax50, "rh50")):
        for m in range(TOPO.nmodes):
            if runs[tag]["num"][:, m].max() * 1e-6 > 1e-3:
                ax.plot(t_hr, runs[tag]["num"][:, m] * 1e-6, color=MODE_C[m],
                        lw=1.5, label=MODES[m])
        ax.set_yscale("log"); ax.set_ylabel(r"mode number (# mg$^{-1}$)")
    ax5.set_title("(c) per-mode number — RH 5%", loc="left")
    ax50.set_title("(d) per-mode number — RH 50%", loc="left")
    ax5.legend(fontsize=8, framealpha=0.95)
    for ax in axes.ravel():
        ax.grid(True, **GRID); ax.set_xlabel("time (h)")
    fig.suptitle("Number and mass budgets — stratospheric RH sweep", y=0.998,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "cam_strat_rh_budgets_6h.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_experiments_12h(out: Path):
    """Weak-forcing 12-h experiments: burst / background / condensation-only."""
    nstep = 1440
    t_hr = (np.arange(nstep) + 1) * DT / 3600.0
    vmr_1e7 = 1.0e7 / N_AIR
    exps = {"burst": (0.0, vmr_1e7, True),
            "background": (1.0e-10, vmr_1e7, True),
            "condonly": (1.0e-10, vmr_1e7, False)}
    exp_lbl = {"burst": "1) burst: H$_2$SO$_4$=1e7 cm$^{-3}$, no SO$_2$",
               "background": "2) background: SO$_2$=1e-10 vmr",
               "condonly": "3) condensation-only, SO$_2$=1e-10"}
    runs = {(e, tag): run(rh, nstep, qso2=s, qh2so4=h, do_newnuc=nuc)
            for e, (s, h, nuc) in exps.items()
            for tag, rh in zip(RH_TAGS, RHS)}

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True, sharey=True)
    for r, e in enumerate(exps):
        for c, tag in enumerate(RH_TAGS):
            pm = banana_panel(axes[r, c], runs[(e, tag)], t_hr,
                              first=(r, c) == (0, 0))
            pm.set_norm(LogNorm(vmin=1e0, vmax=1e5))
            axes[r, c].set_xscale("log")
            axes[r, c].set_xlim(t_hr[0], 12.0)
            if r == 0:
                axes[r, c].set_title(RH_LBL[tag], loc="left")
            if c == 0:
                axes[r, c].set_ylabel(exp_lbl[e] + "\ndry $D_p$ (nm)", fontsize=8.5)
            if r == 2:
                axes[r, c].set_xlabel("time (h, log)")
    fig.colorbar(pm, ax=axes, label=r"dN/dln$D_p$  (# mg$^{-1}$)", pad=0.015)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.45, -0.03),
               title="mode $d_{gn}$ overlays")
    fig.suptitle("Banana plots (log time) — stratospheric RH sweep, 12 h", y=0.995)
    fig.savefig(out / "cam_strat_rh_banana_12h.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    num0, dgn0 = init_dist_inputs()
    snap_h = [0.0, 1.0, 3.0, 6.0, 12.0]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True, sharey=True)
    for r, e in enumerate(exps):
        for c, tag in enumerate(RH_TAGS):
            ax = axes[r, c]
            rd = runs[(e, tag)]
            for h, col in zip(snap_h, GREYS):
                if h == 0.0:
                    tot, _ = dist(num0, dgn0)
                    lbl = "initial"
                else:
                    it = int(h * 3600 / DT) - 1
                    tot, _ = dist(rd["num"][it], rd["dgn"][it])
                    lbl = f"{h:g} h"
                ax.plot(DP * 1e9, tot, color=col, lw=1.5,
                        label=lbl if (r, c) == (0, 0) else None)
            totw, _ = dist(rd["num"][-1], rd["dgnwet"][-1])
            ax.plot(DP * 1e9, totw, color="#1c5cab", lw=1.5, ls="--",
                    label="12 h, WET $D_p$" if (r, c) == (0, 0) else None)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(1, 3000); ax.set_ylim(1e-2, 3e5)
            ax.grid(True, which="major", **GRID)
            if r == 0:
                ax.set_title(RH_LBL[tag], loc="left")
            if c == 0:
                ax.set_ylabel(exp_lbl[e] + "\n" + r"dN/dln$D_p$ (# mg$^{-1}$)",
                              fontsize=8.5)
            if r == 2:
                ax.set_xlabel("$D_p$ (nm)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.025))
    fig.suptitle("Snapshots (solid grey = dry, by time; dashed blue = wet at 12 h)",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out / "cam_strat_rh_snapshots_12h.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.0))
    for r, e in enumerate(exps):
        axN, axM, axG, axD = axes[r]
        for tag, col in zip(RH_TAGS, RH_C):
            rd = runs[(e, tag)]
            axN.plot(t_hr, rd["num"].sum(1) * 1e-6, color=col, lw=1.7,
                     label=RH_LBL[tag])
            axM.plot(t_hr, rd["so4"].sum(1) * 1e9, color=col, lw=1.7)
            axG.plot(t_hr, rd["h2so4"], color=col, lw=1.7)
            axD.plot(t_hr, rd["dgnwet"][:, 1] * 1e9, color=col, lw=1.7)
        axD.plot(t_hr, runs[(e, "rh50")]["dgn"][:, 1] * 1e9, color="#888888",
                 lw=1.3, ls=":", label="dry (RH 50%)" if r == 0 else None)
        axN.set_yscale("log"); axG.set_yscale("log")
        axN.set_ylabel(exp_lbl[e] + "\n" + r"total N (# mg$^{-1}$)", fontsize=8.5)
        axM.set_ylabel(r"total SO$_4$ (µg kg$^{-1}$)")
        axG.set_ylabel(r"H$_2$SO$_4$(g) (molec cm$^{-3}$)")
        axD.set_ylabel("aitken $D_{gn}$ (nm)")
        for ax in axes[r]:
            ax.grid(True, **GRID)
            if r == 2:
                ax.set_xlabel("time (h)")
    axes[0, 0].set_title("(a) total number", loc="left")
    axes[0, 1].set_title("(b) total sulfate aerosol mass", loc="left")
    axes[0, 2].set_title("(c) H$_2$SO$_4$ gas", loc="left")
    axes[0, 3].set_title("(d) aitken wet diameter (dotted grey = dry)", loc="left")
    axes[0, 0].legend(fontsize=8, framealpha=0.95)
    axes[0, 3].legend(fontsize=8, framealpha=0.95)
    fig.suptitle("Budgets — stratospheric RH sweep, 12 h", y=0.998)
    fig.tight_layout()
    fig.savefig(out / "cam_strat_rh_budgets_12h.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "docs" / "figures")
    out.mkdir(parents=True, exist_ok=True)
    _style()
    fig_sweep_6h(out)
    print("6-h sweep figures written")
    fig_experiments_12h(out)
    print("12-h experiment figures written")
