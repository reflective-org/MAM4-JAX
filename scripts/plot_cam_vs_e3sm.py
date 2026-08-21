#!/usr/bin/env python3
"""CAM vs E3SM: where the two MAM code lines actually differ, and by how much.

Three panels, each answering a question that came up porting CAM's kernels:

  1. Rename: how far apart are the two algorithms, and where?
  2. Rename: is the CAM branch right? (against CAM's own Fortran)
  3. Condensation: is the tropospheric H2SO4 path the same maths?

Data from /tmp/camfig/data.json (see the generator in the commit that added
this). Writes docs/figures/cam_vs_e3sm.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.loads(Path("/tmp/camfig/data.json").read_text())
OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "cam_vs_e3sm.png"

# Validated categorical palette (dataviz reference instance, light mode):
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light
#   -> ALL CHECKS PASS
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e3e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titlepad": 8,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 8,
})

fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.4))
fig.suptitle("CAM vs E3SM — where the two MAM code lines diverge",
             fontsize=13, x=0.008, ha="left", y=0.975)
fig.text(0.008, 0.925,
         "JAX port validated against CAM cam6_4_187 via mam-box-fortran · "
         "aitken→accum rename, and tropospheric H2SO4 condensation",
         fontsize=8.5, color=INK2, ha="left")

# ---- 1. rename: transfer fraction vs growth --------------------------------
p = D["p1"]; g = np.asarray(p["growths"])
a = ax[0]
a.plot(g, p["cam_1.2e-7"], color=BLUE, label="CAM")
a.plot(g, p["e3sm_1.2e-7"], color=ORANGE, label="E3SM")
a.plot(g, p["cam_9e-8"], color=BLUE, ls=(0, (5, 3)), lw=1.6)
a.plot(g, p["e3sm_9e-8"], color=ORANGE, ls=(0, (5, 3)), lw=1.6)
a.set_xscale("log"); a.set_yscale("log")
a.text(0.03, 0.96, "solid: dgn_old = 1.2e-7 m\ndashed: dgn_old = 9e-8 m",
       transform=a.transAxes, fontsize=8, color=INK2, va="top")
a.annotate("E3SM skips CAM's up-front growth gate,\nso it transfers where CAM does nothing",
           xy=(g[1], p["e3sm_1.2e-7"][1]), xytext=(0.05, 0.16),
           textcoords="axes fraction", fontsize=8, color=ORANGE, weight="bold",
           arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
a.annotate("converge once growth pushes\ndgn_new past dp_cut",
           xy=(g[-3], p["cam_1.2e-7"][-3]), xytext=(0.42, 0.60),
           textcoords="axes fraction", fontsize=8, color=INK2,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.set_title("1 · Rename: up to 4.3× apart, and only in a window", loc="left")
a.set_xlabel("fractional growth of the aitken mode over the step")
a.set_ylabel("fraction of aitken number transferred")
a.legend(loc="lower right")

# ---- 2. rename: CAM branch vs CAM's Fortran --------------------------------
p2 = D["p2"]; dgs = np.asarray(p2["dgns"]) * 1e9
a = ax[1]
a.plot(dgs, p2["cam_0.01"], color=BLUE, label="CAM (JAX port)")
a.plot(dgs, p2["e3sm_0.01"], color=ORANGE, label="E3SM (JAX port)")
ref = D["ref"]
rx = [r["dgn_old"] * 1e9 for r in ref if r["growth"] == 0.01]
ry = [r["xferfrac_num"] for r in ref if r["growth"] == 0.01]
a.plot(rx, ry, "o", ms=11, mfc="none", mec=INK, mew=2.0, ls="none",
       label="CAM Fortran (reference)")
a.axvline(p2["dp_cut"] * 1e9, color=INK2, lw=0.9, ls=(0, (4, 3)))
a.text(0.50, 0.40, f"dp_cut = {p2['dp_cut']*1e9:.1f} nm",
       transform=a.transAxes, fontsize=8, color=INK2)
a.annotate("the port sits on the reference\nto 8.1e-15 — machine precision",
           xy=(rx[0], ry[0]), xytext=(0.42, 0.12), textcoords="axes fraction",
           fontsize=8, color=INK, weight="bold",
           arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
a.set_xscale("log"); a.set_yscale("log")
a.set_title("2 · The CAM branch is right, the E3SM one is not", loc="left")
a.set_xlabel("aitken dgn before growth (nm), growth = 1 %")
a.set_ylabel("fraction of aitken number transferred")
a.legend(loc="upper left")

# ---- 3. condensation equivalence -------------------------------------------
errs = np.asarray(D["p3"]["errs"])
a = ax[2]
a.hist(errs, bins=np.logspace(-17, -12, 34), color=AQUA, edgecolor=SURFACE, lw=0.6)
a.set_xscale("log")
a.axvline(np.finfo(np.float64).eps, color=INK2, lw=0.9, ls=(0, (4, 3)))
a.text(0.30, 0.96, "1 machine ε", transform=a.transAxes,
       fontsize=8, color=INK2, va="top")
a.text(0.40, 0.82,
       f"400 random cases, worst {errs.max():.2e}\n\n"
       "CAM's fgain + avg_uprt formulation and\nthe JAX closed form are the same\n"
       "competing-sink ODE — not an\napproximation, identical.",
       transform=a.transAxes, fontsize=8, color=INK2, va="top")
a.set_title("3 · Condensation: the same maths, to round-off", loc="left")
a.set_xlabel("relative difference, CAM formula vs JAX closed form")
a.set_ylabel("cases")

fig.tight_layout(rect=(0, 0.005, 1, 0.90))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170)
print(f"wrote {OUT}")
print(f"  condensation worst err : {errs.max():.3e}")
print(f"  rename ref points      : {len(ref)}")
