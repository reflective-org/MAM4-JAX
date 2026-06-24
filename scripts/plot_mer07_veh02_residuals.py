"""Render the mer07_veh02 dispatcher JAX-vs-Fortran residual figure.

Loads ``tests/reference/mer07_veh02/reference.npz`` (output of
``scripts/reference_drivers/mer07_veh02_driver.F90``), runs the JAX
port, and writes ``docs/figures/mer07_veh02_residuals.png``:

    Top:    Nucleation cluster-rate (``dnclusterdt``) vs [H₂SO₄] for a
            few representative (T, z, uptkrate) slices, JAX (dashed)
            overlaying Fortran (solid). Log-log axes span 10+ orders
            of magnitude.
    Bottom: Per-record |rel-err| for the four physics outputs
            (``qnuma_del``, ``qso4a_del``, ``qh2so4_del``,
            ``dnclusterdt``) vs ADR-003's 1e-6 + float64 ε reference.

Usage:
    python scripts/plot_mer07_veh02_residuals.py
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
from mam4_jax.newnuc import mer07_veh02_nuc_mosaic_1box

REF_PATH = REPO_ROOT / "tests" / "reference" / "mer07_veh02" / "reference.npz"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "mer07_veh02_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


def _rel_err(jax_arr, ref_arr) -> np.ndarray:
    return np.abs(np.asarray(jax_arr) - ref_arr) / np.maximum(np.abs(ref_arr), 1e-30)


def main() -> int:
    d = {k: np.asarray(v) for k, v in np.load(REF_PATH).items()}

    out = mer07_veh02_nuc_mosaic_1box(
        dtnuc=30.0,
        temp=jnp.asarray(d["temp"]), rh=jnp.asarray(d["rh"]),
        press=1.0e5, zm=jnp.asarray(d["zm"]), pblh=1000.0,
        qh2so4_cur=jnp.asarray(d["qh2so4"]),
        qh2so4_avg=jnp.asarray(d["qh2so4"]),
        h2so4_uptkrate=jnp.asarray(d["uptkrate"]),
        dplom_sect=0.0087e-6, dphim_sect=0.0520e-6,
        newnuc_method_flagaa=11,
    )
    (_isize, qnuma_jax, qso4_jax, _qnh4, qh2so4_jax,
     _qnh3, _dens, dncl_jax) = out

    n_records = d["temp"].size
    record_idx = np.arange(n_records)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8))

    # ------------------------------------------------------------------
    # Top: dnclusterdt vs qh2so4 across a few representative slices.
    # The fixture is a 5D sweep flattened to 1D in this order:
    #   (n_temp, n_rh, n_zm, n_so4, n_uptk) = (6, 5, 3, 8, 3)
    # Pick three (T, zm, uptk) slices at fixed RH=middle.
    # ------------------------------------------------------------------
    n_temp, n_rh, n_zm, n_so4, n_uptk = 6, 5, 3, 8, 3
    expected = n_temp * n_rh * n_zm * n_so4 * n_uptk
    assert n_records == expected, f"sweep shape changed: {n_records} != {expected}"

    dncl_3d_jax = np.asarray(dncl_jax).reshape(n_temp, n_rh, n_zm, n_so4, n_uptk)
    dncl_3d_ref = d["dnclusterdt"].reshape(n_temp, n_rh, n_zm, n_so4, n_uptk)
    qh2so4_3d   = d["qh2so4"].reshape(n_temp, n_rh, n_zm, n_so4, n_uptk)
    temp_3d     = d["temp"].reshape(n_temp, n_rh, n_zm, n_so4, n_uptk)
    zm_3d       = d["zm"].reshape(n_temp, n_rh, n_zm, n_so4, n_uptk)
    uptk_3d     = d["uptkrate"].reshape(n_temp, n_rh, n_zm, n_so4, n_uptk)

    ir = n_rh // 2   # fix RH at midpoint
    iu = n_uptk // 2 # fix uptkrate at midpoint
    slices = [(0, 0), (n_temp // 2, 1), (n_temp - 1, 2)]   # (it, iz)
    cmap = plt.get_cmap("viridis")
    for j, (it, iz) in enumerate(slices):
        color = cmap(j / max(len(slices) - 1, 1))
        label_T = temp_3d[it, ir, iz, 0, iu]
        label_z = zm_3d[it, ir, iz, 0, iu]
        x   = qh2so4_3d[it, ir, iz, :, iu]
        ref = dncl_3d_ref[it, ir, iz, :, iu]
        jxr = dncl_3d_jax[it, ir, iz, :, iu]
        # Mask zeros so log-scale doesn't choke.
        safe_ref = np.maximum(ref, 1e-300)
        safe_jxr = np.maximum(jxr, 1e-300)
        top.loglog(x, safe_ref, color=color, lw=2.0,
                   label=f"Fortran T={label_T:.0f} K, z={label_z:.0f} m")
        top.loglog(x, safe_jxr, color=color, lw=0.9, ls="--",
                   label=f"JAX     T={label_T:.0f} K, z={label_z:.0f} m")
    top.set_xlabel(r"$q_{H_2SO_4}$  (mol/mol-air)")
    top.set_ylabel("dnclusterdt  (# / m³ / s)")
    top.set_title("mer07_veh02 dispatcher: cluster nucleation rate vs $q_{H_2SO_4}$")
    top.grid(True, which="both", alpha=0.3)
    top.legend(fontsize=8, ncol=2, loc="best")

    # ------------------------------------------------------------------
    # Bottom: per-record rel-err for all four physics outputs.
    # ------------------------------------------------------------------
    series = [
        ("qnuma_del",   _rel_err(qnuma_jax,  d["qnuma_del"])),
        ("qso4a_del",   _rel_err(qso4_jax,   d["qso4a_del"])),
        ("qh2so4_del",  _rel_err(qh2so4_jax, d["qh2so4_del"])),
        ("dnclusterdt", _rel_err(dncl_jax,   d["dnclusterdt"])),
    ]

    palette = [cmap(k / max(len(series) - 1, 1)) for k in range(len(series))]
    for k, (label, rel) in enumerate(series):
        bot.semilogy(record_idx, np.maximum(rel, EPS),
                     color=palette[k], lw=0.7, marker=",",
                     label=label)
    bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                label=f"ADR-003 tol ({TOLERANCE:.0e})")
    bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                label=f"float64 ε ({EPS:.0e})")
    bot.set_xlabel(f"grid record index ({n_records} total)")
    bot.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    bot.set_title("Per-record relative error across all sweep regimes")
    bot.grid(True, which="both", alpha=0.3)
    bot.legend(fontsize=8, loc="best", ncol=2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_mer07_veh02_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  worst rel-err: {max(rel.max() for _, rel in series):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
