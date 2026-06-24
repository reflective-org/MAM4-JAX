"""Render the newnuc-helpers JAX-vs-Fortran residual figure.

Loads ``tests/reference/newnuc_helpers/reference.npz`` (output of
``scripts/reference_drivers/newnuc_helpers_driver.F90``), runs the JAX
ports, and writes ``docs/figures/newnuc_helpers_residuals.png``:

    Top:    Vehkamäki binary nucleation rate vs. so4vol for a few
            representative (T, RH) slices, JAX (dashed) overlaying
            Fortran (solid). Log-log scale spans 10–12 orders of magnitude.
    Bottom: per-record |rel-err| for each output (Vehkamäki + Wang
            flagaa=11 + Wang flagaa=12), versus record index in the
            flattened (T, RH, so4vol) grid. ADR-003 1e-6 reference line.

Usage:
    python scripts/plot_newnuc_helpers_residuals.py
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
from mam4_jax.newnuc import binary_nuc_vehk2002, pbl_nuc_wang2008

REF_PATH = REPO_ROOT / "tests" / "reference" / "newnuc_helpers" / "reference.npz"
FIG_PATH = REPO_ROOT / "docs" / "figures" / "newnuc_helpers_residuals.png"

TOLERANCE = 1e-6
EPS = np.finfo(np.float64).eps


def _rel_err(jax_arr, ref_arr) -> np.ndarray:
    j = np.asarray(jax_arr)
    return np.abs(j - ref_arr) / np.maximum(np.abs(ref_arr), 1e-30)


def main() -> int:
    d = {k: np.asarray(v) for k, v in np.load(REF_PATH).items()}

    # Re-run the JAX ports on the captured grid.
    temp = jnp.asarray(d["temp"])
    rh   = jnp.asarray(d["rh"])
    so4  = jnp.asarray(d["so4vol"])

    bin_rate, bin_log, bin_ch, bin_ct, bin_rad = binary_nuc_vehk2002(temp, rh, so4)

    cn_in = jnp.zeros_like(jnp.asarray(d["binary_ratenucl"]))
    pbl11 = pbl_nuc_wang2008(
        so4, 11, jnp.asarray(d["binary_ratenucl"]), jnp.asarray(d["binary_rateloge"]),
        jnp.asarray(d["binary_cnum_tot"]), jnp.asarray(d["binary_cnum_h2so4"]),
        cn_in, jnp.asarray(d["binary_radius"]),
    )
    pbl12 = pbl_nuc_wang2008(
        so4, 12, jnp.asarray(d["binary_ratenucl"]), jnp.asarray(d["binary_rateloge"]),
        jnp.asarray(d["binary_cnum_tot"]), jnp.asarray(d["binary_cnum_h2so4"]),
        cn_in, jnp.asarray(d["binary_radius"]),
    )

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8))

    # ------------------------------------------------------------------
    # Top: binary nucleation rate vs so4vol on a few (T, RH) slices.
    # Pick a sparse set of (T, RH) pairs across the grid.
    # ------------------------------------------------------------------
    n_so4 = 12   # matches the driver's n_so4
    n_rh  = 10
    n_temp = 16

    # Reshape grid: (n_temp, n_rh, n_so4) so we can slice cleanly.
    bin_rate_3d_jax = np.asarray(bin_rate).reshape(n_temp, n_rh, n_so4)
    bin_rate_3d_ref = d["binary_ratenucl"].reshape(n_temp, n_rh, n_so4)
    temp_3d         = d["temp"].reshape(n_temp, n_rh, n_so4)
    so4_3d          = d["so4vol"].reshape(n_temp, n_rh, n_so4)
    rh_3d           = d["rh"].reshape(n_temp, n_rh, n_so4)

    # 3 representative (T, RH) slices spanning the grid.
    slices = [(0,  0), (n_temp // 2, n_rh // 2), (n_temp - 1, n_rh - 1)]
    cmap = plt.get_cmap("viridis")
    for j, (it, ir) in enumerate(slices):
        color = cmap(j / max(len(slices) - 1, 1))
        label_T  = temp_3d[it, ir, 0]
        label_RH = rh_3d[it, ir, 0]
        so4_line = so4_3d[it, ir, :]
        ref_line = bin_rate_3d_ref[it, ir, :]
        jax_line = bin_rate_3d_jax[it, ir, :]
        # Mask near-zero rates so log-scale doesn't choke.
        safe_ref = np.maximum(ref_line, 1e-300)
        safe_jax = np.maximum(jax_line, 1e-300)
        top.loglog(so4_line, safe_ref, color=color, lw=2.0,
                   label=f"Fortran T={label_T:.0f} K, RH={label_RH:.2f}")
        top.loglog(so4_line, safe_jax, color=color, lw=0.9, ls="--",
                   label=f"JAX     T={label_T:.0f} K, RH={label_RH:.2f}")
    top.set_xlabel(r"[H$_2$SO$_4$]  (molec / cm³)")
    top.set_ylabel("Vehkamäki binary nucleation rate (# / cm³ / s)")
    top.set_title("binary_nuc_vehk2002: ratenucl vs [H$_2$SO$_4$] across (T, RH) slices")
    top.grid(True, which="both", alpha=0.3)
    top.legend(fontsize=8, ncol=2, loc="best")

    # ------------------------------------------------------------------
    # Bottom: per-record |rel-err| for each output.
    # ------------------------------------------------------------------
    n_records = d["temp"].size
    record_idx = np.arange(n_records)

    series = [
        ("binary ratenucl",   _rel_err(bin_rate, d["binary_ratenucl"])),
        ("binary rateloge",   _rel_err(bin_log,  d["binary_rateloge"])),
        ("binary cnum_h2so4", _rel_err(bin_ch,   d["binary_cnum_h2so4"])),
        ("binary cnum_tot",   _rel_err(bin_ct,   d["binary_cnum_tot"])),
        ("binary radius",     _rel_err(bin_rad,  d["binary_radius"])),
        ("pbl11 ratenucl",    _rel_err(pbl11[1], d["pbl11_ratenucl"])),
        ("pbl12 ratenucl",    _rel_err(pbl12[1], d["pbl12_ratenucl"])),
    ]

    palette = [cmap(k / max(len(series) - 1, 1)) for k in range(len(series))]
    for k, (label, rel_err) in enumerate(series):
        bot.semilogy(record_idx, np.maximum(rel_err, EPS),
                     color=palette[k], lw=0.7, marker=",",
                     label=label)
    bot.axhline(TOLERANCE, color="red", ls=":", lw=1.5,
                label=f"ADR-003 tol ({TOLERANCE:.0e})")
    bot.axhline(EPS, color="grey", ls=":", lw=1, alpha=0.5,
                label=f"float64 ε ({EPS:.0e})")
    bot.set_xlabel(f"grid record index ({n_records} total)")
    bot.set_ylabel(r"$|{\rm JAX} - {\rm Fortran}| / |{\rm Fortran}|$")
    bot.set_title("Per-record relative error — all binary / PBL outputs")
    bot.grid(True, which="both", alpha=0.3)
    bot.legend(fontsize=7, loc="best", ncol=2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot_newnuc_helpers_residuals] wrote {FIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  worst rel-err across all outputs: "
          f"{max(arr.max() for _, arr in series):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
