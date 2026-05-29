"""Cloud chemistry — parameterized aqueous SO2 → SO4 (M8 PR-K2).

Mirrors Fortran ``box_model_utils/cloudchem_simple.F90``'s subroutine
``cloudchem_simple_sub``: a simple e-folding parameterization that
transfers gas-phase SO2 (and all in-cloud H2SO4) into cloud-borne
sulfate aerosol over the cloud-water residence time
(τ = ``TAU_CLOUDCHEM_SIMPLE`` = 1800 s).

**Operates on vmr space.** Inputs ``vmr`` and ``vmrcw`` are volume
mixing ratios with the amicphys-internal ``gas_pcnst`` third-dim
(30 for MAM4-MOM), not mass mixing ratios with ``pcnst=35``. The
driver wrapper does the mmr ↔ vmr conversion around the call
(landing in PR-K3); per-process tests (this PR) pass vmr directly
from the captured fixture.

**Per-gridcell behavior** (per Fortran ``cloudchem_simple.F90:80-131``):

.. code-block:: text

   if (cldn <= 0.009)  CYCLE
   tmpf = min(1.0, cldn)
   tmpd = max(vmrcw[NUM_C1], 1.0)   # accum cloud-borne number
   tmpe = max(vmrcw[NUM_C2], 0.0)   # aitken cloud-borne number
   tmpd_frac = tmpd / (tmpd + tmpe)
   tmpe_frac = max(0.0, 1.0 - tmpd_frac)

   tmpa = tmpf * vmr[SO2]   * exp(-deltat / τ)   # SO2 transfer
   tmpb = tmpf * vmr[H2SO4]                       # H2SO4 transfer
   vmr[SO2]      -= tmpa
   vmr[H2SO4]    -= tmpb
   vmrcw[SO4_C1] += tmpd_frac * (tmpa + tmpb)    # accum
   vmrcw[SO4_C2] += tmpe_frac * (tmpa + tmpb)    # aitken

**NH3 branch absent.** Fortran's optional NH3 → NH4 branch fires only
if ``l_nh3g > 0``. For MAM4-MOM ``cnst_get_ind('NH3', ...)`` returns
``-1`` (NH3 not in the constituent registry) — the branch is
structurally dead. Omitted from the JAX port per CLAUDE.md rule on
"don't add error handling for impossible scenarios."

**Coarse mode untouched.** Cloudchem only deposits into accum and
aitken (mode 0 and mode 1). Coarse-mode cloud-borne sulfate
(``vmrcw[SO4_C3]``) is read by other processes but stays at its
input value through this step. Visible as a flat baseline in the
per-mode SO4_cw panel of ``docs/figures/cloudchem_residuals.png``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import data


#: SO2 → SO4 e-folding timescale in cloud water (s). Verbatim from
#: ``cloudchem_simple.F90:106`` (``tau_cloudchem_simple = 1800.0_r8``).
TAU_CLOUDCHEM_SIMPLE: float = 1800.0

#: Internal cycle threshold from ``cloudchem_simple.F90:108``. Per-
#: gridcell tendencies are zero where ``cldn <= CLDN_CYCLE_THRESHOLD``.
CLDN_CYCLE_THRESHOLD: float = 0.009


@jax.jit
def cloudchem_simple_sub(
    vmr: jnp.ndarray,
    vmrcw: jnp.ndarray,
    cldn: jnp.ndarray,
    deltat: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One application of the parameterized cloud-chem step.

    Parameters
    ----------
    vmr
        Volume mixing ratios, shape ``(ncol, pver, gas_pcnst)``.
    vmrcw
        Cloud-borne volume mixing ratios, same shape.
    cldn
        Stratiform cloud fraction, shape ``(ncol, pver)``, in ``[0, 1]``.
    deltat
        Timestep in seconds.

    Returns
    -------
    (vmr_new, vmrcw_new)
        Updated volume mixing ratios. Shapes match the inputs.
    """
    # Per-gridcell cloud-fraction weight + cycle mask. tmpf clips cldn
    # to [0, 1] (Fortran takes min(1.0, cldn); the cldn=0.5 fixture
    # passes through unchanged).
    tmpf = jnp.minimum(1.0, cldn)                          # (ncol, pver)
    fired = cldn > CLDN_CYCLE_THRESHOLD                    # (ncol, pver) bool

    # Cloud-borne number distribution: accum (mode 0) vs aitken (mode 1).
    # Fortran applies max(..., 1.0) to the accum number to avoid divide-by-
    # zero; we mirror that here. Both modes are guaranteed present in
    # MAM4-MOM (VMRCW_NUM = (12, 17, 25, 29) — all valid slots).
    num_accum  = jnp.maximum(vmrcw[..., data.VMRCW_NUM[data.ACCUM_MODE_IDX]], 1.0)
    num_aitken = jnp.maximum(vmrcw[..., data.VMRCW_NUM[data.AITKEN_MODE_IDX]], 0.0)
    tmpd_frac = num_accum / (num_accum + num_aitken)
    tmpe_frac = jnp.maximum(0.0, 1.0 - tmpd_frac)

    # Gas-side transfer amounts (would-be tendencies before cycle masking).
    so2_in   = vmr[..., data.VMR_SO2]
    h2so4_in = vmr[..., data.VMR_H2SO4]
    tmpa_raw = tmpf * so2_in   * jnp.exp(-deltat / TAU_CLOUDCHEM_SIMPLE)
    tmpb_raw = tmpf * h2so4_in

    # Apply cycle mask: zero out tendencies where cloud fraction is
    # sub-threshold. Fortran's `cycle` skips the body entirely; we
    # replace with `jnp.where`-masked zeros, which is JIT/vmap-clean.
    tmpa = jnp.where(fired, tmpa_raw, 0.0)
    tmpb = jnp.where(fired, tmpb_raw, 0.0)
    transfer = tmpa + tmpb

    # Apply tendencies. The ACCUM and AITKEN cloud-borne sulfate slots are
    # always valid in MAM4-MOM (VMRCW_SO4[0] = 5, VMRCW_SO4[1] = 13);
    # coarse (VMRCW_SO4[2] = 20) and pcarbon (VMRCW_SO4[3] = -1) are not
    # written by cloudchem_simple_sub.
    new_vmr = vmr.at[..., data.VMR_SO2].add(-tmpa)
    new_vmr = new_vmr.at[..., data.VMR_H2SO4].add(-tmpb)
    new_vmrcw = vmrcw.at[..., data.VMRCW_SO4[data.ACCUM_MODE_IDX]].add(
        tmpd_frac * transfer
    )
    new_vmrcw = new_vmrcw.at[..., data.VMRCW_SO4[data.AITKEN_MODE_IDX]].add(
        tmpe_frac * transfer
    )

    return new_vmr, new_vmrcw
