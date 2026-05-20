"""Modal aerosol water uptake — JAX port of ``modal_aero_wateruptake_dr``/``_sub``.

This is the user-facing process function per ADR-009. It composes the
Köhler equilibrium solver from :mod:`mam4_jax.kohler` and the saturation
vapor pressure / humidity primitives from :mod:`mam4_jax.saturation`.

Port targets (`mam4-original-src-code/e3sm_src_modified/modal_aero_wateruptake.F90`):

* ``modal_aero_wateruptake_dr`` (lines 130–392) — driver. Extracts per-
  mode dry quantities from the tracer array via the ``IndexTables``
  bookkeeping (ADR-008) and the per-species property tables
  (``mam4_jax.data.SPECDENS_AMODE`` / ``SPECHYGRO_AMODE``); computes RH
  from ``q[h2ommr]`` and ``qsat_water(t, pmid)`` with the clear-sky
  cloud adjustment; orchestrates the per-mode Köhler call.

* ``modal_aero_wateruptake_sub`` (lines 396–485) — per-(column, level,
  mode) Köhler + deliquescence/crystallization hysteresis. Folded into
  this module because both ports are tightly coupled and share the same
  pre-built arrays.

The Fortran's ``state`` / ``physics_buffer_desc`` argument tree is
flattened to a plain dict here. Inputs and outputs:

    state['q']           shape (..., pcnst)         — tracer mass mixing ratios
    state['dgncur_a']    shape (..., ntot_amode)    — dry mode diameter (m)
    state['t']           shape (...,)               — temperature (K)
    state['pmid']        shape (...,)               — mid-layer pressure (Pa)
    state['cldn']        shape (...,)               — cloud fraction (0..1)

    return value: new state dict carrying the same keys plus:
    ['dgncur_awet']      shape (..., ntot_amode)    — wet mode diameter (m)
    ['qaerwat']          shape (..., ntot_amode)    — aerosol water (kg/kg)
    ['wetdens']          shape (..., ntot_amode)    — wet aerosol density (kg/m³)
"""
from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from mam4_jax.constants import RHOH2O
from mam4_jax.data import (
    INDEX_TABLES,
    LSPECTYPE_AMODE,
    NTOT_AMODE,
    PER_SLOT_DENSITY,
    PER_SLOT_HYGRO,
    RHCRYSTAL_AMODE,
    RHDELIQUES_AMODE,
    SIGMAG_AMODE,
    SLOT_VALID,
)
from mam4_jax.kohler import modal_aero_kohler
from mam4_jax.saturation import qsat_water

# Geometric constants matching the Fortran's local parameters
# (modal_aero_wateruptake.F90:31-32).
_PI    = math.pi
_PI43  = _PI * 4.0 / 3.0
_THIRD = 1.0 / 3.0

# h2ommr index in the tracer array. The driver does:
#   h2ommr => state%q(:,:,1)         (Fortran 1-based → Python 0-based = 0)
_H2OMMR_IDX: int = 0


def _safe_div(numer, denom, floor: float, fallback):
    """Return ``numer / denom`` where ``denom > floor``, else ``fallback``.

    Both branches of :func:`jax.numpy.where` are evaluated, so the
    division must use a guarded denominator to avoid NaN propagating into
    the result on the masked-out path.
    """
    safe_denom = jnp.where(denom > floor, denom, 1.0)
    return jnp.where(denom > floor, numer / safe_denom, fallback)


def wateruptake(state: dict[str, Any], params=None, config=None) -> dict[str, Any]:
    """Compute aerosol equilibrium water uptake. ADR-009 entry point.

    See module docstring for the ``state`` dict contract.
    """
    del params, config  # All constants live in mam4_jax.data / .constants.

    q        = jnp.asarray(state["q"],         dtype=jnp.float64)
    dgncur_a = jnp.asarray(state["dgncur_a"],  dtype=jnp.float64)
    t        = jnp.asarray(state["t"],         dtype=jnp.float64)
    pmid     = jnp.asarray(state["pmid"],      dtype=jnp.float64)
    cldn     = jnp.asarray(state["cldn"],      dtype=jnp.float64)

    # ---------------------------------------------------------------------
    # Step 1 — per-mode dry quantities (Fortran lines 263–329).
    # ---------------------------------------------------------------------

    # Gather per-(mode, slot) mass mixing ratio: q[..., lmassptr_amode[m, s]].
    # For unused slots (lmassptr_amode == -1) we use index 0 then zero out
    # the result via SLOT_VALID — this keeps every contribution but ignores
    # unused species.
    lmass_idx = jnp.asarray(INDEX_TABLES.lmassptr_amode, dtype=jnp.int32)
    safe_idx  = jnp.where(jnp.asarray(SLOT_VALID), lmass_idx, 0)
    slot_mask = jnp.asarray(SLOT_VALID, dtype=jnp.float64)

    # raer[..., m, s]: contribution of species (m, s) to mode m, zeroed
    # for invalid slots so the sums below are unaffected.
    q_gathered = jnp.take(q, safe_idx, axis=-1)              # (..., m, s)
    raer = q_gathered * slot_mask                            # (..., m, s)

    per_slot_density = jnp.asarray(PER_SLOT_DENSITY)         # (m, s)
    per_slot_hygro   = jnp.asarray(PER_SLOT_HYGRO)           # (m, s)

    # Mass / dry-volume / volume-weighted hygro per mode.
    maer       = jnp.sum(raer,                            axis=-1)   # (..., m)
    dryvolmr   = jnp.sum(raer / per_slot_density,         axis=-1)
    hygro_volwgt = jnp.sum(
        raer / per_slot_density * per_slot_hygro, axis=-1
    )

    # Default hygroscopicity if dryvolmr is too small (Fortran line 305:
    # `hygro(i,k,m) = spechygro_1`, where spechygro_1 is the hygro of slot
    # 0 captured during the first species iteration).
    spechygro_1 = per_slot_hygro[:, 0]                                # (m,)
    hygro = _safe_div(hygro_volwgt, dryvolmr, 1.0e-30, spechygro_1)   # (..., m)

    # Per-mode geometric quantities (Fortran lines 310–326).
    sigmag = jnp.asarray(SIGMAG_AMODE)                                # (m,)
    alnsg  = jnp.log(sigmag)
    v2ncur_a = 1.0 / ((_PI / 6.0) * dgncur_a ** 3
                       * jnp.exp(4.5 * alnsg ** 2))                   # (..., m)
    naer    = dryvolmr * v2ncur_a
    drydens = _safe_div(maer, dryvolmr, 1.0e-31, 1.0)
    dryvol  = 1.0 / v2ncur_a
    drymass = drydens * dryvol
    dryrad  = (dryvol / _PI43) ** _THIRD

    # ---------------------------------------------------------------------
    # Step 2 — relative humidity (Fortran lines 333–362).
    # ---------------------------------------------------------------------

    h2ommr = q[..., _H2OMMR_IDX]
    qs     = qsat_water(t, pmid)

    rh = jnp.where(qs > h2ommr, h2ommr / jnp.maximum(qs, 1e-30), 0.98)
    rh = jnp.minimum(jnp.maximum(rh, 0.0), 0.98)

    # Clear-sky adjustment (cldn_thresh = 1.0 for the non-pergro_mods path).
    rh = jnp.where(
        cldn < 1.0,
        (rh - cldn) / jnp.maximum(1.0 - cldn, 1e-30),
        rh,
    )
    rh = jnp.maximum(rh, 0.0)

    # ---------------------------------------------------------------------
    # Step 3 — per-mode wet quantities (Fortran lines 437–476, the "_sub").
    # ---------------------------------------------------------------------

    # Broadcast rh to per-mode shape for jnp.where with (..., m) arrays.
    rh_pm = jnp.broadcast_to(rh[..., None], dryrad.shape)

    # Call the Köhler solver on a flattened view so it sees a 1D batch.
    flat_shape = dryrad.shape
    wetrad_kohler = modal_aero_kohler(
        dryrad.ravel(), hygro.ravel(), rh_pm.ravel()
    ).reshape(flat_shape)

    # Quartic-solution post-processing (Fortran lines 448–452).
    wetrad_q = jnp.maximum(wetrad_kohler, dryrad)
    wetvol_q = jnp.maximum(_PI43 * wetrad_q ** 3, dryvol)
    wtrvol_q = jnp.maximum(wetvol_q - dryvol, 0.0)

    # Hysteresis branches (Fortran lines 457–466):
    #   rh < rhcrystal           → collapse to dry
    #   rhcrystal ≤ rh < rhdeliques → linear interpolation
    #   rh ≥ rhdeliques          → use the Köhler result as-is
    rhcrystal  = jnp.asarray(RHCRYSTAL_AMODE)
    rhdeliques = jnp.asarray(RHDELIQUES_AMODE)
    hystfac    = 1.0 / jnp.maximum(1.0e-5, rhdeliques - rhcrystal)

    below_crystal = rh_pm < rhcrystal
    in_hysteresis = (~below_crystal) & (rh_pm < rhdeliques)

    wtrvol_h = jnp.maximum(wtrvol_q * hystfac * (rh_pm - rhcrystal), 0.0)
    wetvol_h = dryvol + wtrvol_h
    wetrad_h = (wetvol_h / _PI43) ** _THIRD

    # Compose final per-mode values.
    wtrvol = jnp.where(below_crystal, 0.0,
              jnp.where(in_hysteresis, wtrvol_h, wtrvol_q))
    wetvol = jnp.where(below_crystal, dryvol,
              jnp.where(in_hysteresis, wetvol_h, wetvol_q))
    wetrad = jnp.where(below_crystal, dryrad,
              jnp.where(in_hysteresis, wetrad_h, wetrad_q))

    # Outputs (Fortran lines 469–476).
    dgncur_awet = dgncur_a * (wetrad / dryrad)
    qaerwat     = RHOH2O * naer * wtrvol

    # specdens_1 per mode is the density of slot 0 (Fortran line 282).
    specdens_1 = per_slot_density[:, 0]                               # (m,)
    wetdens = _safe_div(
        drymass + RHOH2O * wtrvol, wetvol, 1.0e-30, specdens_1
    )

    return {
        **state,
        "dgncur_awet": dgncur_awet,
        "qaerwat":     qaerwat,
        "wetdens":     wetdens,
    }
