"""Modal aerosol water uptake — JAX port of ``modal_aero_wateruptake_dr``/``_sub``.

This is the user-facing process function per ADR-009. It composes the
Köhler equilibrium solver from :mod:`mam4_jax.physics.kohler` and the saturation
vapor pressure / humidity primitives from :mod:`mam4_jax.physics.saturation`.

Port targets (`mam4-original-src-code/e3sm_src_modified/modal_aero_wateruptake.F90`):

* ``modal_aero_wateruptake_dr`` (lines 130–392) — driver. Extracts per-
  mode dry quantities from the tracer array via the ``IndexTables``
  bookkeeping (ADR-008) and the per-species property tables
  (``mam4_jax.core.data.SPECDENS_AMODE`` / ``SPECHYGRO_AMODE``); computes RH
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

from mam4_jax.core.constants import RHOH2O
from mam4_jax.core.data import (
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
from mam4_jax.physics.kohler import modal_aero_kohler
from mam4_jax.physics.saturation import qsat_water

# Geometric constants matching the Fortran's local parameters
# (modal_aero_wateruptake.F90:31-32).
_PI    = math.pi
_PI43  = _PI * 4.0 / 3.0
_THIRD = 1.0 / 3.0

# h2ommr index in the tracer array. The driver does:
#   h2ommr => state%q(:,:,1)         (Fortran 1-based → Python 0-based = 0)
_H2OMMR_IDX: int = 0


class WateruptakeTables:
    """Every mode/species table wateruptake reads, as one bundle.

    The default (`_E3SM_TABLES`, from the module constants above) keeps
    the E3SM path bit-identical; the CAM driver builds instances from a
    `Topology` with indices in its own state coordinate (plan 025 G4b).
    """

    def __init__(self, *, lmassptr, slot_valid, per_slot_density,
                 per_slot_hygro, sigmag, rhcrystal, rhdeliques):
        self.lmassptr = np.asarray(lmassptr)
        self.slot_valid = np.asarray(slot_valid)
        self.per_slot_density = np.asarray(per_slot_density)
        self.per_slot_hygro = np.asarray(per_slot_hygro)
        self.sigmag = np.asarray(sigmag)
        self.rhcrystal = np.asarray(rhcrystal)
        self.rhdeliques = np.asarray(rhdeliques)


_E3SM_TABLES = WateruptakeTables(
    lmassptr=INDEX_TABLES.lmassptr_amode, slot_valid=SLOT_VALID,
    per_slot_density=PER_SLOT_DENSITY, per_slot_hygro=PER_SLOT_HYGRO,
    sigmag=SIGMAG_AMODE, rhcrystal=RHCRYSTAL_AMODE,
    rhdeliques=RHDELIQUES_AMODE,
)


def _safe_div(numer, denom, floor: float, fallback):
    """Return ``numer / denom`` where ``denom > floor``, else ``fallback``.

    Both branches of :func:`jax.numpy.where` are evaluated, so the
    division must use a guarded denominator to avoid NaN propagating into
    the result on the masked-out path.
    """
    safe_denom = jnp.where(denom > floor, denom, 1.0)
    return jnp.where(denom > floor, numer / safe_denom, fallback)


def wateruptake(state: dict[str, Any], params=None, config=None,
                *, tables: WateruptakeTables | None = None,
                qv=None, strat=None) -> dict[str, Any]:
    """Compute aerosol equilibrium water uptake. ADR-009 entry point.

    See module docstring for the ``state`` dict contract.

    ``tables``: mode/species tables; ``None`` (default) = the E3SM
    MAM4-MOM constants — bit-identical to the pre-parameter behaviour.
    ``qv``: water-vapor mass mixing ratio; ``None`` (default) reads
    ``q[..., 0]`` (the E3SM box contract, where slot 0 is Q). The CAM
    driver keeps water vapor OUTSIDE its aerosol window and passes it
    explicitly.

    ``strat``: CAM's stratospheric-sulfate water uptake
    (``modal_aero_wateruptake_sub``'s ``modal_strat_sulfate .and.
    k < troplev`` branch, CESM3 modal_aero_wateruptake.F90:583-591 —
    the box driver sets its tropopause above the single level, so under
    ``strat`` the branch is live EVERYWHERE). ``None`` (default) = the
    Köhler + hysteresis path only, bit-identical to before. Otherwise a
    dict with:

    * ``so4_slot``: per-mode slot index of the so4 species on the slot
      axis (−1 = mode has none; its ``so4dryvol`` is then 0 and the
      formula collapses to a dry particle);
    * ``so4specdens``: sulfate density (kg/m³);
    * ``wtpct``, ``sulden``: (..., nmodes) — the Tabazadeh composition
      (weight % H2SO4) and solution density (g/cm³) from
      :func:`mam4_jax.physics.strat_sulfate.calc_h2so4_equilib_mixrat`.

    The wet volume is then solution volume from the wt% composition:
    ``(dryvol − so4dryvol) + so4dryvol·ρ_so4/(sulden·wtpct·10)``, floored
    at ``dryvol`` — no Köhler, no hysteresis (upstream behaviour).
    """
    del params, config
    tb = _E3SM_TABLES if tables is None else tables

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
    lmass_idx = jnp.asarray(tb.lmassptr, dtype=jnp.int32)
    safe_idx  = jnp.where(jnp.asarray(tb.slot_valid), lmass_idx, 0)
    slot_mask = jnp.asarray(tb.slot_valid, dtype=jnp.float64)

    # raer[..., m, s]: contribution of species (m, s) to mode m, zeroed
    # for invalid slots so the sums below are unaffected.
    q_gathered = jnp.take(q, safe_idx, axis=-1)              # (..., m, s)
    raer = q_gathered * slot_mask                            # (..., m, s)

    per_slot_density = jnp.asarray(tb.per_slot_density)      # (m, s)
    per_slot_hygro   = jnp.asarray(tb.per_slot_hygro)        # (m, s)

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
    sigmag = jnp.asarray(tb.sigmag)                                   # (m,)
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

    h2ommr = q[..., _H2OMMR_IDX] if qv is None else jnp.asarray(qv)
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
    rhcrystal  = jnp.asarray(tb.rhcrystal)
    rhdeliques = jnp.asarray(tb.rhdeliques)
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

    if strat is not None:
        # CAM stratospheric-sulfate water (see docstring). so4dryvol is
        # the single-particle so4 dry volume, calcsize's normalisation
        # (modal_aero_calcsize.F90:1571-1576):
        #   so4dryvol = dryvol * so4dryvolmr/dryvolmr   (0 below 1e-31)
        so4_slot = np.asarray(strat["so4_slot"])
        nmode = raer.shape[-1] if raer.ndim == 2 else raer.shape[-2]
        so4dryvolmr = jnp.zeros_like(dryvolmr)
        for m in range(len(so4_slot)):
            if so4_slot[m] >= 0:
                so4dryvolmr = so4dryvolmr.at[..., m].set(
                    jnp.maximum(raer[..., m, int(so4_slot[m])], 0.0)
                    / strat["so4specdens"])
        safe_dvmr = jnp.where(dryvolmr > 1.0e-31, dryvolmr, 1.0)
        so4dryvol = jnp.where(so4dryvolmr > 1.0e-31,
                              dryvol * so4dryvolmr / safe_dvmr, 0.0)
        wetvol_s = ((dryvol - so4dryvol)
                    + so4dryvol * strat["so4specdens"]
                    / (jnp.asarray(strat["sulden"])
                       * jnp.asarray(strat["wtpct"]) * 10.0))
        wetvol = jnp.maximum(wetvol_s, dryvol)
        wetrad = jnp.maximum((wetvol / _PI43) ** _THIRD, dryrad)
        wtrvol = jnp.maximum(wetvol - dryvol, 0.0)

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
