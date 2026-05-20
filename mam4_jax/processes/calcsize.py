"""Modal aerosol size redistribution — JAX port of ``modal_aero_calcsize_sub``.

This is the user-facing process function per ADR-009. Port target:
``mam4-original-src-code/box_model_utils/modal_aero_calcsize.F90:471-1387``.

**Current scope (M3.5 PR-A):** the per-mode number-bounds adjustment and
the recomputation of ``dgncur_a`` from updated mass + adjusted number.
The Aitken ↔ accumulation mode-transfer block (Fortran lines 944–1380)
is **not yet implemented** here — calling this function is equivalent
to invoking the Fortran ``modal_aero_calcsize_sub`` with
``do_aitacc_transfer_in=.false.``. The transfer block lands in PR-B.

In the canonical box-model reference (``run_test.csh`` namelist defaults)
the Aitken ↔ accum transfer **never triggers** anyway (mode mean
diameters stay inside the ``dgnumlo``/``dgnumhi`` bounds for the full
60-step run), so the JAX output here is numerically identical to the
full Fortran output for this configuration. See ``docs/DEFERRED.md``
for the coverage-gap discussion.

Configuration assumptions (matching Fortran defaults for MAM4-MOM):

* ``mprognum_amode == 1`` for all modes (number is *prognostic*, dgnum is
  diagnostic). The Fortran "option 1" branch (lines 688–711, number
  diagnosed from fixed dgnum) is not implemented.
* ``do_adjust == True`` (number-bounds adjustment is on). The
  ``do_adjust == False`` branch is not implemented; the JAX function
  always applies the 3-step bounds adjustment.
* ``do_aitacc_transfer == False`` (set internally for PR-A; PR-B will
  add the True branch).

State dict contract:

    state['q']        shape (..., pcnst)       — interstitial tracer mass mixing ratios
    state['qqcw']     shape (..., pcnst)       — cloud-borne tracer mass mixing ratios
    state['dgncur_a'] shape (..., ntot_amode)  — current dry mode diameter (m)
    state['deltat']   scalar                   — timestep (s)

    returns new state with same keys updated.

The function additionally writes ``dgncur_c`` (cloud-borne diameter) and
``v2ncur_a`` / ``v2ncur_c`` (volume-to-number ratios) into the returned
state for downstream use.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from mam4_jax.data import (
    ALNSG_AMODE,
    DGNUM_AMODE,
    DGNUMHI_AMODE,
    DGNUMLO_AMODE,
    DUMFAC_AMODE,
    INDEX_TABLES,
    PER_SLOT_DENSITY,
    SLOT_VALID,
    VOLTONUMB_AMODE,
    VOLTONUMBHI_AMODE,
    VOLTONUMBLO_AMODE,
)

# Constants from Fortran modal_aero_calcsize.F90:
#   line 533 — third = 1/3
#   line 627 — tadj  = 86400 (1 day adjustment time scale)
#   line 748 — frelaxadj = 27 (= 3^3; relaxed bounds = strict ÷ frelaxadj
#              on the upper side and × frelaxadj on the lower)
_THIRD     = 1.0 / 3.0
_TADJ_S    = 86400.0
_FRELAXADJ = 27.0


def _gather_per_slot(q: jnp.ndarray, lmass_idx: jnp.ndarray,
                     slot_mask: jnp.ndarray) -> jnp.ndarray:
    """Gather q[..., lmassptr_amode[m, s]] with unused slots zeroed.

    Returns array of shape ``(..., NTOT_AMODE, MAXD_ASPECTYPE)``. The
    same trick used in :mod:`mam4_jax.processes.wateruptake` —
    ``safe_idx`` replaces -1 sentinels with 0 so ``jnp.take`` succeeds;
    the result is then masked back to zero where slots are unused.
    """
    safe_idx = jnp.where(slot_mask.astype(jnp.bool_), lmass_idx, 0)
    gathered = jnp.take(q, safe_idx, axis=-1)
    return gathered * slot_mask


def _adjusted_num_drv_c_zero(num_a: jnp.ndarray, drv_a: jnp.ndarray,
                             v2nxx: jnp.ndarray, v2nyy: jnp.ndarray,
                             fracadj: jnp.ndarray) -> jnp.ndarray:
    """Number-bounds adjustment for the ``drv_c <= 0`` branch.

    Fortran lines 794–802: apply the strict bounds (skip the relaxed step
    2) and nudge by ``fracadj``. Returns the updated ``num_a``.
    """
    num_a1 = num_a
    numbnd = jnp.maximum(drv_a * v2nxx,
                         jnp.minimum(drv_a * v2nyy, num_a1))
    return num_a1 + (numbnd - num_a1) * fracadj


def _adjusted_num_drv_a_zero(num_c: jnp.ndarray, drv_c: jnp.ndarray,
                             v2nxx: jnp.ndarray, v2nyy: jnp.ndarray,
                             fracadj: jnp.ndarray) -> jnp.ndarray:
    """Number-bounds adjustment for the ``drv_a <= 0`` branch (mirror)."""
    num_c1 = num_c
    numbnd = jnp.maximum(drv_c * v2nxx,
                         jnp.minimum(drv_c * v2nyy, num_c1))
    return num_c1 + (numbnd - num_c1) * fracadj


def _adjusted_num_both_positive(
    num_a: jnp.ndarray, drv_a: jnp.ndarray,
    num_c: jnp.ndarray, drv_c: jnp.ndarray,
    v2nxx: jnp.ndarray, v2nyy: jnp.ndarray,
    v2nxxrl: jnp.ndarray, v2nyyrl: jnp.ndarray,
    fracadj: jnp.ndarray,
):
    """3-step number-bounds adjustment (Fortran lines 812–869) when both
    interstitial and cloud-borne dry volumes are positive.

    Returns ``(num_a, num_c)`` after the full step1 → step2 → step3 chain.
    """
    # Step 1: just enforce non-negative (already done by callers).
    num_a1 = num_a
    num_c1 = num_c

    # Step 2: relaxed bounds applied to interstitial and cloud-borne
    # individually, with a "transfer" between a and c when only one
    # branch needed adjustment.
    numbnd_a = jnp.maximum(drv_a * v2nxxrl,
                            jnp.minimum(drv_a * v2nyyrl, num_a1))
    delnum_a2 = (numbnd_a - num_a1) * fracadj
    num_a2 = num_a1 + delnum_a2

    numbnd_c = jnp.maximum(drv_c * v2nxxrl,
                            jnp.minimum(drv_c * v2nyyrl, num_c1))
    delnum_c2 = (numbnd_c - num_c1) * fracadj
    num_c2 = num_c1 + delnum_c2

    # Cross-coupling: when only one side changed, push the other in the
    # opposite direction as much as relaxed bounds allow.
    a_zero_c_nonzero = (delnum_a2 == 0.0) & (delnum_c2 != 0.0)
    a_nonzero_c_zero = (delnum_a2 != 0.0) & (delnum_c2 == 0.0)
    num_a2 = jnp.where(
        a_zero_c_nonzero,
        jnp.maximum(drv_a * v2nxxrl,
                    jnp.minimum(drv_a * v2nyyrl, num_a1 - delnum_c2)),
        num_a2,
    )
    num_c2 = jnp.where(
        a_nonzero_c_zero,
        jnp.maximum(drv_c * v2nxxrl,
                    jnp.minimum(drv_c * v2nyyrl, num_c1 - delnum_a2)),
        num_c2,
    )

    # Step 3: stricter bounds on the combined number num_t = num_a + num_c.
    drv_t  = drv_a + drv_c
    num_t2 = num_a2 + num_c2

    below_lo = num_t2 < drv_t * v2nxx
    above_hi = num_t2 > drv_t * v2nyy

    delnum_t3 = jnp.where(below_lo, (drv_t * v2nxx - num_t2) * fracadj,
                jnp.where(above_hi, (drv_t * v2nyy - num_t2) * fracadj,
                          0.0))

    # When pushing toward v2nxx (number too low), allocate increase among
    # the side(s) that are below v2nxx; symmetric when pushing toward v2nyy.
    a_low = num_a2 < drv_a * v2nxx
    c_low = num_c2 < drv_c * v2nxx
    a_hi  = num_a2 > drv_a * v2nyy
    c_hi  = num_c2 > drv_c * v2nyy

    # safe_t2 avoids 0/0 in the "both below_lo with num_t2 == 0" edge case
    safe_t2 = jnp.where(num_t2 != 0.0, num_t2, 1.0)

    delnum_a3 = jnp.where(
        below_lo,
        jnp.where(a_low & c_low, delnum_t3 * (num_a2 / safe_t2),
                  jnp.where(a_low, delnum_t3, 0.0)),
        jnp.where(
            above_hi,
            jnp.where(a_hi & c_hi, delnum_t3 * (num_a2 / safe_t2),
                      jnp.where(a_hi, delnum_t3, 0.0)),
            0.0,
        ),
    )
    delnum_c3 = jnp.where(
        below_lo,
        jnp.where(a_low & c_low, delnum_t3 * (num_c2 / safe_t2),
                  jnp.where(c_low, delnum_t3, 0.0)),
        jnp.where(
            above_hi,
            jnp.where(a_hi & c_hi, delnum_t3 * (num_c2 / safe_t2),
                      jnp.where(c_hi, delnum_t3, 0.0)),
            0.0,
        ),
    )

    return num_a2 + delnum_a3, num_c2 + delnum_c3


def _compute_dgn_v2n(num: jnp.ndarray, drv: jnp.ndarray,
                    v2nxx: jnp.ndarray, v2nyy: jnp.ndarray,
                    dgnxx: jnp.ndarray, dgnyy: jnp.ndarray,
                    dumfac: jnp.ndarray):
    """Recompute ``(dgncur, v2ncur)`` from final ``(num, drv)``.

    Fortran lines 877–887:
      * if num <= drv*v2nxx → clamp to dgnxx, v2nxx
      * if num >= drv*v2nyy → clamp to dgnyy, v2nyy
      * else                → dgncur = (drv / (dumfac · num))^(1/3),
                              v2ncur = num / drv

    When drv <= 0 the caller keeps the default (dgnum_amode, voltonumb_amode)
    that was set at the top of the per-mode loop.
    """
    above_hi = num <= drv * v2nxx
    below_lo = num >= drv * v2nyy
    safe_num = jnp.where(num > 0, num, 1.0)
    safe_drv = jnp.where(drv > 0, drv, 1.0)
    dgn_mid = (safe_drv / (dumfac * safe_num)) ** _THIRD
    v2n_mid = num / safe_drv

    dgn = jnp.where(above_hi, dgnxx,
          jnp.where(below_lo, dgnyy, dgn_mid))
    v2n = jnp.where(above_hi, v2nxx,
          jnp.where(below_lo, v2nyy, v2n_mid))
    return dgn, v2n


def calcsize(state: dict[str, Any], params=None, config=None) -> dict[str, Any]:
    """Apply size redistribution. ADR-009 entry point.

    See module docstring for the ``state`` dict contract and the
    PR-A / PR-B scope split. PR-A handles the per-mode bounds adjustment
    and dgncur_a recomputation; the Aitken↔accum transfer (PR-B) is
    deferred but is a no-op in the canonical reference.
    """
    del params, config

    q        = jnp.asarray(state["q"],        dtype=jnp.float64)
    qqcw     = jnp.asarray(state["qqcw"],     dtype=jnp.float64)
    deltat   = jnp.asarray(state["deltat"],   dtype=jnp.float64)
    # dgncur_a is in state for the wateruptake / downstream consumers,
    # but calcsize derives its own new value below.

    # Index tables and per-(mode, slot) species properties.
    lmass_idx        = jnp.asarray(INDEX_TABLES.lmassptr_amode,   dtype=jnp.int32)
    lmass_idx_cw     = jnp.asarray(INDEX_TABLES.lmassptrcw_amode, dtype=jnp.int32)
    numptr           = jnp.asarray(INDEX_TABLES.numptr_amode,     dtype=jnp.int32)
    numptr_cw        = jnp.asarray(INDEX_TABLES.numptrcw_amode,   dtype=jnp.int32)
    slot_mask        = jnp.asarray(SLOT_VALID, dtype=jnp.float64)
    per_slot_density = jnp.asarray(PER_SLOT_DENSITY)

    # Per-mode bound constants (broadcast as (m,)).
    v2nxx = jnp.asarray(VOLTONUMBHI_AMODE)
    v2nyy = jnp.asarray(VOLTONUMBLO_AMODE)
    v2nxxrl = v2nxx / _FRELAXADJ
    v2nyyrl = v2nyy * _FRELAXADJ
    dgnxx = jnp.asarray(DGNUMHI_AMODE)
    dgnyy = jnp.asarray(DGNUMLO_AMODE)
    dumfac = jnp.asarray(DUMFAC_AMODE)
    voltonumb_amode = jnp.asarray(VOLTONUMB_AMODE)

    # Adjustment time scale (Fortran lines 626–631).
    tadj = jnp.maximum(_TADJ_S, deltat)
    tadj_inv = 1.0 / (tadj * (1.0 + 1.0e-15))
    fracadj = jnp.clip(deltat * tadj_inv, 0.0, 1.0)

    # Per-mode dry volume "mixing ratio" (m³/kg) for interstitial / cloud-borne.
    raer_a = _gather_per_slot(q,    lmass_idx,    slot_mask)      # (..., m, s)
    raer_c = _gather_per_slot(qqcw, lmass_idx_cw, slot_mask)

    # Fortran line 668 uses max(0, q) — protect against negative spurious
    # mass values.
    pos_raer_a = jnp.maximum(raer_a, 0.0)
    pos_raer_c = jnp.maximum(raer_c, 0.0)
    drv_a = jnp.sum(pos_raer_a / per_slot_density, axis=-1)        # (..., m)
    drv_c = jnp.sum(pos_raer_c / per_slot_density, axis=-1)

    # Gather number tracers per mode.
    num_a_in = jnp.take(q,    numptr,    axis=-1)                  # (..., m)
    num_c_in = jnp.take(qqcw, numptr_cw, axis=-1)
    num_a = jnp.maximum(num_a_in, 0.0)
    num_c = jnp.maximum(num_c_in, 0.0)

    # Bounds-adjustment branches.
    a_pos = drv_a > 0.0
    c_pos = drv_c > 0.0
    both_zero        = (~a_pos) & (~c_pos)
    only_a_positive  =  a_pos   & (~c_pos)
    only_c_positive  = (~a_pos) &   c_pos
    both_positive    =  a_pos   &   c_pos

    # Compute candidate new num_a / num_c for each branch.
    num_a_only_a = _adjusted_num_drv_c_zero(num_a, drv_a, v2nxx, v2nyy, fracadj)
    num_c_only_c = _adjusted_num_drv_a_zero(num_c, drv_c, v2nxx, v2nyy, fracadj)
    num_a_both, num_c_both = _adjusted_num_both_positive(
        num_a, drv_a, num_c, drv_c,
        v2nxx, v2nyy, v2nxxrl, v2nyyrl, fracadj,
    )

    # Compose final num_a / num_c. Branch precedence mirrors the Fortran
    # nested if/elif:
    num_a_final = jnp.where(both_zero,       0.0,
                  jnp.where(only_a_positive, num_a_only_a,
                  jnp.where(only_c_positive, 0.0,
                  jnp.where(both_positive,   num_a_both, num_a))))

    num_c_final = jnp.where(both_zero,       0.0,
                  jnp.where(only_a_positive, 0.0,
                  jnp.where(only_c_positive, num_c_only_c,
                  jnp.where(both_positive,   num_c_both, num_c))))

    # Recompute dgncur_a / v2ncur_a (and _c) from final (num, drv).
    # When drv <= 0 the Fortran loop keeps the per-mode defaults
    # (dgnum_amode, voltonumb_amode) that were set at the top of the loop.
    # We mirror that with a final jnp.where fallback.
    dgnum_amode = jnp.asarray(DGNUM_AMODE)

    dgncur_a_new, v2ncur_a_new = _compute_dgn_v2n(
        num_a_final, drv_a, v2nxx, v2nyy, dgnxx, dgnyy, dumfac,
    )
    dgncur_a_new = jnp.where(drv_a > 0.0, dgncur_a_new,
                             jnp.broadcast_to(dgnum_amode, drv_a.shape))
    v2ncur_a_new = jnp.where(drv_a > 0.0, v2ncur_a_new,
                             jnp.broadcast_to(voltonumb_amode, drv_a.shape))

    dgncur_c_new, v2ncur_c_new = _compute_dgn_v2n(
        num_c_final, drv_c, v2nxx, v2nyy, dgnxx, dgnyy, dumfac,
    )
    dgncur_c_new = jnp.where(drv_c > 0.0, dgncur_c_new,
                             jnp.broadcast_to(dgnum_amode, drv_c.shape))
    v2ncur_c_new = jnp.where(drv_c > 0.0, v2ncur_c_new,
                             jnp.broadcast_to(voltonumb_amode, drv_c.shape))

    # Scatter the updated number tracers back into q / qqcw via the
    # tendency Fortran computes (num_final - num_in) * deltat_inv * deltat
    # = num_final - num_in. Equivalent to writing num_final at the pcnst
    # index.
    new_q    = q   .at[..., numptr   ].set(num_a_final)
    new_qqcw = qqcw.at[..., numptr_cw].set(num_c_final)

    return {
        **state,
        "q":         new_q,
        "qqcw":      new_qqcw,
        "dgncur_a":  dgncur_a_new,
        "dgncur_c":  dgncur_c_new,
        "v2ncur_a":  v2ncur_a_new,
        "v2ncur_c":  v2ncur_c_new,
    }
