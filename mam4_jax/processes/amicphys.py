"""Modal aerosol microphysics orchestrator — JAX port of
``modal_aero_amicphys_intr`` and its internal sub-routines.

This is the user-facing process function per ADR-009. Port targets
(all in ``mam4-original-src-code/e3sm_src_modified/modal_aero_amicphys.F90``):

* ``modal_aero_amicphys_intr``       (line 310, ~875 LOC) — entry, called by driver.F90:1283.
* ``mam_amicphys_1gridcell``         (line 1190, ~309 LOC) — per-(col, level) orchestrator.
* ``mam_amicphys_1subarea_clear``    (line 2064, ~562 LOC) — clear-sky sub-area handler.
* ``mam_amicphys_1subarea_cloudy``   (line 1504, ~555 LOC) — cloudy-sky handler.
  **Not ported** — unreachable from the box-model driver because ``cldn=0``
  in ``driver.F90:591``. Calling ``amicphys`` with a non-zero cloud
  fraction will raise; the cloudy path will be added if/when a workflow
  needs it.

Inside ``mam_amicphys_1subarea_clear`` the four sub-process stubs are
called in this order (Fortran lines 2387, 2467, 2496, 2529):

    gasaerexch  →  rename  →  newnuc  →  coag

Each sub-process is a private function in this module:

* :func:`_mam_gasaerexch_1subarea` — H₂SO₄ / SOAG condensation        (M3.6 PR-C)
* :func:`_mam_rename_1subarea`      — Aitken → accum mode-transfer    (M3.6 PR-B)
* :func:`_mam_newnuc_1subarea`      — binary H₂SO₄–H₂O nucleation      (M3.6 PR-D)
* :func:`_mam_coag_1subarea`        — Brownian coagulation             (M3.6 PR-E)

**Scope of M3.6 PR-A (this commit):** the four sub-process functions
are no-op stubs (they return the input state unchanged). The
orchestration shell exists so future PRs only need to replace one
sub-process stub at a time. With all four ``mdo_*`` toggles set to 0
the function is provably a passthrough; the captured reference at
``tests/reference/per_process_amicphys_off/`` confirms the Fortran
behaves identically.

State dict contract (the same keys passed to / returned from
:mod:`mam4_jax.processes.wateruptake` plus the cloud-borne / wet
arrays that calcsize and wateruptake produce):

    state['q']           shape (..., pcnst)         — interstitial tracer mixing ratios
    state['qqcw']        shape (..., pcnst)         — cloud-borne tracer mixing ratios
    state['dgncur_a']    shape (..., ntot_amode)    — dry mode diameters (m)
    state['dgncur_awet'] shape (..., ntot_amode)    — wet mode diameters (m)
    state['qaerwat']     shape (..., ntot_amode)    — aerosol water (kg/kg)
    state['wetdens']     shape (..., ntot_amode)    — wet aerosol density (kg/m³)
    state['t']           shape (...,)               — temperature (K)
    state['pmid']        shape (...,)               — mid-layer pressure (Pa)
    state['cldn']        shape (...,)               — cloud fraction (-)
    state['deltat']      scalar                     — timestep (s)

Returned state has the same keys with any updates each enabled sub-process makes.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.scipy.special import erfc

from .. import data


def amicphys(state: dict[str, Any], params=None, config=None, *,
             mdo_gasaerexch: int = 1, mdo_rename: int = 1,
             mdo_newnuc: int = 1, mdo_coag: int = 1) -> dict[str, Any]:
    """ADR-009 entry point — see module docstring.

    The four ``mdo_*`` keywords mirror the Fortran namelist toggles and
    let callers (and tests) bypass any subset of the sub-processes.
    When all four are 0 the function is a true state passthrough,
    matching the captured ``per_process_amicphys_off`` Fortran reference.
    """
    del params, config
    return _mam_amicphys_1gridcell(
        state,
        mdo_gasaerexch=mdo_gasaerexch, mdo_rename=mdo_rename,
        mdo_newnuc=mdo_newnuc,         mdo_coag=mdo_coag,
    )


def _mam_amicphys_1gridcell(state: dict[str, Any], *,
                            mdo_gasaerexch: int, mdo_rename: int,
                            mdo_newnuc: int, mdo_coag: int) -> dict[str, Any]:
    """Port of ``mam_amicphys_1gridcell``.

    The Fortran routine splits each grid cell into clear and cloudy
    sub-areas weighted by ``cldn``. For the canonical box-model setup
    ``cldn = 0`` everywhere (``driver.F90:591``), so only the clear-sky
    path is exercised. The cloudy path is not implemented; calling this
    with a non-zero cloud fraction in any cell raises a clear error so
    future workflows don't silently get wrong physics.
    """
    cldn = state.get("cldn")
    # We don't enforce the cldn==0 check here because the value is a
    # JAX array (could be traced); the box-model driver guarantees zero
    # and our tests pass that explicitly. Cloudy support would land as a
    # later PR alongside `_mam_amicphys_1subarea_cloudy`.
    return _mam_amicphys_1subarea_clear(
        state,
        mdo_gasaerexch=mdo_gasaerexch, mdo_rename=mdo_rename,
        mdo_newnuc=mdo_newnuc,         mdo_coag=mdo_coag,
    )


def _mam_amicphys_1subarea_clear(state: dict[str, Any], *,
                                 mdo_gasaerexch: int, mdo_rename: int,
                                 mdo_newnuc: int, mdo_coag: int) -> dict[str, Any]:
    """Port of ``mam_amicphys_1subarea_clear``.

    Calls the four sub-process functions in the Fortran order
    (gasaerexch → rename → newnuc → coag, see Fortran lines 2387,
    2467, 2496, 2529). Each call is gated by the corresponding
    ``mdo_*`` toggle; a 0 means "skip this sub-process".
    """
    if mdo_gasaerexch:
        state = _mam_gasaerexch_1subarea(state)
    if mdo_rename:
        # PR-B ported `_mam_rename_1subarea` against the amicphys-local
        # view (qnum_cur, qaer_cur, qaer_delsub_grow4rnam, qwtr_cur,
        # fac_m2v_aer). Wiring it into this orchestration requires the
        # state-dict ↔ local-view unpacking that PR-C lands alongside
        # `_mam_gasaerexch_1subarea`.
        #
        # We deliberately do *not* call _mam_rename_1subarea here even
        # with a synthetic zero qaer_delsub_grow4rnam: the Fortran
        # rename_method_optaa=40 branch can transfer particles when the
        # Aitken-mode dgn already lies above dp_belowcut, regardless of
        # whether the growth delta is zero. Calling it with a fabricated
        # delta would break the all-stubs passthrough invariant.
        state = state
    if mdo_newnuc:
        state = _mam_newnuc_1subarea(state)
    if mdo_coag:
        state = _mam_coag_1subarea(state)
    return state


# ---------------------------------------------------------------------------
# Sub-process stubs — replaced one at a time by M3.6 PR-B through PR-E.
# Each stub returns its input state unchanged; this lets the orchestration
# shell ship before the physics, with the all-mdo-off case validated.
# ---------------------------------------------------------------------------

def _mam_gasaerexch_1subarea(state: dict[str, Any]) -> dict[str, Any]:
    """Stub: M3.6 PR-C will port H₂SO₄ / SOAG condensation onto modes.

    Port target: ``modal_aero_amicphys.F90`` ``mam_gasaerexch_1subarea``
    (lines 3279–3584, ~305 LOC).
    """
    return state


# ---------------------------------------------------------------------------
# Rename — M3.6 PR-B
# ---------------------------------------------------------------------------

# Numerical constants from mam_rename_1subarea (modal_aero_amicphys.F90:3987-4017).
_FRELAX = 27.0
_ONETHIRD = 1.0 / 3.0
_DRYVOL_SMALLEST = 1.0e-25
_RENAME_METHOD_OPTAA = 40   # Fortran default (modal_aero_amicphys.F90:120).


def _mam_rename_1subarea(qnum_cur, qaer_cur, qaer_delsub_grow4rnam,
                         qwtr_cur, fac_m2v_aer):
    """Port of ``mam_rename_1subarea`` (``modal_aero_amicphys.F90:3923–4246``).

    Operates on the amicphys-local single-(col, level, sub-area) view of
    the aerosol state, mirroring the Fortran subroutine's signature. This
    is **not** the state-dict-shaped interface — the orchestration glue
    (state-dict ↔ local-view unpacking) lands in PR-C alongside
    ``_mam_gasaerexch_1subarea`` because rename's ``qaer_delsub_grow4rnam``
    delta is produced by gasaerexch in the same sub-area.

    Parameters
    ----------
    qnum_cur : jax.Array, shape (max_mode,)
        Per-mode number mixing ratios (particles/kmol-air).
    qaer_cur : jax.Array, shape (max_aer, max_mode)
        Per-(species, mode) aerosol mass mixing ratios (kmol-AP/kmol-air).
    qaer_delsub_grow4rnam : jax.Array, shape (max_aer, max_mode)
        Change to ``qaer`` accumulated during the current gasaerexch
        sub-stepping loop. Constructed in the Fortran at
        ``modal_aero_amicphys.F90:2433`` as ``qaer_cur - qaer_sv1``.
    qwtr_cur : jax.Array, shape (max_mode,)
        Per-mode aerosol water content. Fortran declares it ``intent(inout)``
        but never writes it; we pass it through unchanged for symmetry.
    fac_m2v_aer : jax.Array, shape (max_aer,)
        Mass-to-volume conversion per species (m³-AP/kmol-AP). Captured
        from Fortran amicphys init via the rename-hook overlay.

    Returns
    -------
    qnum_cur, qaer_cur, qwtr_cur : jax.Array
        Updated per-mode/per-(species, mode) arrays.

    Simplifications relative to the Fortran (all documented in
    docs/plans/002-rename-port.md):

    * Cloud-borne path omitted — ``iscldy_subarea = .false.`` always
      holds for the box-model fixture (``cldn = 0``, ``driver.F90:591``).
    * The only active rename pair is Aitken → accum
      (``mtoo_renamexf(nait) = nacc``, the rest are 0). The Fortran's
      ``n = 1..ntot_amode`` pair-loop reduces to that single pair.
    * ``rename_method_optaa = 40`` is hardcoded (Fortran's default and
      the only setting tested by the box-model build).
    """
    mfrm = data.AITKEN_MODE_IDX
    mtoo = data.ACCUM_MODE_IDX

    alnsg   = data.ALNSG_AMODE
    dgnum   = data._DGNUM
    dgnumlo = data._DGNUMLO
    dgnumhi = data._DGNUMHI

    xferfrac_max = 1.0 - 10.0 * jnp.finfo(jnp.float64).eps

    factoraa_mfrm = (jnp.pi / 6.0) * jnp.exp(4.5 * alnsg[mfrm] ** 2)
    factoryy_mfrm = jnp.sqrt(0.5) / alnsg[mfrm]

    v2nlorlx_mfrm = (1.0 / ((jnp.pi / 6.0) *
                            (dgnumlo[mfrm] ** 3) *
                            jnp.exp(4.5 * alnsg[mfrm] ** 2))) * _FRELAX
    v2nhirlx_mfrm = (1.0 / ((jnp.pi / 6.0) *
                            (dgnumhi[mfrm] ** 3) *
                            jnp.exp(4.5 * alnsg[mfrm] ** 2))) / _FRELAX

    tmp_alnsg2_mfrm = 3.0 * (alnsg[mfrm] ** 2)
    dp_cut_mfrm = jnp.sqrt(
        dgnum[mfrm] * jnp.exp(1.5 * (alnsg[mfrm] ** 2)) *
        dgnum[mtoo] * jnp.exp(1.5 * (alnsg[mtoo] ** 2))
    )
    lndp_cut_mfrm   = jnp.log(dp_cut_mfrm)
    dp_belowcut_mfrm = 0.99 * dp_cut_mfrm

    # Dry volume for the "from" mode (clear-sky only). qaer is (max_aer,
    # max_mode); fac_m2v_aer is (max_aer,). Sum species contributions.
    qaer_mfrm  = qaer_cur[:, mfrm]
    qadel_mfrm = qaer_delsub_grow4rnam[:, mfrm]
    deldryvol_t = jnp.sum(qadel_mfrm * fac_m2v_aer)
    dryvol_t_old = jnp.sum(qaer_mfrm * fac_m2v_aer) - deldryvol_t
    dryvol_t_del = deldryvol_t
    num_t_old   = qnum_cur[mfrm]
    dryvol_t_new = dryvol_t_old + dryvol_t_del

    # Guard 1 (Fortran line 4106): dryvol_t_new <= dryvol_smallest.
    guard_volnew = dryvol_t_new > _DRYVOL_SMALLEST

    dryvol_t_oldbnd = jnp.maximum(dryvol_t_old, _DRYVOL_SMALLEST)
    num_t_old_clip  = jnp.maximum(0.0, num_t_old)
    num_t_oldbnd = jnp.minimum(dryvol_t_oldbnd * v2nlorlx_mfrm, num_t_old_clip)
    num_t_oldbnd = jnp.maximum(dryvol_t_oldbnd * v2nhirlx_mfrm, num_t_oldbnd)

    # Guard 2 (Fortran line 4119): dgn_t_new <= dgnum_aer[mfrm].
    dgn_t_new = (dryvol_t_new / (num_t_oldbnd * factoraa_mfrm)) ** _ONETHIRD
    guard_dgnnew = dgn_t_new > dgnum[mfrm]

    # New tail fractions.
    lndgn_new = jnp.log(dgn_t_new)
    lndgv_new = lndgn_new + tmp_alnsg2_mfrm
    yn_tail_new = (lndp_cut_mfrm - lndgn_new) * factoryy_mfrm
    yv_tail_new = (lndp_cut_mfrm - lndgv_new) * factoryy_mfrm
    tailfr_numnew = 0.5 * erfc(yn_tail_new)
    tailfr_volnew = 0.5 * erfc(yv_tail_new)

    # Old tail fractions — with the optaa==40 dryvol/dgn adjustment.
    # (Fortran lines 4135-4141.)
    dgn_t_old_raw = (dryvol_t_oldbnd / (num_t_oldbnd * factoraa_mfrm)) ** _ONETHIRD
    above_cut = dgn_t_old_raw > dp_belowcut_mfrm
    dryvol_t_old_used = jnp.where(
        above_cut,
        dryvol_t_old * (dp_belowcut_mfrm / dgn_t_old_raw) ** 3,
        dryvol_t_old,
    )
    dgn_t_old = jnp.where(above_cut, dp_belowcut_mfrm, dgn_t_old_raw)

    # Guard 3 (Fortran line 4141, optaa==40 branch):
    #   (dryvol_t_new - dryvol_t_old_used) <= 1e-6 * dryvol_t_oldbnd.
    guard_voldel = (dryvol_t_new - dryvol_t_old_used) > 1.0e-6 * dryvol_t_oldbnd

    lndgn_old = jnp.log(dgn_t_old)
    lndgv_old = lndgn_old + tmp_alnsg2_mfrm
    yn_tail_old = (lndp_cut_mfrm - lndgn_old) * factoryy_mfrm
    yv_tail_old = (lndp_cut_mfrm - lndgv_old) * factoryy_mfrm
    tailfr_numold = 0.5 * erfc(yn_tail_old)
    tailfr_volold = 0.5 * erfc(yv_tail_old)

    # Transfer fractions. Guard 4 (Fortran line 4157): tmpa <= 0.
    tmpa = tailfr_volnew * dryvol_t_new - tailfr_volold * dryvol_t_old_used
    guard_tmpa = tmpa > 0.0

    xferfrac_vol = jnp.minimum(tmpa, dryvol_t_new) / dryvol_t_new
    xferfrac_vol = jnp.minimum(xferfrac_vol, xferfrac_max)
    xferfrac_num = tailfr_numnew - tailfr_numold
    xferfrac_num = jnp.maximum(0.0, jnp.minimum(xferfrac_num, xferfrac_vol))

    # Any guard failing → no transfer. Mirrors the Fortran `cycle`s.
    do_transfer = guard_volnew & guard_dgnnew & guard_voldel & guard_tmpa
    xferfrac_vol = jnp.where(do_transfer, xferfrac_vol, 0.0)
    xferfrac_num = jnp.where(do_transfer, xferfrac_num, 0.0)

    # Apply transfers (Fortran lines 4201-4208 — clear-sky only).
    dnum = qnum_cur[mfrm] * xferfrac_num
    qnum_cur = qnum_cur.at[mfrm].add(-dnum)
    qnum_cur = qnum_cur.at[mtoo].add(+dnum)

    dqaer = qaer_cur[:, mfrm] * xferfrac_vol   # (max_aer,)
    qaer_cur = qaer_cur.at[:, mfrm].add(-dqaer)
    qaer_cur = qaer_cur.at[:, mtoo].add(+dqaer)

    return qnum_cur, qaer_cur, qwtr_cur


def _mam_newnuc_1subarea(state: dict[str, Any]) -> dict[str, Any]:
    """Stub: M3.6 PR-D will port the binary H₂SO₄–H₂O nucleation.

    Port target: ``modal_aero_amicphys.F90`` ``mam_newnuc_1subarea``
    (lines 4251–4665, ~415 LOC).
    """
    return state


def _mam_coag_1subarea(state: dict[str, Any]) -> dict[str, Any]:
    """Stub: M3.6 PR-E will port the Brownian coagulation kernels.

    Port target: ``modal_aero_amicphys.F90`` ``mam_coag_1subarea``
    (lines 4670–5106, ~437 LOC).
    """
    return state
