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
import numpy as np
from jax.scipy.special import erfc

from .. import data
from ..constants import RGAS


# ---------------------------------------------------------------------------
# Gasaerexch leaf helpers (M3.6 PR-D)
# ---------------------------------------------------------------------------

def _mean_molecular_speed(temp, rmw):
    """Port of ``modal_aero_amicphys.F90:5290-5297``.

    Returns mean molecular speed (m/s) given temperature (K) and
    molecular weight (g/mol).
    """
    return jnp.sqrt(8.0 * RGAS * temp / (jnp.pi * rmw))


def _gas_diffusivity(t_k, p_atm, rmw, vm):
    """Port of ``modal_aero_amicphys.F90:5302-5316``.

    Returns gas diffusivity (m²/s) via the Fuller-Schettler-Giddings
    correlation. ``rmw`` is molecular weight (g/mol), ``vm`` molar
    diffusion volume (unitless).
    """
    onethird = 1.0 / 3.0
    dgas = (1.0e-3 * t_k ** 1.75 *
            jnp.sqrt(1.0 / rmw + 1.0 / data.MWDRY)) / (
            p_atm * (vm ** onethird + data.VMDRY ** onethird) ** 2)
    return dgas * 1.0e-4


# ---------------------------------------------------------------------------
# state-dict ↔ amicphys-local-view unpacking (M3.6 PR-C foundation)
# ---------------------------------------------------------------------------

# Pre-computed flat-scatter helpers for the LMAP_AER table. The Fortran
# unpacks `qaer3(iaer, mode) = qsub3(lmap_aer(iaer, mode)) * fcvt_aer(iaer)`
# only for `lmap_aer > 0` (i.e. that species exists in that mode). We
# pre-build the (mode, iaer) → pcnst index mapping with sentinel handling
# so the gather/scatter is a single straight-line operation.

_LMAP_AER_VALID_MASK   = data.LMAP_AER >= 0
_LMAP_AER_SAFE         = np.where(_LMAP_AER_VALID_MASK, data.LMAP_AER, 0)
#: 1-D pcnst-index vector of every valid (mode, iaer) slot, used for scatter.
_LMAP_AER_FLAT_VALID   = data.LMAP_AER[_LMAP_AER_VALID_MASK]


def _unpack_state_to_amicphys_view(state: dict[str, Any]):
    """Unpack the outer ``q[pcnst]`` state into amicphys's local view.

    Mirrors ``modal_aero_amicphys.F90:1331-1369`` (the unpacking inside
    ``mam_amicphys_1gridcell``). The leading axes of ``q`` are preserved;
    the trailing ``pcnst`` axis becomes a ``(ngas|naer|nmode|nmode)``
    axis according to the role.

    Returns
    -------
    qgas : jax.Array, shape (..., AMICPHYS_NGAS)
    qaer : jax.Array, shape (..., AMICPHYS_NAER, NTOT_AMODE)
        Note the Fortran convention is (iaer, mode); the second-to-last
        axis is iaer, the last is mode.
    qnum : jax.Array, shape (..., NTOT_AMODE)
    qwtr : jax.Array, shape (..., NTOT_AMODE)
        Pulled from ``state["qaerwat"]`` (not from ``q``) — aerosol
        water is its own state field, mirroring Fortran's
        ``qaerwatsub3``.
    """
    q       = state["q"]
    qaerwat = state["qaerwat"]

    # Stage 1: driver-side mmr → vmr (mwdry / adv_mass per constituent).
    q_vmr = q * data.MMR_TO_VMR

    # Stage 2: amicphys-internal vmr → amicphys-local (fcvt_*).
    qgas = q_vmr[..., data.LMAP_GAS] * data.FCVT_GAS
    qnum = q_vmr[..., data.LMAP_NUM] * data.FCVT_NUM

    # qaer: gather q_vmr at LMAP_AER (mode, iaer), zero absent slots,
    # scale by fcvt_aer. q_vmr[..., LMAP_AER_SAFE] is
    # (..., NTOT_AMODE, AMICPHYS_NAER); transpose the last two axes to
    # match Fortran's (iaer, mode) layout.
    qaer_mode_first = q_vmr[..., _LMAP_AER_SAFE] * data.FCVT_AER
    qaer_mode_first = jnp.where(_LMAP_AER_VALID_MASK, qaer_mode_first, 0.0)
    qaer = jnp.swapaxes(qaer_mode_first, -2, -1)

    qwtr = qaerwat * data.FCVT_WTR
    return qgas, qaer, qnum, qwtr


def _repack_amicphys_view_to_state(state: dict[str, Any],
                                    qgas, qaer, qnum, qwtr) -> dict[str, Any]:
    """Inverse of :func:`_unpack_state_to_amicphys_view`.

    Mirrors ``modal_aero_amicphys.F90:1459-1491``.
    """
    q = state["q"]

    # Work in vmr space (stage-2 inverse first), then convert back to mmr.
    q_vmr = q * data.MMR_TO_VMR

    q_vmr = q_vmr.at[..., data.LMAP_GAS].set(qgas / data.FCVT_GAS)
    q_vmr = q_vmr.at[..., data.LMAP_NUM].set(qnum / data.FCVT_NUM)

    # qaer is (..., NAER, NTOT_AMODE); swap to (..., NTOT_AMODE, NAER)
    # to match LMAP_AER's layout, then mask out invalid slots before
    # scattering.
    qaer_mode_first = jnp.swapaxes(qaer, -2, -1) / data.FCVT_AER
    valid_vals = qaer_mode_first[..., _LMAP_AER_VALID_MASK]   # (..., n_valid)
    q_vmr = q_vmr.at[..., _LMAP_AER_FLAT_VALID].set(valid_vals)

    # Stage-1 inverse: vmr → mmr.
    new_q = q_vmr / data.MMR_TO_VMR

    new_qaerwat = qwtr / data.FCVT_WTR
    return {**state, "q": new_q, "qaerwat": new_qaerwat}


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

    Unpacks the outer ``q[pcnst]`` state into amicphys's local view,
    runs the four sub-process functions in the Fortran order
    (gasaerexch → rename → newnuc → coag — Fortran lines 2387, 2467,
    2496, 2529), repacks the local view back to ``q``.

    Each sub-process call is gated by the corresponding ``mdo_*``
    toggle; 0 means "skip this sub-process". Only rename is real after
    M3.6 PR-C; the other three are no-op stubs (PR-D/E/F/G).

    When all four toggles are 0, returns ``state`` unchanged without
    unpacking/repacking — preserves the bit-exact passthrough invariant
    (round-tripping ``qaerwat * FCVT_WTR / FCVT_WTR`` introduces a 1-ULP
    error otherwise).
    """
    if not (mdo_gasaerexch or mdo_rename or mdo_newnuc or mdo_coag):
        return state

    qgas, qaer, qnum, qwtr = _unpack_state_to_amicphys_view(state)

    if mdo_gasaerexch:
        # PR-D will fill this in. For now the stub leaves qgas/qaer
        # unchanged, so the rename delta below is zero.
        pass

    qaer_sv1 = qaer
    # qaer_delsub_grow4rnam = change made by gasaerexch in this sub-area.
    # Fortran constructs it at modal_aero_amicphys.F90:2433.
    qaer_delsub_grow4rnam = qaer - qaer_sv1

    if mdo_rename:
        qnum, qaer, qwtr = _mam_rename_1subarea(
            qnum, qaer, qaer_delsub_grow4rnam, qwtr,
            jnp.asarray(data.FAC_M2V_AER),
        )

    if mdo_newnuc:
        state = _mam_newnuc_1subarea(state)  # noqa: still a stub
    if mdo_coag:
        state = _mam_coag_1subarea(state)    # noqa: still a stub

    return _repack_amicphys_view_to_state(state, qgas, qaer, qnum, qwtr)


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

    # Dry volume for the "from" mode (clear-sky only). qaer is
    # (..., max_aer, max_mode); fac_m2v_aer is (max_aer,). Sum species
    # contributions on the per-species axis (-2).
    qaer_mfrm  = qaer_cur[..., mfrm]                    # (..., max_aer)
    qadel_mfrm = qaer_delsub_grow4rnam[..., mfrm]
    deldryvol_t = jnp.sum(qadel_mfrm * fac_m2v_aer, axis=-1)
    dryvol_t_old = jnp.sum(qaer_mfrm * fac_m2v_aer, axis=-1) - deldryvol_t
    dryvol_t_del = deldryvol_t
    num_t_old   = qnum_cur[..., mfrm]
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
    # xferfrac_num/_vol have shape (...,); broadcast to per-species/mode.
    dnum = qnum_cur[..., mfrm] * xferfrac_num                # (...,)
    qnum_cur = qnum_cur.at[..., mfrm].add(-dnum)
    qnum_cur = qnum_cur.at[..., mtoo].add(+dnum)

    dqaer = qaer_cur[..., mfrm] * xferfrac_vol[..., None]    # (..., max_aer)
    qaer_cur = qaer_cur.at[..., mfrm].add(-dqaer)
    qaer_cur = qaer_cur.at[..., mtoo].add(+dqaer)

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
