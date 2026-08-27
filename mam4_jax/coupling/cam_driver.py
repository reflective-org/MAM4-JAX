"""CAM's process coupling — plan 024 PR G / plan 025 §7.

CAM drives MAM microphysics as a SEQUENTIAL chain on grid-cell means
(``aero_model.F90:1202-1247``): ``modal_aero_gasaerexch_sub`` (which calls
rename internally) → ``modal_aero_newnuc_sub`` → ``modal_aero_coag_sub`` —
no sub-areas, no amicphys. This module ports that chain for the
**SO4-only box scope** (owner 2026-08-20: coagulation, condensation,
nucleation; no deposition, no transport, no organics evaporation).

Everything here operates on the GAS-WINDOW view: arrays of shape
``(..., gas_pcnst)`` holding volume mixing ratios (mol/mol for mass,
#/kmol-air for number) — exactly what CAM passes (``vmr``,
aero_model.F90:1202) and what the sibling-repo box driver hands to
``mam_microphysics_cam``. The mmr↔vmr boundary conversion belongs to the
driver-assembly step (plan 025 G4), not here.

Topology-threading: this module is the first real consumer of the
``Topology`` axis. Every mode/species table is resolved from a
``Topology`` instance plus the generated ``cam_params`` layer through
:func:`_cam_tables` (cached per topology name, plain numpy — safe to read
at trace time because the TOPOLOGY IS AN ARGUMENT, never a module-global
read inside a kernel; see core/topology.py's staleness guard).

SO4-only reductions, each exact for the scope rather than approximate:

* NH3/MSA: absent from the chemistry mechanism used (`do_nh4g`/`do_msag`
  false paths).
* SOA: the mechanism HAS soa/SOAG tracers, but the SO4-only scenarios hold
  them identically zero, and zero gas + zero aerosol is a fixed point of
  ``modal_aero_soaexch`` — assumption **A13**, verified empirically by the
  capture-parity tests (the capture runs the FULL Fortran subroutine).
* Cloud-borne (qqcw) tracers: identically zero (``cldfr = 0`` box), so the
  qqcw half of rename is a no-op and is not carried.
* Diagnostics (``qsrflx``, ``dotend``, outfld) are not carried; tendencies
  are applied densely (adding an exact 0.0 tendency is the identity).

Faithfulness notes:

* ``1 - exp(-x)`` is written ``-expm1(-x)`` (repo standard, plan 026 /
  ADR-019; worth ~1e-11 relative near thresholds, nothing elsewhere).
* The legacy primary-carbon aging block INSIDE gasaerexch
  (modal_aero_gasaerexch.F90:719-806) ages at
  ``n_so4_monolayers_pcage = 8.0`` — the gasaerexch module's own
  ``parameter`` (:37). This is a different code line from amicphys'
  phys_control-fed 3.0 and is ported as CAM has it, NOT shared with the
  E3SM path's ``configure_pcarbon_aging``.
* CAM's ``gas_aer_uptkrates`` is ported as its own function — it is a
  THIRD variant (fixed ``beta = 2``, hardcoded ac = 0.65 literals,
  truncated ``tworootpi``/``root2``), not the amicphys quadrature already
  in ``coupling/amicphys.py``. See plan 025 §7.
"""
from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

from mam4_jax.core.cam_params import CAM_PARAMS
from mam4_jax.core.topology import Topology, get_topology
from mam4_jax.physics.strat_sulfate import h2so4_reversible_uptake

__all__ = [
    "gas_aer_uptkrates_cam",
    "modal_aero_coag_cam",
    "modal_aero_gasaerexch_cam",
    "modal_aero_newnuc_cam",
]

# CAM physconst constants, spelled EXACTLY as shr_const_mod computes them:
# SHR_CONST_RGAS = AVOGAD * BOLTZ = 8314.467591 J/K/kmol — NOT the rounded
# 8.31446e3 (that shortcut is 9.1e-7 relative off, which the implicit
# coagulation number solves amplify to ~2e-6; found by capture parity).
_RGAS_UNIV = 6.02214e26 * 1.38065e-23   # J/K/kmol (shr_const_rgas)
_MWDAIR = 28.966                        # kg/kmol  (shr_const_mwdair)
_RAIR = _RGAS_UNIV / _MWDAIR

#: gasaerexch's own aging threshold (modal_aero_gasaerexch.F90:37-44):
#: a Fortran ``parameter`` — the legacy code line's value, NOT the
#: amicphys/phys_control 3.0. Deliberately not configurable here.
_N_SO4_MONOLAYERS_PCAGE = 8.0
_DR_SO4_MONOLAYERS_PCAGE = _N_SO4_MONOLAYERS_PCAGE * 4.76e-10


class _CamTables:
    """Static per-topology tables for the CAM chain, gas-window indexed.

    Plain numpy / Python ints, built once per topology (cached). All
    tracer indices are 0-based GAS-WINDOW indices (pcnst index − loffset);
    ``-1`` marks an absent slot.
    """

    def __init__(self, topology: Topology):
        p = CAM_PARAMS[topology.name]
        names = p["cnst_names"]
        off = p["loffset"]
        nm = topology.nmodes
        self.topology = topology
        self.gas_pcnst = p["gas_pcnst"]
        self.loffset = off
        self.mwdry = p["mwdry"]
        self.adv_mass = np.asarray(p["adv_mass"])
        self.cnst_names = names

        self.l_h2so4 = names.index("H2SO4")
        self.l_so2 = names.index("SO2") if "SO2" in names else -1

        # Mode-indexed tables (gas-window indices).
        self.num_ptr = np.asarray(
            [topology.numptr_amode[m] - off for m in range(nm)])
        self.lmass = [
            [topology.lmassptr_amode[m][s] - off
             for s in range(topology.nspec_amode[m])]
            for m in range(nm)
        ]
        # fac_m2v per (mode, slot): specmw/specdens, (m3-AP/kmol-AP).
        mw = p["specmw_amode"]
        dens = topology.specdens_amode
        self.fac_m2v = [
            [mw[t] / dens[t] for t in topology.lspectype_amode[m][
                :topology.nspec_amode[m]]]
            for m in range(nm)
        ]

        # so4 per mode: window index of the mode's so4 tracer, else -1.
        so4_type = topology.specname_amode.index("so4")
        self.lptr_so4 = np.full(nm, -1, dtype=int)
        for m in range(nm):
            for s in range(topology.nspec_amode[m]):
                if topology.lspectype_amode[m][s] == so4_type:
                    self.lptr_so4[m] = topology.lmassptr_amode[m][s] - off
        self.fac_m2v_so4 = mw[so4_type] / dens[so4_type]

        # ido_so4a (gasaerexch_init:303-345): 1 = mode carries so4;
        # 2 = the aging source mode (pcarbon); 0 = inactive.
        self.mode_pcarbon = (topology.mode_index("primary_carbon")
                             if topology.has_mode("primary_carbon") else -1)
        self.mode_accum = topology.mode_index("accum")
        self.mode_aitken = topology.mode_index("aitken")
        self.ido_so4a = np.where(self.lptr_so4 >= 0, 1, 0)
        if self.mode_pcarbon >= 0 and self.lptr_so4[self.mode_accum] >= 0:
            self.ido_so4a[self.mode_pcarbon] = 2

        # Primary-carbon aging pair list (gasaerexch_init: number first,
        # then each pcarbon species matched to the accum species of the
        # same name prefix; aerosol water skipped). (from, to) window
        # indices; to = -1 when no counterpart exists.
        self.pcage_pairs: list[tuple[int, int]] = []
        if self.mode_pcarbon >= 0 and self.ido_so4a[self.mode_pcarbon] == 2:
            mf, mt = self.mode_pcarbon, self.mode_accum
            self.pcage_pairs.append(
                (int(self.num_ptr[mf]), int(self.num_ptr[mt])))
            for s in range(topology.nspec_amode[mf]):
                lf = topology.lmassptr_amode[mf][s] - off
                prefix = names[lf].rsplit("_", 1)[0]
                lt = -1
                for s2 in range(topology.nspec_amode[mt]):
                    cand = topology.lmassptr_amode[mt][s2] - off
                    if names[cand].rsplit("_", 1)[0] == prefix:
                        lt = cand
                        break
                self.pcage_pairs.append((lf, lt))
        alnsg = np.log(np.asarray(topology.sigmag_amode))
        self.alnsg = alnsg
        self.sigmag = np.asarray(topology.sigmag_amode)
        if self.mode_pcarbon >= 0:
            self.fac_volsfc_pcarbon = float(
                np.exp(2.5 * alnsg[self.mode_pcarbon] ** 2))
            self.fac_m2v_pcarbon = np.asarray(
                self.fac_m2v[self.mode_pcarbon])

        # Rename pair (A1, acc_crs off): aitken -> accum only
        # (rename_init). Species list: NUMBER FIRST (iq=1 moves by
        # xferfrac_num), then each aitken species matched to accum by
        # name prefix (all four match in both topologies).
        mf, mt = self.mode_aitken, self.mode_accum
        self.rename_pairs: list[tuple[int, int]] = [
            (int(self.num_ptr[mf]), int(self.num_ptr[mt]))]
        for s in range(topology.nspec_amode[mf]):
            lf = topology.lmassptr_amode[mf][s] - off
            prefix = names[lf].rsplit("_", 1)[0]
            lt = -1
            for s2 in range(topology.nspec_amode[mt]):
                cand = topology.lmassptr_amode[mt][s2] - off
                if names[cand].rsplit("_", 1)[0] == prefix:
                    lt = cand
                    break
            self.rename_pairs.append((lf, lt))
        # Rename-from dry-volume ingredients: (window index, fac_m2v) per
        # aitken slot.
        self.rename_frm_slots = [
            (topology.lmassptr_amode[mf][s] - off, self.fac_m2v[mf][s])
            for s in range(topology.nspec_amode[mf])
        ]

        # Coagulation pair species lists (coag_init:800-870, name-matched
        # against the EFFECTIVE destination = accum for all three pairs).
        # Pair 3 (ait->pca, effective accum) carries the same (from, to)
        # species list as pair 1, so only two lists are needed; what pair
        # 3 adds is fac_m2v_aitage (init:906-955): so4 at face value, soa
        # scaled by its equivalent-so4 hygroscopicity factor
        # (spechygro_soa / spechygro_so4, gasaerexch_init:1861), every
        # other species contributing ZERO shell volume.
        def _match(mfrm, mtoo):
            pairs = []
            for s_ in range(topology.nspec_amode[mfrm]):
                lf_ = topology.lmassptr_amode[mfrm][s_] - off
                pre = names[lf_].rsplit("_", 1)[0]
                lt_ = -1
                for s2_ in range(topology.nspec_amode[mtoo]):
                    cand_ = topology.lmassptr_amode[mtoo][s2_] - off
                    if names[cand_].rsplit("_", 1)[0] == pre:
                        lt_ = cand_
                        break
                pairs.append((lf_, lt_))
            return pairs

        self.coag_ait_pairs = _match(self.mode_aitken, self.mode_accum)
        self.coag_pca_pairs = _match(self.mode_pcarbon, self.mode_accum) \
            if self.mode_pcarbon >= 0 else []
        so4_t = topology.specname_amode.index("so4")
        soa_t = (topology.specname_amode.index("soa")
                 if "soa" in topology.specname_amode else -1)
        hygro = topology.spechygro_amode
        self.fac_m2v_aitage = []
        for s_ in range(topology.nspec_amode[self.mode_aitken]):
            t_ = topology.lspectype_amode[self.mode_aitken][s_]
            if t_ == so4_t:
                self.fac_m2v_aitage.append(self.fac_m2v[self.mode_aitken][s_])
            elif t_ == soa_t:
                self.fac_m2v_aitage.append(
                    (hygro[soa_t] / hygro[so4_t])
                    * self.fac_m2v[self.mode_aitken][s_])
            else:
                self.fac_m2v_aitage.append(0.0)


@functools.lru_cache(maxsize=8)
def _cam_tables_by_name(name: str) -> _CamTables:
    from mam4_jax.core import topology as topo_mod
    try:
        return _CamTables(topo_mod._REGISTRY[name])  # noqa: SLF001
    except KeyError:
        raise KeyError(f"no registered topology named {name!r}") from None


def _cam_tables(topology: Topology | None) -> _CamTables:
    if topology is None:
        topology = get_topology()
    return _cam_tables_by_name(topology.name)


# ---------------------------------------------------------------------------
# gas_aer_uptkrates — CAM's variant (modal_aero_gasaerexch.F90:953-1086)
# ---------------------------------------------------------------------------

# CAM's own truncated literals — ported verbatim, NOT replaced with exact
# sqrt(pi)/sqrt(2) (the amicphys port uses exact values; this variant's
# reference is CAM, and normalising them changes answers at ~1e-8).
_TWOROOTPI_CAM = 3.5449077
_ROOT2_CAM = 1.4142135
_BETA_CAM = 2.0
_XGHQ_CAM = (0.70710678, -0.70710678)
_WGHQ_CAM = (0.88622693, 0.88622693)


def gas_aer_uptkrates_cam(qnum, t, pmid, dgncur_awet, sigmag):
    """H2SO4 gas-to-aerosol uptake rate per mode (1/s) — CAM's variant.

    Port of ``gas_aer_uptkrates`` (modal_aero_gasaerexch.F90:953-1086):
    two-point Gauss-Hermite quadrature of ``2*pi*D*Dp*F(Kn, ac)`` over the
    log-normal, with **fixed** ``beta = 2`` (the amicphys variant computes
    a Knudsen-dependent beta), hardcoded ac = 0.65 Fuchs-Sutugin literals
    (``0.4875 = 0.75*0.65``, ``1.184 = 1 + 0.283*0.65``), CAM's own
    ``gasdiffus = 0.557e-4 * T**1.75 / p`` and
    ``gasspeed = 14.70 * sqrt(T)``, and the truncated
    ``tworootpi``/``root2``/quadrature literals above.

    Parameters
    ----------
    qnum : (..., nmodes) — mode number mixing ratios (#/kmol-air).
    t, pmid : (...,) — temperature (K), pressure (Pa).
    dgncur_awet : (..., nmodes) — wet number-median diameters (m).
    sigmag : (nmodes,) — geometric standard deviations (static).

    Returns ``uptkrate`` with shape (..., nmodes).
    """
    t = jnp.asarray(t)[..., None]
    pmid = jnp.asarray(pmid)[..., None]
    rhoair = pmid / (_RAIR * t)                      # kg/m3
    aircon = rhoair / _MWDAIR                        # kmol-air/m3
    num_a = qnum * aircon                            # #/m3

    gasdiffus = 0.557e-4 * t ** 1.75 / pmid          # m2/s
    gasspeed = 1.470e1 * jnp.sqrt(t)                 # m/s
    freepathx2 = 6.0 * gasdiffus / gasspeed          # m

    lnsg = jnp.log(jnp.asarray(sigmag))              # (nmodes,)
    lndpgn = jnp.log(dgncur_awet)
    const = _TWOROOTPI_CAM * num_a * jnp.exp(
        _BETA_CAM * lndpgn + 0.5 * (_BETA_CAM * lnsg) ** 2)

    sumghq = jnp.zeros_like(dgncur_awet)
    for xq, wq in zip(_XGHQ_CAM, _WGHQ_CAM):
        lndp = lndpgn + _BETA_CAM * lnsg ** 2 + _ROOT2_CAM * lnsg * xq
        dp = jnp.exp(lndp)
        knudsen = freepathx2 / dp
        fuchs_sutugin = (0.4875 * (1.0 + knudsen)) / (
            knudsen * (1.184 + knudsen) + 0.4875)
        sumghq = sumghq + wq * dp * fuchs_sutugin / dp ** _BETA_CAM

    return const * gasdiffus * sumghq


# ---------------------------------------------------------------------------
# rename A1 — modal_aero_rename_no_acc_crs_sub (modal_aero_rename.F90:243-624)
# ---------------------------------------------------------------------------

_FRELAX = 27.0
_DRYVOL_SMALLEST = 1.0e-25


def _rename_no_acc_crs_cam(q, dqdt, deltat, tables: _CamTables):
    """CAM's default rename (A1), interstitial half, single pair.

    Adds the aitken → accum renaming tendencies onto ``dqdt`` and returns
    it. ``q`` is the PRE-growth state; the incoming ``dqdt`` carries the
    continuous-growth (condensation + aging) tendencies, exactly as the
    Fortran receives them. ``dqdt_other`` and the cloud-borne half are
    identically zero in the box scope and not carried.

    The Fortran's per-(i,k) early ``cycle``s become ``where`` masks: a
    cell that fails a guard contributes an exact 0.0 tendency.
    """
    topo = tables.topology
    mf, mt = tables.mode_aitken, tables.mode_accum
    alnsg_f = float(tables.alnsg[mf])
    alnsg_t = float(tables.alnsg[mt])
    dgnum_f = topo.dgnum_amode[mf]

    deltatinv = 1.0 / (deltat * (1.0 + 1.0e-15))
    xferfrac_max = 1.0 - 10.0 * float(jnp.finfo(jnp.float64).eps)

    factoraa = (np.pi / 6.0) * np.exp(4.5 * alnsg_f ** 2)
    factoryy = np.sqrt(0.5) / alnsg_f
    v2nlorlx = topo.voltonumblo_amode[mf] * _FRELAX
    v2nhirlx = topo.voltonumbhi_amode[mf] / _FRELAX
    dum3alnsg2 = 3.0 * alnsg_f ** 2
    dp_cut = np.sqrt(
        topo.dgnum_amode[mf] * np.exp(1.5 * alnsg_f ** 2)
        * topo.dgnum_amode[mt] * np.exp(1.5 * alnsg_t ** 2))
    lndp_cut = np.log(dp_cut)
    dp_belowcut = 0.99 * dp_cut

    # Dry volume of the "from" mode and its growth increment, from ALL of
    # the mode's species (dqdt_other = 0 in scope).
    dryvol_t_old = jnp.zeros(q.shape[:-1], dtype=q.dtype)
    dryvol_t_del = jnp.zeros(q.shape[:-1], dtype=q.dtype)
    for lw, m2v in tables.rename_frm_slots:
        dryvol_t_old = dryvol_t_old + m2v * jnp.maximum(0.0, q[..., lw])
        dryvol_t_del = dryvol_t_del + (m2v * deltat) * dqdt[..., lw]
    dryvol_t_new = dryvol_t_old + dryvol_t_del
    dryvol_t_oldbnd = jnp.maximum(dryvol_t_old, _DRYVOL_SMALLEST)

    ok = dryvol_t_new > _DRYVOL_SMALLEST
    ok = ok & (dryvol_t_del > 1.0e-6 * dryvol_t_oldbnd)

    num_t_old = jnp.maximum(0.0, q[..., tables.num_ptr[mf]])
    num_t_oldbnd = jnp.minimum(dryvol_t_oldbnd * v2nlorlx, num_t_old)
    num_t_oldbnd = jnp.maximum(dryvol_t_oldbnd * v2nhirlx, num_t_oldbnd)

    dgn_t_new = (dryvol_t_new / (num_t_oldbnd * factoraa)) ** (1.0 / 3.0)
    ok = ok & (dgn_t_new > dgnum_f)

    from jax.scipy.special import erfc

    lndgn_new = jnp.log(dgn_t_new)
    lndgv_new = lndgn_new + dum3alnsg2
    tailfr_numnew = 0.5 * erfc((lndp_cut - lndgn_new) * factoryy)
    tailfr_volnew = 0.5 * erfc((lndp_cut - lndgv_new) * factoryy)

    dgn_t_old = (dryvol_t_oldbnd / (num_t_oldbnd * factoraa)) ** (1.0 / 3.0)
    dgn_t_old = jnp.where(dgn_t_new >= dp_cut,
                          jnp.minimum(dgn_t_old, dp_belowcut), dgn_t_old)
    lndgn_old = jnp.log(dgn_t_old)
    lndgv_old = lndgn_old + dum3alnsg2
    tailfr_numold = 0.5 * erfc((lndp_cut - lndgn_old) * factoryy)
    tailfr_volold = 0.5 * erfc((lndp_cut - lndgv_old) * factoryy)

    dum = tailfr_volnew * dryvol_t_new - tailfr_volold * dryvol_t_old
    ok = ok & (dum > 0.0)

    safe_new = jnp.where(ok, dryvol_t_new, 1.0)
    xferfrac_vol = jnp.minimum(dum, safe_new) / safe_new
    xferfrac_vol = jnp.minimum(xferfrac_vol, xferfrac_max)
    xferfrac_num = tailfr_numnew - tailfr_numold
    xferfrac_num = jnp.maximum(0.0, jnp.minimum(xferfrac_num, xferfrac_vol))
    xferfrac_vol = jnp.where(ok, xferfrac_vol, 0.0)
    xferfrac_num = jnp.where(ok, xferfrac_num, 0.0)

    for iq, (lf, lt) in enumerate(tables.rename_pairs):
        xfercoef = (xferfrac_num if iq == 0 else xferfrac_vol) * deltatinv
        xfertend = xfercoef * jnp.maximum(
            0.0, q[..., lf] + dqdt[..., lf] * deltat)
        dqdt = dqdt.at[..., lf].add(-xfertend)
        if lt >= 0:
            dqdt = dqdt.at[..., lt].add(xfertend)
    return dqdt


# ---------------------------------------------------------------------------
# gasaerexch — modal_aero_gasaerexch_sub, SO4-only
# ---------------------------------------------------------------------------

def modal_aero_gasaerexch_cam(q, t, pmid, deltat, dgncur_a, dgncur_awet,
                              *, topology=None, sulfeq=None):
    """One gasaerexch call on the gas window — CAM's SO4-only chain.

    Port of ``modal_aero_gasaerexch_sub`` (modal_aero_gasaerexch.F90),
    reduced exactly to the SO4-only box scope (module docstring):

    1. per-mode uptake rates (:func:`gas_aer_uptkrates_cam`);
    2. H2SO4 condensation — irreversible ``fgain``-split when ``sulfeq``
       is None (tropospheric), or the reversible sulfeq-limited solve
       (stratospheric; ``k <= troplev`` everywhere in the box);
    3. the legacy primary-carbon aging block at 8.0 monolayers
       (:719-806) — this-step condensed shell only;
    4. rename A1 (aitken → accum), fed the accumulated ``dqdt``;
    5. ``q += dqdt * deltat``.

    Parameters
    ----------
    q : (..., gas_pcnst) — VOLUME mixing ratios (mol/mol, #/kmol-air).
    t, pmid : (...,); deltat : scalar (s).
    dgncur_a, dgncur_awet : (..., nmodes) — dry / wet mode diameters (m).
    topology : static ``Topology`` (None → active topology).
    sulfeq : None for the tropospheric path, else (..., nmodes)
        equilibrium H2SO4 over each mode (mol/mol) from
        :func:`mam4_jax.physics.strat_sulfate.calc_h2so4_equilib_mixrat`.

    Returns the updated ``q``.
    """
    tb = _cam_tables(topology)
    deltatxx = deltat * (1.0 + 1.0e-15)

    qnum = q[..., tb.num_ptr]                                # (..., nmodes)
    uptkrate = gas_aer_uptkrates_cam(qnum, t, pmid, dgncur_awet, tb.sigmag)

    active = jnp.asarray(tb.ido_so4a > 0)
    uptk = jnp.where(active, uptkrate, 0.0)
    sum_uprt_so4 = jnp.sum(uptk, axis=-1)

    qgas = q[..., tb.l_h2so4]

    if sulfeq is None:
        # Tropospheric: irreversible uptake, fgain split (:492-575).
        safe_sum = jnp.where(sum_uprt_so4 > 0.0, sum_uprt_so4, 1.0)
        fgain = jnp.where(sum_uprt_so4[..., None] > 0.0,
                          uptk / safe_sum[..., None], 0.0)
        avg_uprt = -jnp.expm1(-deltatxx * sum_uprt_so4) / deltatxx
        sum_dqdt_so4 = qgas * avg_uprt
        dqdt_so4 = fgain * sum_dqdt_so4[..., None]           # (..., nmodes)
        sum_dqdt_out = sum_dqdt_so4
    else:
        # Stratospheric: reversible, sulfeq-limited (:523-566).
        qaer_so4 = jnp.stack(
            [q[..., tb.lptr_so4[m]] if tb.lptr_so4[m] >= 0
             else jnp.zeros_like(qgas)
             for m in range(tb.topology.nmodes)], axis=-1)
        dqdt_so4, sum_dqdt_out = h2so4_reversible_uptake(
            qgas, qaer_so4, uptkrate, sulfeq, deltat,
            jnp.asarray(tb.ido_so4a))

    # Assemble dqdt on the window: so4 slots (ido == 1) and the gas.
    dqdt = jnp.zeros_like(q)
    for m in range(tb.topology.nmodes):
        if tb.ido_so4a[m] == 1:
            dqdt = dqdt.at[..., tb.lptr_so4[m]].set(dqdt_so4[..., m])
    dqdt = dqdt.at[..., tb.l_h2so4].set(-sum_dqdt_out)

    # Legacy primary-carbon aging (:719-806). SO4-only: the shell is this
    # step's condensed so4 on the pcarbon mode; nh4/soa terms absent.
    if tb.mode_pcarbon >= 0 and tb.ido_so4a[tb.mode_pcarbon] == 2:
        mp = tb.mode_pcarbon
        vol_shell = deltat * dqdt_so4[..., mp] * tb.fac_m2v_so4
        vol_core = jnp.zeros_like(vol_shell)
        for s, lw in enumerate(tb.lmass[mp]):
            vol_core = vol_core + q[..., lw] * tb.fac_m2v_pcarbon[s]
        tmp1 = vol_shell * dgncur_a[..., mp] * tb.fac_volsfc_pcarbon
        tmp2 = jnp.maximum(6.0 * _DR_SO4_MONOLAYERS_PCAGE * vol_core, 0.0)
        xferfrac_max = 1.0 - 10.0 * float(jnp.finfo(jnp.float64).eps)
        saturated = tmp1 >= tmp2
        safe_tmp2 = jnp.where(saturated, 1.0, tmp2)
        xferfrac = jnp.where(saturated, xferfrac_max,
                             jnp.minimum(tmp1 / safe_tmp2, xferfrac_max))
        fire = xferfrac > 0.0
        rate = jnp.where(fire, xferfrac, 0.0) / deltat
        for lf, lt in tb.pcage_pairs:
            xferrate = rate * q[..., lf]
            dqdt = dqdt.at[..., lf].add(-xferrate)
            if lt >= 0:
                dqdt = dqdt.at[..., lt].add(xferrate)
        # Condensed-on-pcarbon so4 lands on accum's so4 slot — but ONLY
        # when the aging fires (:770): with xferfrac == 0 CAM drops it.
        if tb.ido_so4a[tb.mode_accum] > 0:
            dqdt = dqdt.at[..., tb.lptr_so4[tb.mode_accum]].add(
                jnp.where(fire, dqdt_so4[..., mp], 0.0))

    # Rename A1 (aitken -> accum), then apply everything (:806-838).
    dqdt = _rename_no_acc_crs_cam(q, dqdt, deltat, tb)
    return q + dqdt * deltat


# ---------------------------------------------------------------------------
# newnuc — modal_aero_newnuc_sub (modal_aero_newnuc.F90:59-520)
# ---------------------------------------------------------------------------

#: skip nucleation entirely below this H2SO4 vmr (newnuc.F90:30).
_QH2SO4_CUTOFF = 4.0e-16


def modal_aero_newnuc_cam(q, t, pmid, deltat, qv, zm, pblh,
                          del_h2so4_gasprod, del_h2so4_aeruptk,
                          *, topology=None):
    """One newnuc call on the gas window — CAM's wrapper, SO4-only.

    Port of ``modal_aero_newnuc_sub`` around the already-ported
    ``mer07_veh02_nuc_mosaic_1box`` dispatcher (the leaf
    parameterizations are 0-diff between CAM and E3SM; the WRAPPER is
    CAM-specific). What the wrapper owns:

    * the reconstruction of the step-average H2SO4 from
      ``del_h2so4_gasprod`` (gas-phase production over the step) and
      ``del_h2so4_aeruptk`` (loss to condensation over the step, <= 0):
      ``tmp_q2 = q3 + max(0, -aeruptk)`` is the pre-uptake gas,
      ``tmpb = log(q2/q3)`` (clamped to 20, with q3 clamped up to
      ``q2*exp(-20)``) gives the uptake rate, and the average follows the
      production-during-decay closed form (:262-292);
    * relative humidity through CAM's TABLE ``qsat``
      (:mod:`mam4_jax.physics.cam_saturation`) — mixed-phase, NOT the
      direct over-water formula;
    * the two H2SO4 cutoffs (4e-16 on current AND average), the
      100 #/kmol-air/s rate floor, and the grown-particle size
      constraints against the aitken lo/hi single-particle masses;
    * tendencies onto (H2SO4, so4_aitken, num_aitken). ``cld = 0`` in
      the box scope, so the ``(1-cldx)`` weights are 1 and the
      ``cld >= 0.99`` skip never fires; NH3 is absent.

    Every Fortran ``cycle`` becomes a mask; dispatcher inputs are
    clamped to benign values on masked cells (double-where) so no dead
    branch poisons reverse-mode.

    Returns the updated ``q``.
    """
    from mam4_jax.physics.cam_saturation import qsat_cam
    from mam4_jax.physics.newnuc import mer07_veh02_nuc_mosaic_1box

    tb = _cam_tables(topology)
    topo = tb.topology
    mait = tb.mode_aitken
    lnum = int(tb.num_ptr[mait])
    lso4 = int(tb.lptr_so4[mait])
    mw_so4 = CAM_PARAMS[topo.name]["specmw_amode"][
        topo.specname_amode.index("so4")]
    dens_so4 = topo.specdens_amode[topo.specname_amode.index("so4")]

    # Grown-particle dry-diameter window and single-particle masses.
    dplom = float(np.exp(0.67 * np.log(topo.dgnumlo_amode[mait])
                         + 0.33 * np.log(topo.dgnum_amode[mait])))
    dphim = topo.dgnumhi_amode[mait]
    mass1p_aitlo = dens_so4 * np.pi / 6.0 * dplom ** 3
    mass1p_aithi = dens_so4 * np.pi / 6.0 * dphim ** 3

    qh2so4_cur = q[..., tb.l_h2so4]
    go = qh2so4_cur > _QH2SO4_CUTOFF

    # Step-average H2SO4 reconstruction (:262-292).
    tmpa = jnp.maximum(0.0, del_h2so4_gasprod)
    tmp_q3 = qh2so4_cur
    tmp_q2 = tmp_q3 + jnp.maximum(0.0, -del_h2so4_aeruptk)
    tmpc = tmp_q2 * np.exp(-20.0)
    no_uptake = tmp_q2 <= tmp_q3
    clamped = (~no_uptake) & (tmp_q3 <= tmpc)
    q3_eff = jnp.where(clamped, tmpc, tmp_q3)
    safe_ratio = jnp.where(no_uptake | clamped, 1.0, tmp_q2 /
                           jnp.where(q3_eff > 0.0, q3_eff, 1.0))
    tmpb = jnp.where(no_uptake, 0.0,
                     jnp.where(clamped, 20.0, jnp.log(safe_ratio)))
    tmp_uptkrate = tmpb / deltat

    small = tmpb <= 0.1
    safe_tmpb = jnp.where(small, 1.0, tmpb)
    tmpc2 = tmpa / safe_tmpb
    avg_big = (q3_eff - tmpc2) * (jnp.expm1(safe_tmpb) / safe_tmpb) + tmpc2
    avg_small = q3_eff * (1.0 + 0.5 * tmpb) - 0.5 * tmpa
    qh2so4_avg = jnp.where(small, avg_small, avg_big)
    go = go & (qh2so4_avg > _QH2SO4_CUTOFF)

    # RH via the TABLE qsat; cld = 0 so grid-average == clear-sky.
    _es, qs = qsat_cam(t, pmid)
    qvswtr = jnp.maximum(qs, 1.0e-20)
    relhum = jnp.clip(qv / qvswtr, 0.0, 1.0)
    relhumnn = jnp.clip(relhum, 0.01, 0.99)

    # Dispatcher, on benign inputs where masked.
    safe_cur = jnp.where(go, qh2so4_cur, 1.0e-14)
    safe_avg = jnp.where(go, qh2so4_avg, 1.0e-14)
    (_isize, qnuma_del, qso4a_del, _qnh4a_del, _qh2so4_del, _qnh3_del,
     _dens, _dncl) = mer07_veh02_nuc_mosaic_1box(
        dtnuc=deltat, temp=t, rh=relhumnn, press=pmid, zm=zm, pblh=pblh,
        qh2so4_cur=safe_cur, qh2so4_avg=safe_avg,
        h2so4_uptkrate=tmp_uptkrate,
        dplom_sect=dplom, dphim_sect=dphim,
        newnuc_method_flagaa=11, mw_so4a_host=mw_so4)

    # (#/mol-air) -> (#/kmol-air); rates; SO4-only mass fraction = 1.
    qnuma_del = jnp.where(go, qnuma_del, 0.0) * 1.0e3
    qso4a_del = jnp.where(go, qso4a_del, 0.0)
    dndt = qnuma_del / deltat
    tmpa_m = qso4a_del * mw_so4
    tmp_frso4 = jnp.maximum(tmpa_m, 1.0e-35) / jnp.maximum(tmpa_m, 1.0e-35)
    dmdt = jnp.maximum(0.0, tmpa_m / deltat)

    # Rate floor (:404) then size constraints (:415-428).
    live = dndt >= 1.0e2
    dndt = jnp.where(live, dndt, 0.0)
    dmdt = jnp.where(live, dmdt, 0.0)
    safe_dndt = jnp.where(live, dndt, 1.0)
    mass1p = jnp.where(live, dmdt / safe_dndt, mass1p_aitlo)
    dndt = jnp.where(mass1p < mass1p_aitlo, dmdt / mass1p_aitlo, dndt)
    dmdt = jnp.where(mass1p > mass1p_aithi, dndt * mass1p_aithi, dmdt)

    dso4dt = dmdt * tmp_frso4 / mw_so4
    q = q.at[..., tb.l_h2so4].add(-dso4dt * deltat)
    q = q.at[..., lso4].add(dso4dt * deltat)
    q = q.at[..., lnum].add(dndt * deltat)
    return q


# ---------------------------------------------------------------------------
# coag — modal_aero_coag_sub, pair_option_acoag == 3
# (modal_aero_coag.F90:73-709; init tables :714-990)
# ---------------------------------------------------------------------------

def _coag_number_new(tmpn, tmpa, tmpb):
    """The three-branch closed form for a mode's number after ``deltat``
    of intermodal (rate ``tmpa/deltat``) plus self (coefficient
    ``tmpb/deltat``) coagulation loss (modal_aero_coag.F90:459-471,
    faithful branch thresholds |tmpc| < 0.01, |tmpa| < 0.001).

    Every branch is evaluated on guarded operands (double-where) so no
    dead branch poisons reverse-mode.
    """
    tmpc = tmpa + tmpb * tmpn
    b_small_c = jnp.abs(tmpc) < 0.01
    b_small_a = jnp.abs(tmpa) < 0.001

    n_small_c = tmpn * jnp.exp(-tmpc)
    n_small_a = jnp.exp(-tmpa) * tmpn / (1.0 + tmpb * tmpn)

    safe_tmpc = jnp.where(jnp.abs(tmpc) > 0.0, tmpc, 1.0)
    tmpf = tmpb * tmpn / safe_tmpc
    tmpg = jnp.exp(-tmpa)
    den = 1.0 - tmpg * tmpf
    safe_den = jnp.where(jnp.abs(den) > 0.0, den, 1.0)
    tmph = tmpg * (1.0 - tmpf) / safe_den
    n_general = tmpn * jnp.maximum(0.0, jnp.minimum(1.0, tmph))

    return jnp.where(b_small_c, n_small_c,
                     jnp.where(b_small_a, n_small_a, n_general))


def modal_aero_coag_cam(q, t, pmid, deltat, dgncur_a, dgncur_awet,
                        wetdens_a, *, topology=None):
    """One coag call on the gas window — CAM's ``pair_option_acoag = 3``.

    Port of ``modal_aero_coag_sub``: three pairs (aitken→accum,
    pcarbon→accum, aitken→pcarbon-effective-accum) with Whitby fast
    coefficients from the byte-identical (and already-ported)
    ``getcoags_wrapper_f``, then:

    1. number: accum self-coag (``n/(1+Δt·βjj0·n)``), then pcarbon and
       aitken via the three-branch closed form
       (:func:`_coag_number_new`), each consuming the PRIOR mode's
       time-average number — sequencing is load-bearing;
    2. aitken mass: one transfer at the COMBINED loss
       ``βij3(ait,acc)·n̄acc + βij3(ait,pca)·n̄pca``, all of it delivered
       to the ACCUM species (the pcarbon-destined share is deemed aged
       through); that share × ``fac_m2v_aitage`` accumulates the aging
       SHELL volume (so4 at face value, soa at its equivalent-so4
       hygroscopicity factor, ncl/dst contributing zero — init:906-955);
    3. the aging fraction from shell vs core against the same legacy
       8.0-monolayer criterion as gasaerexch (the Fortran comment says
       "this duplicates the code in modal_aero_gasaerexch");
    4. pcarbon mass and number: direct coagulation PLUS the aging
       fraction, capped at ``1 − 10ε``.

    ``nfreqcoag = 1`` (the box calls with the model step, so the
    every-3-hours skip logic reduces to "always run"); the ``dqdt``
    diagnostics are not carried (``q`` is updated in place in the
    Fortran and returned here).

    Returns the updated ``q``.
    """
    from mam4_jax.physics.coag import getcoags_wrapper_f

    tb = _cam_tables(topology)
    topo = tb.topology
    mait, macc, mpca = tb.mode_aitken, tb.mode_accum, tb.mode_pcarbon
    xferfrac_max = 1.0 - 10.0 * float(jnp.finfo(jnp.float64).eps)

    aircon = pmid / (_RGAS_UNIV * t)          # kmol-air/m3

    def numbconc(m):
        return jnp.maximum(0.0, q[..., tb.num_ptr[m]] * aircon)

    n_acc, n_ait, n_pca = numbconc(macc), numbconc(mait), numbconc(mpca)

    # Coefficients per pair. (frm, too) play getcoags' (aitken, accum)
    # roles; only ij0/ij3/ii0/jj0 are consumed (as in the Fortran).
    def betas(mf, mt):
        return getcoags_wrapper_f(
            t, pmid,
            dgncur_awet[..., mf], dgncur_awet[..., mt],
            topo.sigmag_amode[mf], topo.sigmag_amode[mt],
            float(tb.alnsg[mf]), float(tb.alnsg[mt]),
            wetdens_a[..., mf], wetdens_a[..., mt])

    ij0_aa, _i2a, _j2a, ij3_aa, ii0_aa, _ii2a, jj0_aa, _jj2a = betas(mait, macc)
    ij0_pa, _i2b, _j2b, ij3_pa, ii0_pa, _ii2b, jj0_pa, _jj2b = betas(mpca, macc)
    ij0_ap, _i2c, _j2c, ij3_ap, ii0_ap, _ii2c, jj0_ap, _jj2c = betas(mait, mpca)

    # --- numbers (:443-500), sequential ------------------------------------
    new_acc = n_acc / (1.0 + deltat * jj0_aa * n_acc)
    avg_acc = 0.5 * (new_acc + n_acc)
    q = q.at[..., tb.num_ptr[macc]].set(new_acc / aircon)

    new_pca = _coag_number_new(
        n_pca, deltat * ij0_pa * avg_acc, deltat * ii0_pa)
    avg_pca = 0.5 * (new_pca + n_pca)
    q = q.at[..., tb.num_ptr[mpca]].set(new_pca / aircon)

    new_ait = _coag_number_new(
        n_ait, deltat * (ij0_aa * avg_acc + ij0_ap * avg_pca),
        deltat * ii0_aa)
    q = q.at[..., tb.num_ptr[mait]].set(new_ait / aircon)

    # --- aitken mass: combined ait->acc + ait->pca(->acc) (:504-527) -------
    dumloss = ij3_aa * avg_acc + ij3_ap * avg_pca
    tmpa_frac = ij3_ap * avg_pca / jnp.maximum(dumloss, 1.0e-37)
    xferfracvol = -jnp.expm1(-dumloss * deltat)
    xferfracvol = jnp.clip(xferfracvol, 0.0, xferfrac_max)
    vol_shell = jnp.zeros_like(dumloss)
    for (lf, lt), m2v_age in zip(tb.coag_ait_pairs, tb.fac_m2v_aitage):
        xferamt = q[..., lf] * xferfracvol
        q = q.at[..., lf].add(-xferamt)
        if lt >= 0:
            q = q.at[..., lt].add(xferamt)
        vol_shell = vol_shell + xferamt * tmpa_frac * m2v_age

    # --- aging fraction (:531-546), same criterion as gasaerexch -----------
    vol_core = jnp.zeros_like(vol_shell)
    for s, lw in enumerate(tb.lmass[mpca]):
        vol_core = vol_core + q[..., lw] * tb.fac_m2v_pcarbon[s]
    tmp1 = vol_shell * dgncur_a[..., mpca] * tb.fac_volsfc_pcarbon
    tmp2 = jnp.maximum(6.0 * _DR_SO4_MONOLAYERS_PCAGE * vol_core, 0.0)
    saturated = tmp1 >= tmp2
    safe_tmp2 = jnp.where(saturated, 1.0, tmp2)
    xferfrac_pcage = jnp.where(
        saturated, xferfrac_max,
        jnp.minimum(tmp1 / safe_tmp2, xferfrac_max))

    # --- pcarbon mass + number: direct coag + aging (:550-580) -------------
    dumloss_p = ij3_pa * avg_acc
    xferfracvol = -jnp.expm1(-dumloss_p * deltat) + xferfrac_pcage
    xferfracvol = jnp.clip(xferfracvol, 0.0, xferfrac_max)
    for lf, lt in tb.coag_pca_pairs:
        xferamt = q[..., lf] * xferfracvol
        q = q.at[..., lf].add(-xferamt)
        if lt >= 0:
            q = q.at[..., lt].add(xferamt)
    xferamt = q[..., tb.num_ptr[mpca]] * xferfrac_pcage
    q = q.at[..., tb.num_ptr[mpca]].add(-xferamt)
    q = q.at[..., tb.num_ptr[macc]].add(xferamt)
    return q
