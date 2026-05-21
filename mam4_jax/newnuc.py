"""Leaf-level nucleation parameterizations (M3.6 PR-F1).

Port of two helpers from ``e3sm_src/modal_aero_newnuc.F90``:

* :func:`binary_nuc_vehk2002` — Vehkamäki et al. (2002) H₂SO₄–H₂O
  binary homogeneous nucleation rate + critical-cluster size.
* :func:`pbl_nuc_wang2008`     — Wang & Penner (2008) boundary-layer
  first- or second-order nucleation overlay.

Both are pure scalar math (no aerosol-state inputs); they take
temperature / RH / [H₂SO₄] and return nucleation diagnostics. PR-F2
will port ``mer07_veh02_nuc_mosaic_1box`` (the dispatcher that chains
these together with the Kerminen-Kulmala 2002 size correction); PR-F3
will wire it all into ``_mam_newnuc_1subarea`` in the amicphys
orchestration.

References:
  Vehkamäki et al., JGR 107, 4622 (2002), doi:10.1029/2002jd002184.
  Wang & Penner, ACPD 8, 13943 (2008).
"""
from __future__ import annotations

import jax.numpy as jnp

from . import data
from .constants import RGAS as _RGAS_J_PER_K_PER_KMOL


def binary_nuc_vehk2002(temp, rh, so4vol):
    """Vehkamäki 2002 binary H₂SO₄–H₂O nucleation parameterization.

    Mirrors ``modal_aero_newnuc.F90:1256-1448``.

    Parameters
    ----------
    temp : array
        Temperature (K).
    rh : array
        Relative humidity (0–1).
    so4vol : array
        H₂SO₄ vapor concentration (molecules / cm³).

    All three arguments broadcast against each other; outputs share
    the broadcast shape.

    Returns
    -------
    ratenucl : array
        Binary nucleation rate j (# cm⁻³ s⁻¹).
    rateloge : array
        ``log(ratenucl)`` (pre-clipping). Note: the Fortran returns
        this BEFORE the ``min(tmpa, log(1e38))`` clip that bounds
        ``ratenucl``, so the two outputs can be inconsistent for very
        large rates. We preserve that behaviour for bit-for-bit match.
    cnum_h2so4 : array
        Number of H₂SO₄ molecules in the critical nucleus.
    cnum_tot : array
        Total number of molecules in the critical nucleus.
    radius_cluster : array
        Critical-cluster radius (nm).
    """
    log_so4 = jnp.log(so4vol)
    log_rh  = jnp.log(rh)
    t1 = temp
    t2 = temp * temp
    t3 = t2 * temp

    # Sulfuric-acid mole fraction in critical cluster (Fortran lines 1295-1303).
    crit_x = (
        0.740997
        - 0.00266379   * t1
        - 0.00349998   * log_so4
        + 0.0000504022 * t1 * log_so4
        + 0.00201048   * log_rh
        - 0.000183289  * t1 * log_rh
        + 0.00157407   * log_rh ** 2
        - 0.0000179059 * t1 * log_rh ** 2
        + 0.000184403  * log_rh ** 3
        - 1.50345e-6   * t1 * log_rh ** 3
    )
    inv_x = 1.0 / crit_x

    # Polynomial coefficients for the nucleation rate (Fortran 1307-1353).
    acoe =  0.14309   + 2.21956   * t1 - 0.0273911   * t2 + 0.0000722811 * t3 +  5.91822 * inv_x
    bcoe =  0.117489  + 0.462532  * t1 - 0.0118059   * t2 + 0.0000404196 * t3 + 15.7963  * inv_x
    ccoe = -0.215554  - 0.0810269 * t1 + 0.00143581  * t2 - 4.7758e-6    * t3 -  2.91297 * inv_x
    dcoe = -3.58856   + 0.049508  * t1 - 0.00021382  * t2 + 3.10801e-7   * t3 -  0.0293333 * inv_x
    ecoe =  1.14598   - 0.600796  * t1 + 0.00864245  * t2 - 0.0000228947 * t3 -  8.44985  * inv_x
    fcoe =  2.15855   + 0.0808121 * t1 - 0.000407382 * t2 - 4.01957e-7   * t3 +  0.721326 * inv_x
    gcoe =  1.6241    - 0.0160106 * t1 + 0.0000377124* t2 + 3.21794e-8   * t3 -  0.0113255 * inv_x
    hcoe =  9.71682   - 0.115048  * t1 + 0.000157098 * t2 + 4.00914e-7   * t3 +  0.71186  * inv_x
    icoe = -1.05611   + 0.00903378* t1 - 0.0000198417* t2 + 2.46048e-8   * t3 -  0.0579087 * inv_x
    jcoe = -0.148712  + 0.00283508* t1 - 9.24619e-6  * t2 + 5.00427e-9   * t3 -  0.0127081 * inv_x

    rateloge = (
        acoe
        + bcoe * log_rh
        + ccoe * log_rh ** 2
        + dcoe * log_rh ** 3
        + ecoe * log_so4
        + fcoe * log_rh * log_so4
        + gcoe * log_rh ** 2 * log_so4
        + hcoe * log_so4 ** 2
        + icoe * log_rh * log_so4 ** 2
        + jcoe * log_so4 ** 3
    )
    rateloge_clipped = jnp.minimum(rateloge, jnp.log(1.0e38))
    ratenucl = jnp.exp(rateloge_clipped)

    # Coefficient polynomials for the total cluster size (Fortran 1377-1423).
    acoe = -0.00295413 - 0.0976834   * t1 + 0.00102485  * t2 - 2.18646e-6  * t3 - 0.101717   * inv_x
    bcoe = -0.00205064 - 0.00758504  * t1 + 0.000192654 * t2 - 6.7043e-7   * t3 - 0.255774   * inv_x
    ccoe =  0.00322308 + 0.000852637 * t1 - 0.0000154757* t2 + 5.66661e-8  * t3 + 0.0338444  * inv_x
    dcoe =  0.0474323  - 0.000625104 * t1 + 2.65066e-6  * t2 - 3.67471e-9  * t3 - 0.000267251* inv_x
    ecoe = -0.0125211  + 0.00580655  * t1 - 0.000101674 * t2 + 2.88195e-7  * t3 + 0.0942243  * inv_x
    fcoe = -0.038546   - 0.000672316 * t1 + 2.60288e-6  * t2 + 1.19416e-8  * t3 - 0.00851515 * inv_x
    gcoe = -0.0183749  + 0.000172072 * t1 - 3.71766e-7  * t2 - 5.14875e-10 * t3 + 0.00026866 * inv_x
    hcoe = -0.0619974  + 0.000906958 * t1 - 9.11728e-7  * t2 - 5.36796e-9  * t3 - 0.00774234 * inv_x
    icoe =  0.0121827  - 0.00010665  * t1 + 2.5346e-7   * t2 - 3.63519e-10 * t3 + 0.000610065* inv_x
    jcoe =  0.000320184 - 0.0000174762 * t1 + 6.06504e-8 * t2 - 1.4177e-11 * t3 + 0.000135751* inv_x

    cnum_tot = jnp.exp(
        acoe
        + bcoe * log_rh
        + ccoe * log_rh ** 2
        + dcoe * log_rh ** 3
        + ecoe * log_so4
        + fcoe * log_rh * log_so4
        + gcoe * log_rh ** 2 * log_so4
        + hcoe * log_so4 ** 2
        + icoe * log_rh * log_so4 ** 2
        + jcoe * log_so4 ** 3
    )

    cnum_h2so4 = cnum_tot * crit_x

    radius_cluster = jnp.exp(
        -1.6524245 + 0.42316402 * crit_x + 0.3346648 * jnp.log(cnum_tot)
    )

    return ratenucl, rateloge, cnum_h2so4, cnum_tot, radius_cluster


# ``adjust_factor_pbl_ratenucl`` — module-data constant in
# ``modal_aero_newnuc.F90:39``. Default 1.0; the box-model namelist
# doesn't change it. If a future configuration sets ``newnuc_adjust_factor_pbl``
# in the namelist, we'd extend the amicphys init dump to capture it.
_ADJUST_FACTOR_PBL_RATENUCL = 1.0


# ---------------------------------------------------------------------------
# Dispatcher-level constants (M3.6 PR-F2).
# All `parameter` in `modal_aero_newnuc.F90`'s
# `mer07_veh02_nuc_mosaic_1box` scope (lines 712-731). Transcribed
# rather than captured because they never change at runtime.
# ---------------------------------------------------------------------------

# Module-level adjustment factor — `parameter` at line 37, default 1.0.
_ADJUST_FACTOR_BIN_TERN_RATENUCL = 1.0

# H₂SO₄ accommodation coefficient in the KK2002 cs_prime calculation.
_ACCOM_COEF_H2SO4 = 0.65

# Ammonium-sulfate / -bisulfate / sulfuric-acid densities (kg/m³) — all
# at the same 1.770e3 to match cam3 modal_aero densities (Fortran lines
# 722-724).
_DENS_AMMSULF   = 1.770e3
_DENS_AMMBISULF = 1.770e3
_DENS_SULFACID  = 1.770e3
# Molecular weights of the same (g/mol). For ammbisulf/sulfacid the
# Fortran uses 114 & 96 (not 115 & 98) because aerosol H⁺ isn't tracked.
_MW_AMMSULF     = 132.0
_MW_AMMBISULF   = 114.0
_MW_SULFACID    =  96.0

# Sulfate and ammonium molecular weights from physconst (mwso4, mwnh4).
# Fortran `parameter`s in box_model_utils/physconst.F90.
_MW_SO4A = 96.0
_MW_NH4A = 18.0


def pbl_nuc_wang2008(so4vol, flagaa,
                     ratenucl, rateloge,
                     cnum_tot, cnum_h2so4, cnum_nh3, radius_cluster):
    """Wang 2008 boundary-layer nucleation overlay.

    Mirrors ``modal_aero_newnuc.F90:1179-1250``. Computes a candidate
    PBL nucleation rate from ``so4vol`` (linear if ``flagaa==11``,
    quadratic if ``flagaa==12``) and returns the binary inputs
    unchanged if that rate is lower than the prior — otherwise replaces
    them with the PBL values (pure-H₂SO₄ 1 nm-diameter clusters).

    ``flagaa`` is a Python int (concrete, not a JAX-traced value) so
    we can branch at trace time. Other arguments broadcast.

    Returns
    -------
    flagaa2 : int array
        ``flagaa`` where the PBL path won, else ``-1`` (Fortran's
        sentinel — caller compares ``newnuc_method_flagaa2 > 0``).
    Updated tuple ``(ratenucl, rateloge, cnum_tot, cnum_h2so4,
    cnum_nh3, radius_cluster)`` matching the Fortran arg order.
    """
    if flagaa == 11:
        tmp_ratenucl = 1.0e-6 * so4vol
    elif flagaa == 12:
        tmp_ratenucl = 1.0e-12 * so4vol ** 2
    else:
        # Fortran early-return — no PBL path active, return inputs unchanged.
        flagaa2 = jnp.full_like(jnp.asarray(so4vol), -1, dtype=jnp.int32)
        return (flagaa2, ratenucl, rateloge,
                cnum_tot, cnum_h2so4, cnum_nh3, radius_cluster)

    tmp_ratenucl = tmp_ratenucl * _ADJUST_FACTOR_PBL_RATENUCL
    tmp_rateloge = jnp.log(jnp.maximum(1.0e-38, tmp_ratenucl))

    # Where the PBL rate beats the prior, swap in the PBL values.
    pbl_wins = tmp_rateloge > rateloge

    # Fixed PBL particle properties (Fortran 1236-1246): 1 nm diameter,
    # pure H₂SO₄, mass = volume * 1.8 g/cm³.
    radius_pbl = 0.5    # nm
    tmp_diam   = radius_pbl * 2.0e-7        # cm
    tmp_volu   = (tmp_diam ** 3) * (jnp.pi / 6.0)   # cm³
    tmp_mass   = tmp_volu * 1.8             # g
    cnum_h2so4_pbl = (tmp_mass / 98.0) * 6.023e23   # # of H₂SO₄ molecules

    new_ratenucl       = jnp.where(pbl_wins, tmp_ratenucl, ratenucl)
    new_rateloge       = jnp.where(pbl_wins, tmp_rateloge, rateloge)
    new_cnum_tot       = jnp.where(pbl_wins, cnum_h2so4_pbl, cnum_tot)
    new_cnum_h2so4     = jnp.where(pbl_wins, cnum_h2so4_pbl, cnum_h2so4)
    new_cnum_nh3       = jnp.where(pbl_wins, 0.0, cnum_nh3)
    new_radius_cluster = jnp.where(pbl_wins, radius_pbl, radius_cluster)
    flagaa2            = jnp.where(pbl_wins, flagaa, -1).astype(jnp.int32)

    return (flagaa2, new_ratenucl, new_rateloge,
            new_cnum_tot, new_cnum_h2so4, new_cnum_nh3, new_radius_cluster)


# ---------------------------------------------------------------------------
# Dispatcher (M3.6 PR-F2)
# ---------------------------------------------------------------------------

def mer07_veh02_nuc_mosaic_1box(dtnuc, temp, rh, press, zm, pblh,
                                 qh2so4_cur, qh2so4_avg, h2so4_uptkrate,
                                 dplom_sect, dphim_sect,
                                 newnuc_method_flagaa=11):
    """Port of ``mer07_veh02_nuc_mosaic_1box`` (``modal_aero_newnuc.F90:598-1173``).

    Dispatcher that wraps the PR-F1 leaf parameterizations
    (:func:`binary_nuc_vehk2002`, :func:`pbl_nuc_wang2008`) with unit
    conversion, the Kerminen-Kulmala 2002 size correction, grown-particle
    composition logic, and the final ``qh2so4_del / qso4a_del /
    qnuma_del`` accounting.

    **MAM4-MOM-specific simplifications** (we don't port the unreachable
    code paths):

    * ``qnh3_cur = 0`` always (no NH₃ in this build) — ternary nucleation
      is unreachable. The Fortran's `if (flagaa != 2) and (nh3ppt >= 0.1)`
      branch is skipped.
    * ``nsize = 1`` — the dispatcher accepts arrays for ``dplom_sect /
      dphim_sect`` but the amicphys caller always passes length-1
      arrays. We accept Python scalars instead.
    * ``newnuc_method_flagaa = 11`` is the only value tested. The
      Fortran also accepts 1, 2, 12; we hardcode the dispatch path and
      assume the caller validates the flag.
    * Diagnostic output (Fortran lines 1074-1170) is omitted.

    Parameters
    ----------
    dtnuc : scalar
        Nucleation time step (s).
    temp, rh, press : array
        Temperature (K), relative humidity (0-1), pressure (Pa).
    zm, pblh : array
        Midpoint altitude (m), PBL height (m).
    qh2so4_cur, qh2so4_avg : array
        Current & average H₂SO₄ gas mixing ratios (mol/mol-air).
    h2so4_uptkrate : array
        H₂SO₄ uptake rate to aerosol (1/s) — from gasaerexch's
        :func:`_gas_aer_uptkrates_1box1gas`.
    dplom_sect, dphim_sect : scalar
        Dry diameter bin bounds (m) for the host code's size bin (length
        1 in MAM4-MOM — Aitken-mode lo/hi from ``data.DGNUMLO_AMODE``).
    newnuc_method_flagaa : int
        Static (Python int). 11 = first-order PBL (Fortran default).

    Returns
    -------
    isize_nuc : int array
        Always 1 in MAM4-MOM (nsize=1).
    qnuma_del : array
        Change to aerosol number mixing ratio (#/mol-air). Aerosol
        deltas > 0; gas deltas < 0.
    qso4a_del, qnh4a_del : array
        Change to aerosol SO₄ / NH₄ mass mixing ratios (mol/mol-air).
        ``qnh4a_del`` is always 0 (no NH₃).
    qh2so4_del, qnh3_del : array
        Change to gas H₂SO₄ / NH₃ mixing ratios (mol/mol-air).
        ``qnh3_del`` is always 0.
    dens_nh4so4a : array
        Dry density of the new NH₄-SO₄ aerosol (kg/m³). For no-NH₃ this
        equals ``_DENS_SULFACID = 1770``.
    dnclusterdt : array
        Cluster nucleation rate (#/m³/s).
    """
    # Local constants from Fortran scope.
    rgas_local = _RGAS_J_PER_K_PER_KMOL / 1.0e3   # J/K/mol (vs J/K/kmol)
    avogad     = 6.02214e26 / 1.0e3  # 1/mol (vs 1/kmol)
    pi = jnp.pi
    onethird = 1.0 / 3.0

    # Convert qh2so4 (mol/mol) → so4vol (molec/cm³).
    cair = press / (temp * rgas_local)                  # mol/m³
    so4vol_in = qh2so4_avg * cair * avogad * 1.0e-6      # molec/cm³

    # Stage 1: binary nucleation (no NH₃ → never ternary). Fortran
    # clamps inputs to Vehkamäki's valid range; we do the same.
    temp_bb   = jnp.maximum(230.15, jnp.minimum(305.15, temp))
    rh_bb     = jnp.maximum(1.0e-4, jnp.minimum(1.0,    rh))
    so4vol_bb = jnp.maximum(1.0e4,  jnp.minimum(1.0e11, so4vol_in))
    rate_bin, log_bin, cnum_h2so4, cnum_tot, radius_cluster = \
        binary_nuc_vehk2002(temp_bb, rh_bb, so4vol_bb)

    # If so4vol_in < 1e4, Fortran skips the binary call and uses the
    # init values (ratenuclt = 1e-38, rateloge = log(1e-38)).
    init_log = jnp.log(1.0e-38)
    above_min_so4 = so4vol_in >= 1.0e4
    ratenuclt = jnp.where(above_min_so4, rate_bin, 1.0e-38)
    rateloge  = jnp.where(above_min_so4, log_bin,  init_log)
    cnum_h2so4 = jnp.where(above_min_so4, cnum_h2so4, 0.0)
    cnum_tot   = jnp.where(above_min_so4, cnum_tot,   0.0)
    radius_cluster = jnp.where(above_min_so4, radius_cluster, 0.0)
    cnum_nh3 = jnp.zeros_like(rateloge)

    # Stage 2: bin/tern adjustment factor.
    rateloge = rateloge + jnp.log(jnp.maximum(1.0e-38, _ADJUST_FACTOR_BIN_TERN_RATENUCL))

    # Stage 3: PBL nucleation overlay.
    # Fortran calls pbl_nuc_wang2008 unconditionally inside the
    # `if (zm <= max(pblh, 100))` guard; we use jnp.where to gate.
    if newnuc_method_flagaa in (11, 12):
        in_pbl = zm <= jnp.maximum(pblh, 100.0)
        pbl_out = pbl_nuc_wang2008(
            so4vol_in, newnuc_method_flagaa,
            ratenuclt, rateloge,
            cnum_tot, cnum_h2so4, cnum_nh3, radius_cluster,
        )
        _flagaa2_pbl, rate_pbl, log_pbl, ct_pbl, ch_pbl, cn_pbl, rad_pbl = pbl_out
        ratenuclt      = jnp.where(in_pbl, rate_pbl, ratenuclt)
        rateloge       = jnp.where(in_pbl, log_pbl,  rateloge)
        cnum_tot       = jnp.where(in_pbl, ct_pbl,   cnum_tot)
        cnum_h2so4     = jnp.where(in_pbl, ch_pbl,   cnum_h2so4)
        cnum_nh3       = jnp.where(in_pbl, cn_pbl,   cnum_nh3)
        radius_cluster = jnp.where(in_pbl, rad_pbl,  radius_cluster)

    # Stage 4: nucleation-rate exit gate. Fortran returns 0 if
    # rateloge <= -13.82. We compute the downstream values
    # unconditionally and mask the outputs at the very end.
    rate_ok = rateloge > -13.82

    ratenuclt = jnp.exp(rateloge)
    ratenuclt_bb = ratenuclt * 1.0e6    # #/m³/s
    dnclusterdt = jnp.where(rate_ok, ratenuclt_bb, 0.0)

    # Stage 5: wet/dry volume ratio for cluster sizing.
    rh_clamp = jnp.maximum(0.10, jnp.minimum(0.95, rh))
    wetvol_dryvol = 1.0 - 0.56 / jnp.log(rh_clamp)

    # Stage 6: size-bin assignment (collapses to isize_nuc=1 for nsize=1).
    voldry_clus = (jnp.maximum(cnum_h2so4, 1.0) * _MW_SO4A + cnum_nh3 * _MW_NH4A) / (
        1.0e3 * _DENS_SULFACID * avogad
    )
    voldry_clus = voldry_clus * (data.MW_SO4A_HOST / _MW_SO4A)
    dpdry_clus = (voldry_clus * 6.0 / pi) ** onethird

    isize_nuc = jnp.ones_like(jnp.asarray(so4vol_in), dtype=jnp.int32)
    # For nsize=1: igrow=1 if dpdry_clus <= dplom; igrow=0 otherwise.
    # dpdry_part = dplom if igrow=1; = clamped dpdry_clus otherwise.
    igrow = (dpdry_clus <= dplom_sect).astype(jnp.float64)
    dpdry_part = jnp.where(
        dpdry_clus <= dplom_sect,
        dplom_sect,
        jnp.where(dpdry_clus >= dphim_sect, dphim_sect, dpdry_clus),
    )
    voldry_part = (pi / 6.0) * (dpdry_part ** 3)

    # Stage 7: grown-particle composition. No NH₃ → tmp_n1=tmp_n2=0,
    # tmp_n3=1 (pure sulfacid). For igrow<=0 the Fortran also picks
    # pure sulfacid.
    # Both branches give the same answer here, so we just hardcode:
    dens_part = _DENS_SULFACID
    dens_nh4so4a = jnp.full_like(rateloge, dens_part)
    mass_part = voldry_part * dens_part
    molenh4a_per_moleso4a = 0.0
    kgaero_per_moleso4a = 1.0e-3 * _MW_SULFACID * (data.MW_SO4A_HOST / _MW_SO4A)

    # Stage 8: wet volume fraction.
    tmpb_wvf = 1.0 + molenh4a_per_moleso4a * 17.0 / 98.0
    wet_volfrac_so4a = 1.0 / (wetvol_dryvol * tmpb_wvf)

    # Stage 9: Kerminen-Kulmala 2002 size correction.
    # igrow > 0 path:
    tmp_spd = 14.7 * jnp.sqrt(temp)                  # H₂SO₄ molecular speed (m/s)
    gr_kk = 3.0e-9 * tmp_spd * _MW_SULFACID * so4vol_in / (dens_part * wet_volfrac_so4a)
    dfin_kk = 1.0e9 * dpdry_part * (wetvol_dryvol ** onethird)
    dnuc_kk = jnp.maximum(2.0 * radius_cluster, 1.0)
    gamma_kk = (0.23 * dnuc_kk ** 0.2
                * (dfin_kk / 3.0) ** 0.075
                * (dens_part * 1.0e-3) ** (-0.33)
                * (temp / 293.0) ** (-0.75))
    tmpa_kk = jnp.maximum(h2so4_uptkrate * 3600.0, 0.0)   # 1/h
    tmpb_kk = 6.7037e-6 * (temp ** 0.75) / cair * 3600.0  # m²/h
    cs_prime_kk = tmpa_kk / (4.0 * pi * tmpb_kk * _ACCOM_COEF_H2SO4)
    nu_kk = gamma_kk * cs_prime_kk / gr_kk
    factor_kk_grow = jnp.exp(nu_kk / dfin_kk - nu_kk / dnuc_kk)

    # igrow <= 0 path: factor_kk = 1.
    factor_kk = jnp.where(igrow > 0, factor_kk_grow, 1.0)
    ratenuclt_kk = ratenuclt_bb * factor_kk

    # Stage 10: compute deltas.
    qmolso4a_del_max = jnp.maximum(0.0, ratenuclt_kk * dtnuc * mass_part) / (
        kgaero_per_moleso4a * cair
    )

    # freducea = qh2so4_cur / qmolso4a_del_max if max exceeds available.
    freducea = jnp.where(
        qmolso4a_del_max > qh2so4_cur,
        qh2so4_cur / jnp.maximum(qmolso4a_del_max, 1.0e-30),
        1.0,
    )
    # freduceb (NH₃ limit) — molenh4a_per_moleso4a < 1e-10, so freduceb = 1.
    freduce = freducea

    # Bottom gate: freduce * ratenuclt_kk <= 1e-12 #/m³/s → return zeros.
    final_ok = rate_ok & (freduce * ratenuclt_kk > 1.0e-12)

    # Compute final deltas.
    qh2so4_del_pos = jnp.minimum(0.9999 * qh2so4_cur, freduce * qmolso4a_del_max)
    qh2so4_del = jnp.where(final_ok, -qh2so4_del_pos, 0.0)
    qnh3_del   = jnp.zeros_like(qh2so4_del)
    qso4a_del  = -qh2so4_del
    qnh4a_del  = jnp.zeros_like(qh2so4_del)

    # qnuma_del = 1e-3 * (qso4a_del * mw_so4a + qnh4a_del * mw_nh4a) / mass_part
    qnuma_del_raw = 1.0e-3 * (qso4a_del * _MW_SO4A + qnh4a_del * _MW_NH4A) / jnp.maximum(
        mass_part, 1.0e-30
    )
    qnuma_del = jnp.where(final_ok, qnuma_del_raw, 0.0)

    return (isize_nuc, qnuma_del, qso4a_del, qnh4a_del,
            qh2so4_del, qnh3_del, dens_nh4so4a, dnclusterdt)
