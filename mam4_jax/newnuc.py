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
