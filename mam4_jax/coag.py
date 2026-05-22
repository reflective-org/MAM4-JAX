"""Modal-aerosol coagulation — leaf functions.

JAX port of the leaf coagulation routines from
``mam4-original-src-code/e3sm_src/modal_aero_coag.F90``, which the
process-level entry point :mod:`mam4_jax.processes.coag` (today a
``NotImplementedError`` stub) will compose in PR-G3.

**Current contents (M3.6 PR-G1 + PR-G2):**

* :func:`getcoags` — closed-form Whitby-style coagulation coefficients
  (Fortran lines 1177–2858). Returns the 8 inter- and intramodal
  zeroeth / second / third-moment coefficients used by
  ``modal_aero_coag_sub``.
* :func:`getcoags_wrapper_f` — the wrapper (Fortran lines 999–1129)
  that preps ``lamda`` / ``knc`` / ``kfmat*`` from ``(T, P, densities)``,
  calls :func:`getcoags`, and post-processes the 8 raw outputs into
  the 8 ``betaij*`` / ``betaii*`` / ``betajj*`` coefficients consumed
  by ``mam_coag_1subarea``.

Both functions are line-by-line transcriptions of the Fortran. The
correction-factor lookup tables (``bm0``, ``bm0ij``, ``bm3i``,
``bm2ii``, ``bm2iitt``, ``bm2ij``, ``bm2ji``) live alongside this
module as ``_coag_tables.npz`` — extracted once from the upstream
Fortran ``data`` declarations by ``scripts/extract_coag_tables.py``.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .constants import BOLTZ, PSTD, TMELT

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_TABLES_PATH = Path(__file__).resolve().parent / "_coag_tables.npz"
_TABLES = np.load(_TABLES_PATH)

_BM0     = jnp.asarray(_TABLES["bm0"])       # shape (10,)
_BM2II   = jnp.asarray(_TABLES["bm2ii"])     # shape (10,)
_BM2IITT = jnp.asarray(_TABLES["bm2iitt"])   # shape (10,)
_BM0IJ   = jnp.asarray(_TABLES["bm0ij"])     # shape (10, 10, 10)
_BM3I    = jnp.asarray(_TABLES["bm3i"])      # shape (10, 10, 10)
_BM2IJ   = jnp.asarray(_TABLES["bm2ij"])     # shape (10, 10, 10)
_BM2JI   = jnp.asarray(_TABLES["bm2ji"])     # shape (10, 10, 10)


# ---------------------------------------------------------------------------
# getcoags — closed-form Whitby coagulation coefficients
# ---------------------------------------------------------------------------

_A        = 1.246
_TWO3RDS  = 2.0 / 3.0
# constii = abs(0.5 * 2**(2/3) - 1) — a constant the Fortran recomputes
# each call. Match the Fortran's expression exactly so any ULP drift in
# constii is identical.
_CONSTII  = abs(0.5 * (2.0) ** _TWO3RDS - 1.0)
_DLGSQT2  = 1.0 / np.log(np.sqrt(2.0))


def _clip_index(idx):
    """Mirror Fortran ``max(1, min(10, nint(x)))`` → 0-based [0, 9]."""
    # jnp.round uses banker's rounding; Fortran nint rounds half away
    # from zero. The arguments here (`4*(sigmag - 0.75)`, `1 + log/ln(√2)`)
    # land on integers only at sigmag/rat values we don't reach in
    # practice (see Whitby 1991 §H), so this difference is moot for the
    # MAM4 use case. Document and move on; if a future fixture lands on
    # a half-integer we'll add an explicit ``floor(x + 0.5)``.
    return jnp.clip(jnp.round(idx).astype(jnp.int32), 1, 10) - 1


def getcoags(lamda, kfmatac, kfmat, kfmac, knc,
             dgatk, dgacc, sgatk, sgacc, xxlsgat, xxlsgac):
    """Whitby-style closed-form coagulation coefficients.

    Direct transcription of ``modal_aero_coag.F90:getcoags`` lines
    1177–2858. All inputs are scalars (or arrays that broadcast); the
    8 returned coefficients have the broadcast shape.

    Parameters
    ----------
    lamda
        Mean free path of air [m].
    kfmatac
        Free-molecular regime coefficient, Aitken→accumulation
        [m^(1/2)·s⁻¹].
    kfmat, kfmac
        Free-molecular regime coefficients, intramodal (Aitken, accum)
        [m^(1/2)·s⁻¹].
    knc
        Near-continuum regime coefficient [m³·s⁻¹] (mode-independent).
    dgatk, dgacc
        Modal geometric-mean diameters [m] (Aitken, accumulation).
    sgatk, sgacc
        Modal geometric standard deviations [-].
    xxlsgat, xxlsgac
        ``log(sgatk)``, ``log(sgacc)``.

    Returns
    -------
    (qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12)
        Coagulation coefficients corresponding to the Fortran outputs of
        the same name. See lines 2659/2711/2718/2747/2769/2787/2824/2853
        for the assignment sites.
    """
    # --- esat / esac series ----------------------------------------------
    esat01 = jnp.exp(0.125 * xxlsgat * xxlsgat)
    esac01 = jnp.exp(0.125 * xxlsgac * xxlsgac)

    esat04 = esat01 ** 4
    esac04 = esac01 ** 4

    esat05 = esat04 * esat01
    esac05 = esac04 * esac01

    esat08 = esat04 * esat04
    esac08 = esac04 * esac04

    esat09 = esat08 * esat01
    esac09 = esac08 * esac01

    esat16 = esat08 * esat08
    esac16 = esac08 * esac08

    esat20 = esat16 * esat04
    esac20 = esac16 * esac04

    esat24 = esat20 * esat04
    esac24 = esac20 * esac04

    esat25 = esat20 * esat05
    esac25 = esac20 * esac05

    esat36 = esat20 * esat16
    esac36 = esac20 * esac16

    esat49 = esat24 * esat25

    esat64 = esat20 * esat20 * esat24
    esac64 = esac20 * esac20 * esac24

    esat100 = esat64 * esat36

    # --- diameter powers --------------------------------------------------
    dgat2 = dgatk * dgatk
    dgat3 = dgatk * dgatk * dgatk
    dgac2 = dgacc * dgacc
    dgac3 = dgacc * dgacc * dgacc

    sqdgat  = jnp.sqrt(dgatk)
    sqdgac  = jnp.sqrt(dgacc)
    sqdgat5 = dgat2 * sqdgat
    sqdgac5 = dgac2 * sqdgac
    sqdgat7 = dgat3 * sqdgat

    # xm2/xm3 are computed in the Fortran but not used in any output —
    # keep them out of the JAX port (would force unused computation).

    # --- diameter ratio (free-molecular regime) ---------------------------
    r   = sqdgac / sqdgat
    r2  = r * r
    rx4 = r2 * r2
    r6  = r2 * r2 * r2
    ri1 = 1.0 / r
    ri2 = 1.0 / r2
    ri3 = 1.0 / (r2 * r)
    ri4 = ri2 * ri2

    kngat = 2.0 * lamda / dgatk
    kngac = 2.0 * lamda / dgacc

    rat = dgacc / dgatk

    # --- correction-factor index lookup -----------------------------------
    n2n = _clip_index(4.0 * (sgatk - 0.75))
    n2a = _clip_index(4.0 * (sgacc - 0.75))
    n1  = _clip_index(1.0 + _DLGSQT2 * jnp.log(rat))

    bm0_n2n     = _BM0[n2n]
    bm0_n2a     = _BM0[n2a]
    bm2ii_n2n   = _BM2II[n2n]
    bm2ii_n2a   = _BM2II[n2a]
    bm2iitt_n2n = _BM2IITT[n2n]
    bm2iitt_n2a = _BM2IITT[n2a]
    bm0ij_v     = _BM0IJ[n1, n2n, n2a]
    bm3i_v      = _BM3I[n1, n2n, n2a]
    bm2ij_v     = _BM2IJ[n1, n2n, n2a]
    bm2ji_v     = _BM2JI[n1, n2n, n2a]

    # --- intermodal: zeroeth moment (lines 2641–2661) ---------------------
    coagnc0 = knc * (
        2.0 + _A * (kngat * (esat04 + r2 * esat16 * esac04)
                  + kngac * (esac04 + ri2 * esac16 * esat04))
        + (r2 + ri2) * esat04 * esac04
    )
    coagfm0 = kfmatac * sqdgat * bm0ij_v * (
        esat01 + r * esac01 + 2.0 * r2 * esat01 * esac04
        + rx4 * esat09 * esac16 + ri3 * esat16 * esac09
        + 2.0 * ri1 * esat04 + esac01
    )
    coagatac0 = coagnc0 * coagfm0 / (coagnc0 + coagfm0)
    qn12 = coagatac0

    # --- intermodal: second moment (lines 2678–2718) ----------------------
    i1nc = knc * dgat2 * (
        2.0 * esat16
        + r2 * esat04 * esac04
        + ri2 * esat36 * esac04
        + _A * kngat * (
            esat04
            + ri2 * esat16 * esac04
            + ri4 * esat36 * esac16
            + r2 * esac04
        )
    )
    i1fm = kfmatac * sqdgat5 * bm2ij_v * (
        esat25
        + 2.0 * r2 * esat09 * esac04
        + rx4 * esat01 * esac16
        + ri3 * esat64 * esac09
        + 2.0 * ri1 * esat36 * esac01
        + r * esat16 * esac01
    )
    i1 = (i1fm * i1nc) / (i1fm + i1nc)

    coagatac2 = i1
    qs12 = coagatac2

    coagacat2 = ((1.0 + r6) ** _TWO3RDS - rx4) * i1
    qs21 = coagacat2 * bm2ji_v

    # --- intermodal: third moment (lines 2724–2747) -----------------------
    coagnc3 = knc * dgat3 * (
        2.0 * esat36
        + _A * kngat * (esat16 + r2 * esat04 * esac04)
        + _A * kngac * (esat36 * esac04 + ri2 * esat64 * esac16)
        + r2 * esat16 * esac04 + ri2 * esat64 * esac04
    )
    coagfm3 = kfmatac * sqdgat7 * bm3i_v * (
        esat49
        + r * esat36 * esac01
        + 2.0 * r2 * esat25 * esac04
        + rx4 * esat09 * esac16
        + ri3 * esat100 * esac09
        + 2.0 * ri1 * esat64 * esac01
    )
    coagatac3 = coagnc3 * coagfm3 / (coagnc3 + coagfm3)
    qv12 = coagatac3

    # --- intramodal: zeroeth moment (lines 2757–2787) ---------------------
    coagnc_at = knc * (1.0 + esat08 + _A * kngat * (esat20 + esat04))
    coagfm_at = kfmat * sqdgat * bm0_n2n * (esat01 + esat25 + 2.0 * esat05)
    coagatat0 = coagfm_at * coagnc_at / (coagfm_at + coagnc_at)
    qn11 = coagatat0

    coagnc_ac = knc * (1.0 + esac08 + _A * kngac * (esac20 + esac04))
    coagfm_ac = kfmac * sqdgac * bm0_n2a * (esac01 + esac25 + 2.0 * esac05)
    coagacac0 = coagfm_ac * coagnc_ac / (coagfm_ac + coagnc_ac)
    qn22 = coagacac0

    # --- intramodal: second moment (lines 2801–2853) ----------------------
    i1nc_at = knc * dgat2 * (
        2.0 * esat16
        + esat04 * esat04
        + esat36 * esat04
        + _A * kngat * (
            2.0 * esat04
            + esat16 * esat04
            + esat36 * esat16
        )
    )
    i1fm_at = kfmat * sqdgat5 * bm2ii_n2n * (
        esat25
        + 2.0 * esat09 * esat04
        + esat01 * esat16
        + esat64 * esat09
        + 2.0 * esat36 * esat01
        + esat16 * esat01
    )
    i1_at = (i1nc_at * i1fm_at) / (i1nc_at + i1fm_at)
    coagatat2 = _CONSTII * i1_at
    qs11 = coagatat2 * bm2iitt_n2n

    i1nc_ac = knc * dgac2 * (
        2.0 * esac16
        + esac04 * esac04
        + esac36 * esac04
        + _A * kngac * (
            2.0 * esac04
            + esac16 * esac04
            + esac36 * esac16
        )
    )
    i1fm_ac = kfmac * sqdgac5 * bm2ii_n2a * (
        esac25
        + 2.0 * esac09 * esac04
        + esac01 * esac16
        + esac64 * esac09
        + 2.0 * esac36 * esac01
        + esac16 * esac01
    )
    i1_ac = (i1nc_ac * i1fm_ac) / (i1nc_ac + i1fm_ac)
    coagacac2 = _CONSTII * i1_ac
    qs22 = coagacac2 * bm2iitt_n2a

    return qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12


# ---------------------------------------------------------------------------
# getcoags_wrapper_f — prep + post-processing around getcoags
# ---------------------------------------------------------------------------


def getcoags_wrapper_f(airtemp, airprs,
                       dgatk, dgacc, sgatk, sgacc,
                       xxlsgat, xxlsgac,
                       pdensat, pdensac):
    """Wrap :func:`getcoags` with input prep and CMAQ→MIRAGE2 conversion.

    Direct transcription of ``modal_aero_coag.F90:getcoags_wrapper_f``
    lines 999–1129. Computes the four free-molecular / near-continuum
    regime coefficients from ``(airtemp, airprs, pdensat, pdensac)``,
    calls :func:`getcoags`, then divides the raw second/third-moment
    outputs by the appropriate ``exp(k·log²σ)·d^p`` factors and clamps
    to ``≥ 0``.

    Parameters
    ----------
    airtemp, airprs
        Air temperature (K) and pressure (Pa).
    dgatk, dgacc
        Aitken / accumulation geometric-mean diameters (m).
    sgatk, sgacc
        Aitken / accumulation geometric standard deviations (-).
    xxlsgat, xxlsgac
        ``log(sgatk)``, ``log(sgacc)``.
    pdensat, pdensac
        Aitken / accumulation modal particle density (kg/m³).

    Returns
    -------
    (betaij0, betaij2i, betaij2j, betaij3, betaii0, betaii2, betajj0, betajj2)
        Post-processed coagulation coefficients consumed by
        ``mam_coag_1subarea``.
    """
    t0 = TMELT + 15.0
    sqrt_temp = jnp.sqrt(airtemp)

    # Mean free path — U.S. Standard Atmosphere 1962, table I.2.8.
    lamda = 6.6328e-8 * PSTD * airtemp / (t0 * airprs)

    # Dynamic viscosity — U.S. Standard Atmosphere 1962, page 14.
    amu = 1.458e-6 * airtemp * sqrt_temp / (airtemp + 110.4)

    knc     = _TWO3RDS * BOLTZ * airtemp / amu
    kfmat   = jnp.sqrt(3.0 * BOLTZ * airtemp / pdensat)
    kfmac   = jnp.sqrt(3.0 * BOLTZ * airtemp / pdensac)
    kfmatac = jnp.sqrt(6.0 * BOLTZ * airtemp / (pdensat + pdensac))

    # getcoags returns (qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12)
    # which the Fortran wrapper stores as
    #   (batat(2), batat(1), bacac(2), bacac(1),
    #    batac(2), bacat(2), batac(1), c3ij).
    qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12 = getcoags(
        lamda, kfmatac, kfmat, kfmac, knc,
        dgatk, dgacc, sgatk, sgacc, xxlsgat, xxlsgac,
    )

    # CMAQ → MIRAGE2 conversion factors.
    dumacc2 = (dgacc * dgacc) * jnp.exp(2.0 * xxlsgac * xxlsgac)
    dumatk2 = (dgatk * dgatk) * jnp.exp(2.0 * xxlsgat * xxlsgat)
    dumatk3 = (dgatk * dgatk * dgatk) * jnp.exp(4.5 * xxlsgat * xxlsgat)

    betaii0  = jnp.maximum(0.0, qn11)          # batat(1)
    betajj0  = jnp.maximum(0.0, qn22)          # bacac(1)
    betaij0  = jnp.maximum(0.0, qn12)          # batac(1)
    betaij3  = jnp.maximum(0.0, qv12 / dumatk3)

    betajj2  = jnp.maximum(0.0, qs22 / dumacc2)
    betaii2  = jnp.maximum(0.0, qs11 / dumatk2)
    betaij2i = jnp.maximum(0.0, qs12 / dumatk2)  # batac(2)
    betaij2j = jnp.maximum(0.0, qs21 / dumatk2)  # bacat(2)

    return (betaij0, betaij2i, betaij2j, betaij3,
            betaii0, betaii2, betajj0, betajj2)
