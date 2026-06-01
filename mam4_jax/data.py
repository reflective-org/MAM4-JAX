"""Mode/species constants and tracer-index bookkeeping for MAM4-MOM.

Per ADR-008, the primary state is a flat array ``q[..., pcnst]`` mirroring
the Fortran ``q(:,:,pcnst)``. This module exposes the compile-time
constants and the ``IndexTables`` interface that maps (mode, species_slot)
to flat pcnst indices.

Compile-time constants are transcribed from the
MODAL_AERO_4MODE_MOM + RAIN_EVAP_TO_COARSE_AERO configuration of
``mam4-original-src-code/e3sm_src/modal_aero_data.F90``. Runtime index
values are captured by ``scripts/capture_reference.py --mode instrumented``
into ``tests/reference/indices/reference.npz`` (committed) and
hard-coded below as ``INDEX_TABLES``. A regression test loads the .npz and
asserts the hard-coded values match.

**Indices are 0-based here**; the Fortran reference uses 1-based pcnst
indices. The empty-slot sentinel is ``-1`` (Fortran writes ``0``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Compile-time constants for the MAM4-MOM reference configuration.
# Sources:
#   mam4-original-src-code/test_drivers/cambox_config.cpp.in
#       (-DMODAL_AERO_4MODE_MOM -DPCNST=35 -DPCOLS=1 -DPVER=1
#        -DRAIN_EVAP_TO_COARSE_AERO -DNBC=1 -DNPOA=1 -DNSOA=1)
#   mam4-original-src-code/e3sm_src/modal_aero_data.F90:13-58, 104-126
# ---------------------------------------------------------------------------

PCNST: int = 35
PCOLS: int = 1
PVER: int = 1

NTOT_AMODE: int = 4
NTOT_ASPECTYPE: int = 9
MAXD_ASPECTYPE: int = 14

NBC: int = 1
NPOA: int = 1
NSOA: int = 1
NSOAG: int = 1

# Mode names in Fortran order (modal_aero_data.F90:104-109).
# Fortran uses 1-based indices; this tuple is 0-indexed.
MODE_NAMES: tuple[str, ...] = (
    "accum",
    "aitken",
    "coarse",
    "primary_carbon",
)

# Aerosol species type names (modal_aero_data.F90:49-52, in 1-based order;
# the per-slot lspectype_amode arrays below are 0-based indices into this).
SPECNAME_AMODE: tuple[str, ...] = (
    "sulfate",
    "ammonium",
    "nitrate",
    "p-organic",
    "s-organic",
    "black-c",
    "seasalt",
    "dust",
    "m-organic",
)

# Number of species in each mode (modal_aero_data.F90:121-123).
NSPEC_AMODE: tuple[int, ...] = (7, 4, 7, 3)


# ---------------------------------------------------------------------------
# Runtime index tables — values captured from a Fortran instrumented build
# (scripts/patches/, ADR-012) and hard-coded here for the pinned MAM4-MOM
# config. Regenerable via:
#     python scripts/capture_reference.py --mode instrumented
# then compare against tests/reference/indices/reference.npz.
#
# All values are 0-based pcnst indices. -1 means "unused slot" (the
# Fortran writes 0 in those positions; we convert).
# ---------------------------------------------------------------------------

# numptr_amode[mode] -> pcnst index of that mode's number tracer.
NUMPTR_AMODE: tuple[int, ...] = (17, 22, 30, 34)

# numptrcw_amode mirrors numptr_amode at init (per modal_aero_initialize_data).
NUMPTRCW_AMODE: tuple[int, ...] = (17, 22, 30, 34)

# lspectype_amode[mode, slot] -> 0-based index into SPECNAME_AMODE for that
# (mode, slot). -1 for slots past nspec_amode[mode].
LSPECTYPE_AMODE: tuple[tuple[int, ...], ...] = (
    (0, 3, 4, 5, 7, 6, 8, -1, -1, -1, -1, -1, -1, -1),
    (0, 4, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (7, 6, 0, 5, 3, 4, 8, -1, -1, -1, -1, -1, -1, -1),
    (3, 5, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
)

# lmassptr_amode[mode, slot] -> pcnst index of mass tracer at (mode, slot).
LMASSPTR_AMODE: tuple[tuple[int, ...], ...] = (
    (10, 11, 12, 13, 14, 15, 16, -1, -1, -1, -1, -1, -1, -1),
    (18, 19, 20, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (23, 24, 25, 26, 27, 28, 29, -1, -1, -1, -1, -1, -1, -1),
    (31, 32, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
)

# Cloud-borne mirror — equal to LMASSPTR_AMODE at init time.
LMASSPTRCW_AMODE: tuple[tuple[int, ...], ...] = LMASSPTR_AMODE


# ---------------------------------------------------------------------------
# Per-species physical properties (indexed by species TYPE — 0-based into
# SPECNAME_AMODE). Values verbatim from
# rad_constituents.F90:96-103 (MODAL_AERO_4MODE_MOM branch).
# ---------------------------------------------------------------------------

#: Aerosol species density (kg/m³) by species type.
SPECDENS_AMODE: tuple[float, ...] = (
    1770.0,    # sulfate
    1770.0,    # ammonium
    1770.0,    # nitrate
    1000.0,    # p-organic
    1000.0,    # s-organic
    1700.0,    # black-c
    1900.0,    # seasalt
    2600.0,    # dust
    1601.0,    # m-organic
)

#: Volume-mean hygroscopicity by species type (dimensionless).
SPECHYGRO_AMODE: tuple[float, ...] = (
    0.507,     # sulfate
    0.507,     # ammonium
    0.507,     # nitrate
    0.010,     # p-organic
    0.140,     # s-organic
    1.0e-10,   # black-c
    1.160,     # seasalt
    0.068,     # dust
    0.100,     # m-organic
)


# ---------------------------------------------------------------------------
# Per-mode log-normal width and crystal/deliques thresholds.
# Sources:
#   - sigmag from rad_constituents.F90:170 (MODAL_AERO_4MODE/4MODE_MOM)
#   - rhcrystal/rhdeliques from rad_constituents.F90:180-181
# All four MAM4-MOM modes share rhcrystal = 0.35 and rhdeliques = 0.80.
# ---------------------------------------------------------------------------

#: Log-normal width of each mode's size distribution.
SIGMAG_AMODE: tuple[float, ...] = (
    1.800,     # accum
    1.600,     # aitken
    1.800,     # coarse
    1.600,     # primary_carbon
)

#: Reference dry diameter per mode (m). From
#: ``rad_constituents.F90:167`` (MAM4-MOM branch). Also the namelist
#: defaults written by ``scripts/capture_reference.py``.
DGNUM_AMODE:   tuple[float, ...] = (0.1100e-6, 0.0260e-6, 2.000e-6, 0.050e-6)

#: Lower / upper bounds on dry diameter per mode (m). Outside these
#: bounds ``modal_aero_calcsize_sub`` triggers its number-bounds
#: adjustment. From ``rad_constituents.F90:168-169``.
DGNUMLO_AMODE: tuple[float, ...] = (0.0535e-6, 0.0087e-6, 1.000e-6, 0.010e-6)
DGNUMHI_AMODE: tuple[float, ...] = (0.4400e-6, 0.0520e-6, 4.000e-6, 0.100e-6)

#: Crystallization relative humidity (below which aerosol is dry).
RHCRYSTAL_AMODE: tuple[float, ...] = (0.350, 0.350, 0.350, 0.350)

#: Deliquescence relative humidity (above which aerosol is fully wet).
RHDELIQUES_AMODE: tuple[float, ...] = (0.800, 0.800, 0.800, 0.800)


# ---------------------------------------------------------------------------
# Derived volume-to-number bounds per mode (used by modal_aero_calcsize_sub).
# Definitions from modal_aero_initialize_data.F90:428-435:
#
#     alnsg     = log(sigmag)
#     voltonumb = 1 / ( (pi/6) * dgnum^3   * exp(4.5 * alnsg^2) )
#     dumfac    =       (pi/6)             * exp(4.5 * alnsg^2)
#
# voltonumb / lo / hi are evaluated at dgnum / dgnumlo / dgnumhi
# respectively. voltonumb has units of 1/m^3 (it's "particles per unit
# volume" so smaller dgnum → larger voltonumb).
# ---------------------------------------------------------------------------

_SIGMAG = np.asarray(SIGMAG_AMODE, dtype=np.float64)
_DGNUM   = np.asarray(DGNUM_AMODE,   dtype=np.float64)
_DGNUMLO = np.asarray(DGNUMLO_AMODE, dtype=np.float64)
_DGNUMHI = np.asarray(DGNUMHI_AMODE, dtype=np.float64)

ALNSG_AMODE: np.ndarray = np.log(_SIGMAG)
_DUMFAC = (np.pi / 6.0) * np.exp(4.5 * ALNSG_AMODE ** 2)
DUMFAC_AMODE: np.ndarray = _DUMFAC

VOLTONUMB_AMODE:   np.ndarray = 1.0 / (_DUMFAC * _DGNUM   ** 3)
VOLTONUMBLO_AMODE: np.ndarray = 1.0 / (_DUMFAC * _DGNUMLO ** 3)
VOLTONUMBHI_AMODE: np.ndarray = 1.0 / (_DUMFAC * _DGNUMHI ** 3)


# ---------------------------------------------------------------------------
# Aitken ↔ accumulation transfer tables (used by modal_aero_calcsize_sub
# when do_aitacc_transfer is True; built once at module import time).
#
# Aitken and accum modes share some species (sulfate, s-organic, seasalt,
# m-organic in MAM4-MOM). When the Aitken mean diameter grows beyond a
# threshold, mass + number transfer to accum; when the accum mean
# diameter shrinks below a threshold, the opposite transfer happens.
# These tables encode the (aitken_pcnst_idx, accum_pcnst_idx) pairs.
# ---------------------------------------------------------------------------

#: 0-based mode indices for Aitken and accumulation in this MAM4-MOM build.
#: Fortran refers to these via modeptr_aitken / modeptr_accum.
AITKEN_MODE_IDX: int = 1   # MODE_NAMES[1] == "aitken"
ACCUM_MODE_IDX:  int = 0   # MODE_NAMES[0] == "accum"

#: 0-based mode index for primary-carbon mode in this MAM4-MOM build
#: (Fortran's ``npca``).
PCARBON_MODE_IDX: int = 3  # MODE_NAMES[3] == "primary_carbon"


# ---------------------------------------------------------------------------
# Coagulation-pair table for MAM4-MOM (Fortran amicphys init at
# modal_aero_amicphys.F90:5974-6012). The init loop iterates over 11
# possible pairs and emits the ones whose mode indices are all positive.
# For MAM4-MOM (``nait, nacc, npca`` active; ``nmait, nmacc`` absent),
# exactly 3 pairs survive:
#
#   ip=1: aitken  → accum
#   ip=2: pcarbon → accum
#   ip=3: aitken  → pcarbon   (aging path; eventually coarsens to accum)
#
# Coarse mode never participates (correct — Brownian coag negligible
# for super-µm diameters).
# ---------------------------------------------------------------------------

N_COAGPAIR: int = 3

#: Source mode index of each coag pair (0-based).
MODEFRM_COAGPAIR: tuple[int, ...] = (
    AITKEN_MODE_IDX,   # ip 1: aitken → accum
    PCARBON_MODE_IDX,  # ip 2: pcarbon → accum
    AITKEN_MODE_IDX,   # ip 3: aitken → pcarbon
)

#: Destination mode index of each coag pair (0-based).
MODETOO_COAGPAIR: tuple[int, ...] = (
    ACCUM_MODE_IDX,    # ip 1
    ACCUM_MODE_IDX,    # ip 2
    PCARBON_MODE_IDX,  # ip 3
)


def _build_aitacc_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match Aitken-mode slots against accum-mode slots by species type.

    Returns 1-D arrays of pcnst indices ``(lsfrma, lstooa, lsfrmc, lstooc)``
    of length ``1 + n_matched`` where index 0 is the number-tracer pair
    and indices 1..n are mass-species pairs.

    The species ordering matches the Fortran's iq=1 → number, iq>1 →
    mass-species convention used in modal_aero_calcsize_init.
    """
    ait_types = LSPECTYPE_AMODE[AITKEN_MODE_IDX]
    acc_types = LSPECTYPE_AMODE[ACCUM_MODE_IDX]
    ait_mass  = LMASSPTR_AMODE[AITKEN_MODE_IDX]
    acc_mass  = LMASSPTR_AMODE[ACCUM_MODE_IDX]
    ait_cw    = LMASSPTRCW_AMODE[AITKEN_MODE_IDX]
    acc_cw    = LMASSPTRCW_AMODE[ACCUM_MODE_IDX]

    pairs_a: list[tuple[int, int]] = []
    pairs_c: list[tuple[int, int]] = []
    # iq=1 — number tracer pair (Fortran convention: numbers come first).
    pairs_a.append((int(NUMPTR_AMODE[AITKEN_MODE_IDX]),
                    int(NUMPTR_AMODE[ACCUM_MODE_IDX])))
    pairs_c.append((int(NUMPTRCW_AMODE[AITKEN_MODE_IDX]),
                    int(NUMPTRCW_AMODE[ACCUM_MODE_IDX])))
    # iq>1 — mass-species pairs, in Aitken slot order.
    for ait_slot, ait_type in enumerate(ait_types):
        if ait_type < 0:
            continue
        for acc_slot, acc_type in enumerate(acc_types):
            if acc_type == ait_type:
                pairs_a.append((int(ait_mass[ait_slot]), int(acc_mass[acc_slot])))
                pairs_c.append((int(ait_cw[ait_slot]),   int(acc_cw[acc_slot])))
                break

    pa = np.asarray(pairs_a, dtype=np.int32)
    pc = np.asarray(pairs_c, dtype=np.int32)
    return pa[:, 0], pa[:, 1], pc[:, 0], pc[:, 1]


_LSFRMA, _LSTOOA, _LSFRMC, _LSTOOC = _build_aitacc_pairs()

#: pcnst indices of Aitken-mode species that transfer to accum (interstitial).
#: Index 0 is the number tracer; subsequent indices are mass species in
#: Aitken-slot order.
LSPECFRMA_CSIZXF: np.ndarray = _LSFRMA

#: pcnst indices of the matching accum species (interstitial).
LSPECTOOA_CSIZXF: np.ndarray = _LSTOOA

#: Cloud-borne counterparts.
LSPECFRMC_CSIZXF: np.ndarray = _LSFRMC
LSPECTOOC_CSIZXF: np.ndarray = _LSTOOC

#: True for accum-mode slots whose species type is NOT present in Aitken
#: (e.g., p-organic, black-c, dust). Their mass is excluded from the
#: drv/num totals when computing accum→Aitken transfer rates.
def _build_noxf_acc2ait() -> np.ndarray:
    ait_types_set = {t for t in LSPECTYPE_AMODE[AITKEN_MODE_IDX] if t >= 0}
    acc_types = LSPECTYPE_AMODE[ACCUM_MODE_IDX]
    mask = np.zeros(MAXD_ASPECTYPE, dtype=bool)
    for s, t in enumerate(acc_types):
        if t < 0:
            continue
        mask[s] = (t not in ait_types_set)
    return mask


NOXF_ACC2AIT: np.ndarray = _build_noxf_acc2ait()

#: Geometric mean v2n between Aitken and accum reference voltonumb.
#: Fortran line 1003: ``v2nzz = sqrt(voltonumb_amode(nait) * voltonumb_amode(nacc))``.
V2NZZ_AIT_ACC: float = float(
    np.sqrt(VOLTONUMB_AMODE[AITKEN_MODE_IDX] * VOLTONUMB_AMODE[ACCUM_MODE_IDX])
)


# ---------------------------------------------------------------------------
# Pre-computed per-(mode, slot) lookup tables for vectorized use.
# PER_SLOT_DENSITY[mode, slot] = SPECDENS_AMODE[LSPECTYPE_AMODE[mode, slot]]
# for valid slots; 1.0 (a harmless default) for unused slots so per-mode
# accumulations sum to 0 when q[unused_slot] == 0.
# ---------------------------------------------------------------------------

def _build_per_slot_table(values: tuple[float, ...], default: float) -> np.ndarray:
    table = np.full((NTOT_AMODE, MAXD_ASPECTYPE), default, dtype=np.float64)
    for m in range(NTOT_AMODE):
        for s, type_idx in enumerate(LSPECTYPE_AMODE[m]):
            if type_idx >= 0:
                table[m, s] = values[type_idx]
    return table


PER_SLOT_DENSITY: np.ndarray = _build_per_slot_table(SPECDENS_AMODE, default=1.0)
PER_SLOT_HYGRO:   np.ndarray = _build_per_slot_table(SPECHYGRO_AMODE, default=0.0)

#: For each (mode, slot), is that slot used (True) or unused (False).
SLOT_VALID: np.ndarray = (
    np.asarray(LSPECTYPE_AMODE, dtype=np.int32) >= 0
)


# ---------------------------------------------------------------------------
# amicphys-internal mapping & conversion tables (M3.6 PR-C foundation).
#
# Fortran ``modal_aero_amicphys.F90`` orders aerosol species by its own
# ``name_aerpfx`` list (set up in ``modal_aero_amicphys_init``), which
# is *different* from ``modal_aero_data``'s ``lmassptr_amode`` ordering.
# The unpacking ``q[pcnst]`` → ``(qgas, qaer, qnum, qwtr)`` uses these
# amicphys-internal tables, not the modal_aero_data ones.
#
# Values captured via ``scripts/patches/amicphys_init_dump.patch``.
# Parity test: ``tests/test_scaffolding.py``.
# ---------------------------------------------------------------------------

#: Number of gases tracked by amicphys (SOAG + H2SO4 for MAM4-MOM).
AMICPHYS_NGAS: int = 2
#: Number of aerosol species tracked by amicphys (soa, so4, ..., 7 in MAM4-MOM).
AMICPHYS_NAER: int = 7
#: ``max_gas`` / ``max_aer`` compile-time bounds from amicphys.
AMICPHYS_MAX_GAS: int = 2
AMICPHYS_MAX_AER: int = 7

#: 0-based pcnst-absolute indices of each gas in ``q``. Derived from the
#: Fortran ``lmap_gas + loffset - 1``.
LMAP_GAS:   np.ndarray = np.asarray([9, 6], dtype=np.int32)
#: 0-based pcnst-absolute indices of each mode's interstitial number tracer.
#: Always matches ``NUMPTR_AMODE``; we keep a separate name for symmetry with
#: the other ``LMAP_*`` tables that come from the same amicphys dump.
LMAP_NUM:   np.ndarray = np.asarray([17, 22, 30, 34], dtype=np.int32)
LMAP_NUMCW: np.ndarray = LMAP_NUM.copy()
#: 0-based pcnst-absolute indices of (mode, amicphys-iaer) interstitial mass
#: tracers. Sentinel ``-1`` for species absent from that mode.
#: Row order: (accum, aitken, coarse, primary_carbon). Column order is
#: amicphys's internal iaer ordering (soa, so4, ...).
LMAP_AER: np.ndarray = np.asarray(
    [[12, 10, 11, 13, 15, 14, 16],
     [19, 18, -1, -1, 20, -1, 21],
     [28, 25, 27, 26, 24, 23, 29],
     [-1, -1, 31, 32, -1, -1, 33]],
    dtype=np.int32,
)
LMAP_AERCW: np.ndarray = LMAP_AER.copy()

#: Unit-conversion factors (kg/kg → mol/mol or kg/kg → #/kmol). Applied by
#: the state-dict → amicphys-view unpacking.
FCVT_GAS: np.ndarray = np.asarray([0.0800733333333333, 1.0], dtype=np.float64)
FCVT_AER: np.ndarray = np.asarray(
    [0.08, 1.0, 0.08, 1.0, 1.0, 1.0, 1.0], dtype=np.float64,
)
FCVT_NUM: float = 1.0
FCVT_WTR: float = 1.607793072824157

# ---------------------------------------------------------------------------
# Driver-level mmr ↔ vmr conversion (driver.F90:1217-1228).
#
# Before calling ``modal_aero_amicphys_intr``, the Fortran driver converts
# each constituent from mass mixing ratio to volume mixing ratio via
# ``vmr(l2) = mmr(l2) * mwdry / adv_mass(l2)``. The amicphys-internal
# ``fcvt_*`` factors are then applied to the *vmr* values inside
# ``mam_amicphys_1gridcell``. For consistency with the captured
# amicphys-local reference data, the JAX unpacking must apply both
# stages.
#
# imozart = 6 (1-based) in this build → loffset = 5, gas_pcnst = 30.
# adv_mass[i] is the molecular weight of pcnst tracer (i + loffset + 1, 1-based).
# Number tracers have adv_mass ≈ 1.0074 (a convention that makes the
# mwdry/adv_mass factor ≈ 28.75 — converts particles/kmol-air to
# something proportional to a volume mixing ratio).
# ---------------------------------------------------------------------------

#: Molar diffusion volume of dry air (unitless). Used by ``gas_diffusivity``
#: (modal_aero_amicphys.F90:5302-5316). Source: ``physconst.F90:80``.
VMDRY: float = 20.1

#: Gas-phase properties indexed by amicphys's igas (1..AMICPHYS_NGAS).
#: Order matches ``name_gas``: [0]=SOA, [1]=H2SO4 for MAM4-MOM.
#: Captured via the amicphys init dump (M3.6 PR-D).
MW_GAS:         np.ndarray = np.asarray([150.0,        98.0784], dtype=np.float64)
VOL_MOLAR_GAS:  np.ndarray = np.asarray([65.63265306122449, 42.88], dtype=np.float64)
ACCOM_COEF_GAS: np.ndarray = np.asarray([0.65,         0.65   ], dtype=np.float64)

MWDRY: float = 28.966
ADV_MASS: np.ndarray = np.asarray([
    34.0136, 98.0784, 64.0648, 62.1324, 12.011,                # 0-4: O, H2SO4, SO2, DMS, C
    115.10734, 12.011, 12.011, 12.011, 135.064039, 58.442468,  # 5-10: soa, ...
    250092.672, 1.0074, 115.10734, 12.011, 58.442468,          # 11-15
    250092.672, 1.0074, 135.064039, 58.442468, 115.10734,      # 16-20
    12.011, 12.011, 12.011, 250092.672, 1.0074,                # 21-25
    12.011, 12.011, 250092.672, 1.0074,                        # 26-29
], dtype=np.float64)
assert ADV_MASS.shape == (30,), "ADV_MASS must match gas_pcnst=30"

#: per-pcnst mmr → vmr factor. Length PCNST=35; entries before imozart-1
#: (the chemistry offset) are 1.0 since those constituents aren't part of
#: the chemistry vmr conversion.
#:
#: We store ``MMR_TO_VMR`` and ``VMR_TO_MMR`` as **two independently
#: computed** arrays (``mwdry/adv_mass`` and ``adv_mass/mwdry``) rather
#: than deriving ``VMR_TO_MMR = 1 / MMR_TO_VMR`` so the JAX round-trip
#: drift matches the Fortran driver's (driver.F90:1224 vs :1321) at ULP
#: level — important for bit-comparable tests on tracers amicphys
#: doesn't touch.
_LOFFSET = 5
_MMR_TO_VMR = np.ones(PCNST, dtype=np.float64)
_VMR_TO_MMR = np.ones(PCNST, dtype=np.float64)
_MMR_TO_VMR[_LOFFSET:] = MWDRY / ADV_MASS
_VMR_TO_MMR[_LOFFSET:] = ADV_MASS / MWDRY
MMR_TO_VMR: np.ndarray = _MMR_TO_VMR
VMR_TO_MMR: np.ndarray = _VMR_TO_MMR

#: Mass→volume conversion per amicphys species (m³-AP / kmol-AP).
#: Distinct from FCVT_AER (which is the kg/kg ↔ mol/mol unit conversion).
#: ``fac_m2v_aer = mw_aer / dens_aer`` in the Fortran amicphys init code.
#: Consumed by rename's dryvol summation (and by gasaerexch/coag/newnuc
#: when they land). Values match the per-record capture in
#: ``tests/reference/per_process/rename_{before,after}.npz``; parity test
#: in ``tests/test_scaffolding.py``.
FAC_M2V_AER: np.ndarray = np.asarray(
    [0.15, 0.06497175141242938, 0.15, 0.007058823529411765,
     0.030789473684210526, 0.051923076923076926, 156.20986883198],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# SOA-specific amicphys init constants (M3.6 PR-E).
# ---------------------------------------------------------------------------

#: Number of primary / secondary organic-aerosol species in amicphys.
AMICPHYS_NPOA: int = 1
AMICPHYS_NSOA: int = 1

#: 0-based amicphys-internal iaer indices for POM and SOA aerosol species
#: (Fortran's iaer_pom, iaer_soa are 1-based; subtract 1).
AMICPHYS_IAER_POM: int = 2
AMICPHYS_IAER_SOA: int = 0

#: 0-based mode index for primary-carbon (Fortran npca, 1-based 4).
AMICPHYS_NPCA: int = 3

#: 0-based mode index for the "ultrafine" mode used as a soaexch
#: exclusion (Fortran nufi). Set to ``-1`` when absent (MAM4-MOM doesn't
#: have an ultrafine mode; the Fortran sentinel is ``-999888777``).
AMICPHYS_NUFI: int = -1

#: Per-mode "aging" flag. 1 means the mode participates in soaexch via
#: the aging path even without a direct lptr2_soa_a_amode entry.
MODE_AGING_OPTAA: np.ndarray = np.asarray([0, 0, 0, 1], dtype=np.int32)

#: Per-(mode, nsoa) flag: True if that mode has a secondary-SOA species
#: pcnst slot. Derived from Fortran's ``lptr2_soa_a_amode > 0`` check —
#: soaexch only uses the boolean, not the actual pcnst index.
LPTR2_SOA_A_AMODE_PRESENT: np.ndarray = np.asarray(
    [[True], [True], [True], [False]], dtype=bool,
)

#: Host-code molecular weight of sulfate aerosol (g/mol). Set in
#: ``modal_aero_amicphys_init`` from ``mwhost_aer(iaer_so4)``. For
#: MAM4-MOM the host treats sulfate as ammonium bisulfate (mw = 115)
#: even though the actual sulfuric-acid mw is 96 — newnuc's dispatcher
#: applies a ``voldry_clus * (mw_so4a_host / mw_so4a)`` correction
#: (modal_aero_newnuc.F90:874).
MW_SO4A_HOST: float = 115.0
#: Host-code molecular weight of ammonium aerosol (g/mol). For
#: MAM4-MOM ``iaer_nh4 <= 0``, so the Fortran falls back to
#: ``mw_nh4a_host = mw_so4a_host``.
MW_NH4A_HOST: float = 115.0
#: Host-code dry density of sulfate aerosol (kg/m³). Matches
#: ``dens_aer(iaer_so4)`` in amicphys init.
DENS_SO4A_HOST: float = 1770.0


@dataclass(frozen=True)
class IndexTables:
    """0-based pcnst index tables for the MAM4 tracer array.

    Mirrors the integer arrays in
    ``mam4-original-src-code/e3sm_src/modal_aero_data.F90:180-185``,
    converted to 0-based and rotated so the leading axis is mode (it's the
    outer loop in the Fortran dump, and the more natural Python ordering).

      * ``numptr_amode``: shape ``(ntot_amode,)``
      * ``lmassptr_amode``: shape ``(ntot_amode, maxd_aspectype)``
      * ``numptrcw_amode``, ``lmassptrcw_amode``: cloud-borne mirrors.

    Sentinel value ``-1`` means "unused slot" (only the first
    ``NSPEC_AMODE[mode]`` slots of each mode are valid).
    """

    numptr_amode: np.ndarray
    lmassptr_amode: np.ndarray
    numptrcw_amode: np.ndarray
    lmassptrcw_amode: np.ndarray


def make_sentinel_tables() -> IndexTables:
    """Construct an ``IndexTables`` filled with the -1 sentinel.

    Kept for tests that exercise the "table not populated" failure mode.
    Production code should use :data:`INDEX_TABLES` instead.
    """
    return IndexTables(
        numptr_amode=np.full(NTOT_AMODE, -1, dtype=np.int32),
        lmassptr_amode=np.full((NTOT_AMODE, MAXD_ASPECTYPE), -1, dtype=np.int32),
        numptrcw_amode=np.full(NTOT_AMODE, -1, dtype=np.int32),
        lmassptrcw_amode=np.full((NTOT_AMODE, MAXD_ASPECTYPE), -1, dtype=np.int32),
    )


def _make_index_tables() -> IndexTables:
    return IndexTables(
        numptr_amode=np.asarray(NUMPTR_AMODE, dtype=np.int32),
        lmassptr_amode=np.asarray(LMASSPTR_AMODE, dtype=np.int32),
        numptrcw_amode=np.asarray(NUMPTRCW_AMODE, dtype=np.int32),
        lmassptrcw_amode=np.asarray(LMASSPTRCW_AMODE, dtype=np.int32),
    )


#: Canonical IndexTables for the MAM4-MOM reference configuration. Use this
#: rather than constructing a fresh one — equality with this instance is the
#: simplest sanity check.
INDEX_TABLES: IndexTables = _make_index_tables()


# ---------------------------------------------------------------------------
# Accessors. All slicing along the last axis of ``q``.
# ---------------------------------------------------------------------------

def _resolve_idx(idx: int, label: str) -> int:
    if idx < 0:
        raise NotImplementedError(
            f"{label}: index is -1 (sentinel). Either the IndexTables haven't "
            "been populated, or this (mode, slot) is unused."
        )
    return idx


def get_number(q, mode: int, tables: IndexTables = INDEX_TABLES):
    """Return the number-mixing-ratio slice along the last axis for ``mode``."""
    idx = _resolve_idx(int(tables.numptr_amode[mode]), "numptr_amode")
    return q[..., idx]


def get_mass(q, mode: int, species_slot: int,
             tables: IndexTables = INDEX_TABLES):
    """Return the mass-mixing-ratio slice for (mode, species_slot)."""
    idx = _resolve_idx(int(tables.lmassptr_amode[mode, species_slot]),
                       "lmassptr_amode")
    return q[..., idx]


def get_mass_by_species_name(q, mode: int, species_name: str,
                             tables: IndexTables = INDEX_TABLES):
    """Return the mass-mixing-ratio slice for a named species in ``mode``.

    Searches LSPECTYPE_AMODE for the species; raises KeyError if the
    species is not present in that mode.
    """
    try:
        species_type_idx = SPECNAME_AMODE.index(species_name)
    except ValueError as exc:
        raise KeyError(
            f"unknown species {species_name!r}; valid: {SPECNAME_AMODE}"
        ) from exc

    slots = LSPECTYPE_AMODE[mode]
    for slot, stype in enumerate(slots):
        if stype == species_type_idx:
            return get_mass(q, mode, slot, tables=tables)
    raise KeyError(
        f"species {species_name!r} is not present in mode "
        f"{MODE_NAMES[mode]!r}"
    )


# ---------------------------------------------------------------------------
# M8 (cloud chemistry): gas pcnst slots + cloud-borne species index tables.
#
# These constants support ``mam4_jax/processes/cloudchem.py`` which mirrors
# Fortran's ``box_model_utils/cloudchem_simple.F90``. Cloudchem operates on
# the *vmr* arrays (volume mixing ratios with ``gas_pcnst=30`` third-dim for
# MAM4-MOM), not the *q* arrays (mass mixing ratios with ``pcnst=35``).
#
# Reference values captured by the extended ``mam4_dump_state::dump_indices``
# in PR-K1 (gas_pcnst_indices section of ``mam4_indices.txt``). For MAM4-MOM:
#   h2so4 = 7 (1-based pcnst) → 6 (0-based pcnst)
#   so2   = 8 → 7
#   nh3   = -1 (absent — not in cnst registry for this config)
#   hcl   = -1 (absent)
#   hno3  = -1 (absent)
#   soag  = 10 → 9
# ---------------------------------------------------------------------------

#: Mode index for the coarse mode (``MODE_NAMES[2] == "coarse"``).
COARSE_MODE_IDX: int = 2

#: 0-based pcnst slot of H2SO4 in ``q``. Also at ``LMAP_GAS[1]``.
PCNST_H2SO4_GAS: int = 6
#: 0-based pcnst slot of SO2 in ``q``. **Not** in ``LMAP_GAS`` (amicphys
#: does not track SO2 — only cloudchem reads it via a separate
#: ``cnst_get_ind`` call).
PCNST_SO2_GAS: int = 7
#: 0-based pcnst slot of NH3 in ``q``. ``-1`` = absent in MAM4-MOM
#: (``cnst_get_ind('NH3', ...)`` returns -1 here). Cloudchem's NH3 → NH4
#: branch is structurally dead for this config.
PCNST_NH3_GAS: int = -1
#: 0-based pcnst slot of SOAG (gas-phase SOA precursor) in ``q``. Also at
#: ``LMAP_GAS[0]``.
PCNST_SOAG_GAS: int = 9

#: amicphys ``loffset`` — pcnst → vmr/amicphys-internal slot offset.
#: ``vmr_slot = pcnst_slot - AMICPHYS_LOFFSET`` (0-based on both sides).
#: Captured by ``mam4_dump_state::dump_amicphys_init`` (amicphys_loffset
#: field of ``tests/reference/indices/reference.npz``).
AMICPHYS_LOFFSET: int = 5


def _to_vmr_slot(pcnst_slot: int) -> int:
    """Translate a 0-based pcnst slot to a 0-based vmr/gas_pcnst slot.

    ``-1`` (absent) maps to ``-1``. The conversion is just a shift by
    ``AMICPHYS_LOFFSET``.
    """
    return pcnst_slot - AMICPHYS_LOFFSET if pcnst_slot >= 0 else -1


def _lookup_cw_amode(species_name: str) -> tuple[int, ...]:
    """Return the per-mode 0-based pcnst slot of cloud-borne ``species_name``.

    Derives from ``LMASSPTRCW_AMODE`` + ``LSPECTYPE_AMODE`` + the species
    name's index into ``SPECNAME_AMODE``. Modes that do not carry the
    species return ``-1``.
    """
    type_idx = SPECNAME_AMODE.index(species_name)
    out: list[int] = []
    for m in range(NTOT_AMODE):
        type_row = LSPECTYPE_AMODE[m]
        if type_idx in type_row:
            slot = type_row.index(type_idx)
            out.append(LMASSPTRCW_AMODE[m][slot])
        else:
            out.append(-1)
    return tuple(out)


#: Per-mode pcnst slot of cloud-borne sulfate. For MAM4-MOM:
#: ``(10, 18, 25, -1)`` — accum / aitken / coarse carry sulfate; primary_carbon
#: doesn't. Cloudchem_simple_sub only deposits into accum (mode 0) and
#: aitken (mode 1); coarse stays unchanged but is read by other processes.
LPTR_SO4_CW_AMODE: tuple[int, ...] = _lookup_cw_amode("sulfate")

#: Per-mode pcnst slot of cloud-borne ammonium. All ``-1`` in MAM4-MOM
#: (no mode carries ammonium in this config; consistent with NH3 absent
#: from the constituent registry).
LPTR_NH4_CW_AMODE: tuple[int, ...] = _lookup_cw_amode("ammonium")


# vmr-space (gas_pcnst third-dim) slots used by cloudchem:

#: vmr slot of H2SO4 gas.
VMR_H2SO4: int = _to_vmr_slot(PCNST_H2SO4_GAS)
#: vmr slot of SO2 gas.
VMR_SO2:   int = _to_vmr_slot(PCNST_SO2_GAS)
#: vmr slot of NH3 gas. ``-1`` in MAM4-MOM (NH3 absent). Defined for
#: forward-compatibility: if a future MAM4 config registers NH3, the
#: cloudchem NH3 → NH4 branch (currently omitted) can be enabled by
#: gating on ``VMR_NH3 >= 0`` without a data.py change. Not consumed
#: by any current call site.
VMR_NH3:   int = _to_vmr_slot(PCNST_NH3_GAS)
#: vmr slot of SOAG gas.
VMR_SOAG:  int = _to_vmr_slot(PCNST_SOAG_GAS)

# Cross-check: PCNST_H2SO4_GAS and PCNST_SOAG_GAS duplicate LMAP_GAS's
# entries (amicphys's gas list). If a future MAM4 config shifts either,
# both paths must update. Module-load assertion catches drift early.
assert int(LMAP_GAS[0]) == PCNST_SOAG_GAS, \
    f"LMAP_GAS[0] = {int(LMAP_GAS[0])} drifted from PCNST_SOAG_GAS = {PCNST_SOAG_GAS}"
assert int(LMAP_GAS[1]) == PCNST_H2SO4_GAS, \
    f"LMAP_GAS[1] = {int(LMAP_GAS[1])} drifted from PCNST_H2SO4_GAS = {PCNST_H2SO4_GAS}"

#: Per-mode vmrcw slot of cloud-borne aerosol number.
VMRCW_NUM: tuple[int, ...] = tuple(_to_vmr_slot(s) for s in NUMPTRCW_AMODE)
#: Per-mode vmrcw slot of cloud-borne sulfate.
VMRCW_SO4: tuple[int, ...] = tuple(_to_vmr_slot(s) for s in LPTR_SO4_CW_AMODE)
#: Per-mode vmrcw slot of cloud-borne ammonium (all -1 in MAM4-MOM).
VMRCW_NH4: tuple[int, ...] = tuple(_to_vmr_slot(s) for s in LPTR_NH4_CW_AMODE)
