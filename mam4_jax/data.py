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
