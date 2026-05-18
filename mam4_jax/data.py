"""Mode/species constants and tracer-index bookkeeping for MAM4-MOM.

Per the tracer-representation ADR, the primary state is a flat array
``q[..., pcnst]`` mirroring the Fortran ``q(:,:,pcnst)``. This module
exposes the compile-time constants and the ``IndexTables`` interface that
maps (mode, species_slot) to flat pcnst indices.

Compile-time constants are transcribed from the
MODAL_AERO_4MODE_MOM + RAIN_EVAP_TO_COARSE_AERO configuration of
``mam4-original-src-code/e3sm_src/modal_aero_data.F90`` and the build
flags in ``mam4-original-src-code/test_drivers/cambox_config.cpp.in``.

Runtime index values (``numptr_amode``, ``lmassptr_amode``, etc.) are
populated by Fortran name-lookup at initialization
(``modal_aero_initialize_data.F90:250-309``). In this scaffold the
tables are sentinel-filled (-1); they will be populated from
instrumented Fortran reference data in Milestone 2.
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

# Aerosol species type names (modal_aero_data.F90:49-52).
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

# Number of species in each mode for MAM4-MOM with RAIN_EVAP_TO_COARSE_AERO
# (modal_aero_data.F90:121-123).
NSPEC_AMODE: tuple[int, ...] = (7, 4, 7, 3)


# ---------------------------------------------------------------------------
# Index tables.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexTables:
    """Mapping from (species_slot, mode) to flat pcnst index.

    Mirrors the Fortran integer arrays in
    ``modal_aero_data.F90:180-185``:

      * ``numptr_amode(ntot_amode)``: pcnst index of the number tracer
        for each mode.
      * ``lmassptr_amode(maxd_aspectype, ntot_amode)``: pcnst index of
        the mass tracer for (species_slot, mode).
      * ``numptrcw_amode``, ``lmassptrcw_amode``: cloud-borne mirrors.

    Sentinel value -1 means "not yet populated".
    """

    numptr_amode: np.ndarray
    lmassptr_amode: np.ndarray
    numptrcw_amode: np.ndarray
    lmassptrcw_amode: np.ndarray


def make_sentinel_tables() -> IndexTables:
    """Construct an ``IndexTables`` filled with the -1 sentinel.

    Used by the scaffold until M2 reference data is available; accessor
    helpers raise ``NotImplementedError`` against sentinel-filled tables.
    """
    return IndexTables(
        numptr_amode=np.full(NTOT_AMODE, -1, dtype=np.int32),
        lmassptr_amode=np.full((MAXD_ASPECTYPE, NTOT_AMODE), -1, dtype=np.int32),
        numptrcw_amode=np.full(NTOT_AMODE, -1, dtype=np.int32),
        lmassptrcw_amode=np.full((MAXD_ASPECTYPE, NTOT_AMODE), -1, dtype=np.int32),
    )


def get_number(q, mode: int, tables: IndexTables):
    """Return the number-mixing-ratio slice along the last axis for ``mode``."""
    idx = int(tables.numptr_amode[mode])
    if idx < 0:
        raise NotImplementedError(
            "IndexTables.numptr_amode is sentinel-filled; "
            "requires M2 reference-data population."
        )
    return q[..., idx]


def get_mass(q, mode: int, species_slot: int, tables: IndexTables):
    """Return the mass-mixing-ratio slice along the last axis for (mode, species_slot)."""
    idx = int(tables.lmassptr_amode[species_slot, mode])
    if idx < 0:
        raise NotImplementedError(
            "IndexTables.lmassptr_amode is sentinel-filled; "
            "requires M2 reference-data population."
        )
    return q[..., idx]
