"""Equilibrium water-uptake stub (port of ``modal_aero_wateruptake``).

Port target: ``mam4-original-src-code/e3sm_src_modified/modal_aero_wateruptake.F90``
(entry: ``modal_aero_wateruptake_dr``).
"""
from __future__ import annotations

from typing import Any


def wateruptake(state: Any, params: Any, config: Any) -> Any:
    """Compute wet diameter, wet density, aerosol water from equilibrium Köhler.

    Not yet implemented — see ``docs/PLANS.md`` Milestone 3.
    """
    raise NotImplementedError(
        "wateruptake: pending port from "
        "e3sm_src_modified/modal_aero_wateruptake.F90."
    )
