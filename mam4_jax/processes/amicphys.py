"""Umbrella microphysics-orchestrator stub (port of ``modal_aero_amicphys``).

Port target: ``mam4-original-src-code/e3sm_src_modified/modal_aero_amicphys.F90``.
Composes ``gasaerexch``, ``newnuc``, ``coag``, and ``rename`` in the
Fortran sub-step order, gated by ``ControlConfig`` toggles.
"""
from __future__ import annotations

from typing import Any


def amicphys(state: Any, params: Any, config: Any) -> Any:
    """Apply gas–aerosol exchange, nucleation, coagulation, and rename in order.

    Not yet implemented — see ``docs/PLANS.md`` Milestone 3 / Milestone 4.
    """
    raise NotImplementedError(
        "amicphys: pending port from "
        "e3sm_src_modified/modal_aero_amicphys.F90."
    )
