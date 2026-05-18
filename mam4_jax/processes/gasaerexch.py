"""Gas–aerosol exchange stub (port of ``modal_aero_gasaerexch``).

Port target: ``mam4-original-src-code/e3sm_src/modal_aero_gasaerexch.F90``.
Handles H2SO4 / SOAG condensation onto modes.
"""
from __future__ import annotations

from typing import Any


def gasaerexch(state: Any, params: Any, config: Any) -> Any:
    """Condense H2SO4 and SOAG onto aerosol modes.

    Not yet implemented — see ``docs/PLANS.md`` Milestone 3.
    """
    raise NotImplementedError(
        "gasaerexch: pending port from e3sm_src/modal_aero_gasaerexch.F90."
    )
