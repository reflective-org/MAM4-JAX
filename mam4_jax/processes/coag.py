"""Coagulation stub (port of ``modal_aero_coag``).

Port target: ``mam4-original-src-code/e3sm_src/modal_aero_coag.F90``.
Implements intra- and inter-modal Brownian coagulation.
"""
from __future__ import annotations

from typing import Any


def coag(state: Any, params: Any, config: Any) -> Any:
    """Apply intra- and inter-modal Brownian coagulation.

    Not yet implemented — see ``docs/PLANS.md`` Milestone 3.
    """
    raise NotImplementedError(
        "coag: pending port from e3sm_src/modal_aero_coag.F90."
    )
