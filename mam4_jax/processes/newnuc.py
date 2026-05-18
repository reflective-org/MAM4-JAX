"""New-particle nucleation stub (port of ``modal_aero_newnuc``).

Port target: ``mam4-original-src-code/e3sm_src/modal_aero_newnuc.F90``.
Implements binary H2SO4–H2O nucleation (Vehkamäki-style).
"""
from __future__ import annotations

from typing import Any


def newnuc(state: Any, params: Any, config: Any) -> Any:
    """Form new aerosol particles via binary H2SO4–H2O nucleation.

    Not yet implemented — see ``docs/PLANS.md`` Milestone 3.
    """
    raise NotImplementedError(
        "newnuc: pending port from e3sm_src/modal_aero_newnuc.F90."
    )
