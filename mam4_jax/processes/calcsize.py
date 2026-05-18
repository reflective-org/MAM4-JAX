"""Size redistribution stub (port of ``modal_aero_calcsize``).

Port target: ``mam4-original-src-code/box_model_utils/modal_aero_calcsize.F90``
(entry: ``modal_aero_calcsize_sub``).
"""
from __future__ import annotations

from typing import Any


def calcsize(state: Any, params: Any, config: Any) -> Any:
    """Recompute dry diameters and transfer particles between modes.

    Not yet implemented — see ``docs/PLANS.md`` Milestone 3.
    """
    raise NotImplementedError(
        "calcsize: pending port from box_model_utils/modal_aero_calcsize.F90."
    )
