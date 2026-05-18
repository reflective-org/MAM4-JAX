"""Mode-transfer (rename) stub (port of ``modal_aero_rename``).

Port target: ``mam4-original-src-code/e3sm_src/modal_aero_rename.F90``.
Transfers aged Aitken-mode particles into the accumulation mode when
size criteria are met.
"""
from __future__ import annotations

from typing import Any


def rename(state: Any, params: Any, config: Any) -> Any:
    """Transfer aged Aitken particles to the accumulation mode.

    Not yet implemented — see ``docs/PLANS.md`` Milestone 3.
    """
    raise NotImplementedError(
        "rename: pending port from e3sm_src/modal_aero_rename.F90."
    )
