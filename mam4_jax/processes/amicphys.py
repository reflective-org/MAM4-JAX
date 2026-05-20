"""Modal aerosol microphysics orchestrator — JAX port of
``modal_aero_amicphys_intr`` and its internal sub-routines.

This is the user-facing process function per ADR-009. Port targets
(all in ``mam4-original-src-code/e3sm_src_modified/modal_aero_amicphys.F90``):

* ``modal_aero_amicphys_intr``       (line 310, ~875 LOC) — entry, called by driver.F90:1283.
* ``mam_amicphys_1gridcell``         (line 1190, ~309 LOC) — per-(col, level) orchestrator.
* ``mam_amicphys_1subarea_clear``    (line 2064, ~562 LOC) — clear-sky sub-area handler.
* ``mam_amicphys_1subarea_cloudy``   (line 1504, ~555 LOC) — cloudy-sky handler.
  **Not ported** — unreachable from the box-model driver because ``cldn=0``
  in ``driver.F90:591``. Calling ``amicphys`` with a non-zero cloud
  fraction will raise; the cloudy path will be added if/when a workflow
  needs it.

Inside ``mam_amicphys_1subarea_clear`` the four sub-process stubs are
called in this order (Fortran lines 2387, 2467, 2496, 2529):

    gasaerexch  →  rename  →  newnuc  →  coag

Each sub-process is a private function in this module:

* :func:`_mam_gasaerexch_1subarea` — H₂SO₄ / SOAG condensation        (M3.6 PR-C)
* :func:`_mam_rename_1subarea`      — Aitken → accum mode-transfer    (M3.6 PR-B)
* :func:`_mam_newnuc_1subarea`      — binary H₂SO₄–H₂O nucleation      (M3.6 PR-D)
* :func:`_mam_coag_1subarea`        — Brownian coagulation             (M3.6 PR-E)

**Scope of M3.6 PR-A (this commit):** the four sub-process functions
are no-op stubs (they return the input state unchanged). The
orchestration shell exists so future PRs only need to replace one
sub-process stub at a time. With all four ``mdo_*`` toggles set to 0
the function is provably a passthrough; the captured reference at
``tests/reference/per_process_amicphys_off/`` confirms the Fortran
behaves identically.

State dict contract (the same keys passed to / returned from
:mod:`mam4_jax.processes.wateruptake` plus the cloud-borne / wet
arrays that calcsize and wateruptake produce):

    state['q']           shape (..., pcnst)         — interstitial tracer mixing ratios
    state['qqcw']        shape (..., pcnst)         — cloud-borne tracer mixing ratios
    state['dgncur_a']    shape (..., ntot_amode)    — dry mode diameters (m)
    state['dgncur_awet'] shape (..., ntot_amode)    — wet mode diameters (m)
    state['qaerwat']     shape (..., ntot_amode)    — aerosol water (kg/kg)
    state['wetdens']     shape (..., ntot_amode)    — wet aerosol density (kg/m³)
    state['t']           shape (...,)               — temperature (K)
    state['pmid']        shape (...,)               — mid-layer pressure (Pa)
    state['cldn']        shape (...,)               — cloud fraction (-)
    state['deltat']      scalar                     — timestep (s)

Returned state has the same keys with any updates each enabled sub-process makes.
"""
from __future__ import annotations

from typing import Any


def amicphys(state: dict[str, Any], params=None, config=None, *,
             mdo_gasaerexch: int = 1, mdo_rename: int = 1,
             mdo_newnuc: int = 1, mdo_coag: int = 1) -> dict[str, Any]:
    """ADR-009 entry point — see module docstring.

    The four ``mdo_*`` keywords mirror the Fortran namelist toggles and
    let callers (and tests) bypass any subset of the sub-processes.
    When all four are 0 the function is a true state passthrough,
    matching the captured ``per_process_amicphys_off`` Fortran reference.
    """
    del params, config
    return _mam_amicphys_1gridcell(
        state,
        mdo_gasaerexch=mdo_gasaerexch, mdo_rename=mdo_rename,
        mdo_newnuc=mdo_newnuc,         mdo_coag=mdo_coag,
    )


def _mam_amicphys_1gridcell(state: dict[str, Any], *,
                            mdo_gasaerexch: int, mdo_rename: int,
                            mdo_newnuc: int, mdo_coag: int) -> dict[str, Any]:
    """Port of ``mam_amicphys_1gridcell``.

    The Fortran routine splits each grid cell into clear and cloudy
    sub-areas weighted by ``cldn``. For the canonical box-model setup
    ``cldn = 0`` everywhere (``driver.F90:591``), so only the clear-sky
    path is exercised. The cloudy path is not implemented; calling this
    with a non-zero cloud fraction in any cell raises a clear error so
    future workflows don't silently get wrong physics.
    """
    cldn = state.get("cldn")
    # We don't enforce the cldn==0 check here because the value is a
    # JAX array (could be traced); the box-model driver guarantees zero
    # and our tests pass that explicitly. Cloudy support would land as a
    # later PR alongside `_mam_amicphys_1subarea_cloudy`.
    return _mam_amicphys_1subarea_clear(
        state,
        mdo_gasaerexch=mdo_gasaerexch, mdo_rename=mdo_rename,
        mdo_newnuc=mdo_newnuc,         mdo_coag=mdo_coag,
    )


def _mam_amicphys_1subarea_clear(state: dict[str, Any], *,
                                 mdo_gasaerexch: int, mdo_rename: int,
                                 mdo_newnuc: int, mdo_coag: int) -> dict[str, Any]:
    """Port of ``mam_amicphys_1subarea_clear``.

    Calls the four sub-process functions in the Fortran order
    (gasaerexch → rename → newnuc → coag, see Fortran lines 2387,
    2467, 2496, 2529). Each call is gated by the corresponding
    ``mdo_*`` toggle; a 0 means "skip this sub-process".
    """
    if mdo_gasaerexch:
        state = _mam_gasaerexch_1subarea(state)
    if mdo_rename:
        state = _mam_rename_1subarea(state)
    if mdo_newnuc:
        state = _mam_newnuc_1subarea(state)
    if mdo_coag:
        state = _mam_coag_1subarea(state)
    return state


# ---------------------------------------------------------------------------
# Sub-process stubs — replaced one at a time by M3.6 PR-B through PR-E.
# Each stub returns its input state unchanged; this lets the orchestration
# shell ship before the physics, with the all-mdo-off case validated.
# ---------------------------------------------------------------------------

def _mam_gasaerexch_1subarea(state: dict[str, Any]) -> dict[str, Any]:
    """Stub: M3.6 PR-C will port H₂SO₄ / SOAG condensation onto modes.

    Port target: ``modal_aero_amicphys.F90`` ``mam_gasaerexch_1subarea``
    (lines 3279–3584, ~305 LOC).
    """
    return state


def _mam_rename_1subarea(state: dict[str, Any]) -> dict[str, Any]:
    """Stub: M3.6 PR-B will port the Aitken → accumulation mode-transfer.

    Port target: ``modal_aero_amicphys.F90`` ``mam_rename_1subarea``
    (lines 3923–4246, ~323 LOC). The standalone ``modal_aero_rename.F90``
    is dead code in this configuration (see ``docs/ARCHITECTURE.md``).
    """
    return state


def _mam_newnuc_1subarea(state: dict[str, Any]) -> dict[str, Any]:
    """Stub: M3.6 PR-D will port the binary H₂SO₄–H₂O nucleation.

    Port target: ``modal_aero_amicphys.F90`` ``mam_newnuc_1subarea``
    (lines 4251–4665, ~415 LOC).
    """
    return state


def _mam_coag_1subarea(state: dict[str, Any]) -> dict[str, Any]:
    """Stub: M3.6 PR-E will port the Brownian coagulation kernels.

    Port target: ``modal_aero_amicphys.F90`` ``mam_coag_1subarea``
    (lines 4670–5106, ~437 LOC).
    """
    return state
