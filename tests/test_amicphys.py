"""Validate the M3.6 PR-A amicphys orchestration shell.

Reference: ``tests/reference/per_process_amicphys_off/amicphys_{before,after}.npz``
captured by ``scripts/capture_reference.py --mode instrumented-amicphys-off``
(namelist with ``mdo_gasaerexch=mdo_rename=mdo_newnuc=mdo_coag=0``).
Under those toggles the Fortran ``modal_aero_amicphys_intr`` is a true
state passthrough — every captured array is bit-exact identical between
the before and after snapshots.

PR-A's JAX shell is correspondingly trivial: the orchestration calls
four sub-process stubs that each return state unchanged. PR-B through
PR-E will replace one stub at a time, and that's when the
``per_process/`` (full-bundle) reference becomes the validation target.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401  - enables jax_enable_x64 by default; JAX_ENABLE_X64=0 to opt out
from mam4_jax.processes.amicphys import amicphys

REF_DIR    = Path(__file__).resolve().parent / "reference" / "per_process_amicphys_off"
BEFORE_NPZ = REF_DIR / "amicphys_before.npz"
AFTER_NPZ  = REF_DIR / "amicphys_after.npz"

# Box-model constants. Same convention as test_wateruptake / test_calcsize.
T_BOX_MODEL    = 273.0
PMID_BOX_MODEL = 1.0e5
CLDN_BOX_MODEL = 0.0
DELTAT_60      = 30.0   # 1800 s / 60 snapshots
# Vertical / RH from driver.F90:577-579 + RH_CLEA namelist.
ZMID_BOX_MODEL = 3.0e3
PBLH_BOX_MODEL = 1.1e3
RH_BOX_MODEL   = 0.9


@pytest.fixture(scope="module")
def captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    before = {k: np.asarray(v) for k, v in np.load(BEFORE_NPZ).items()}
    after  = {k: np.asarray(v) for k, v in np.load(AFTER_NPZ).items()}
    return before, after


def _build_state(before: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    nstep, ncol, pver, _ = before["q"].shape
    return {
        "q":           jnp.asarray(before["q"]),
        "qqcw":        jnp.asarray(before["qqcw"]),
        "dgncur_a":    jnp.asarray(before["dgncur_a"]),
        "dgncur_awet": jnp.asarray(before["dgncur_awet"]),
        "qaerwat":     jnp.asarray(before["qaerwat"]),
        "wetdens":     jnp.asarray(before["wetdens"]),
        "t":           jnp.asarray(np.full((nstep, ncol, pver), T_BOX_MODEL)),
        "pmid":        jnp.asarray(np.full((nstep, ncol, pver), PMID_BOX_MODEL)),
        "cldn":        jnp.asarray(np.full((nstep, ncol, pver), CLDN_BOX_MODEL)),
        "zmid":        jnp.asarray(np.full((nstep, ncol, pver), ZMID_BOX_MODEL)),
        "pblh":        jnp.asarray(np.full((nstep, ncol, pver), PBLH_BOX_MODEL)),
        "relhum":      jnp.asarray(np.full((nstep, ncol, pver), RH_BOX_MODEL)),
        "deltat":      jnp.asarray(DELTAT_60),
    }


def test_amicphys_all_off_is_passthrough(captured) -> None:
    """With all four mdo_* set to 0, amicphys must return its input
    state unchanged for every key the Fortran captures."""
    before, after = captured
    state = _build_state(before)
    new_state = amicphys(state,
                         mdo_gasaerexch=0, mdo_rename=0,
                         mdo_newnuc=0, mdo_coag=0)
    for key in ("q", "qqcw", "dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_array_equal(
            np.asarray(new_state[key]), after[key],
            err_msg=f"all-mdo-off amicphys disturbed {key!r}",
        )


RENAME_ONLY_REF_DIR = (
    Path(__file__).resolve().parent / "reference" / "per_process_rename_only"
)


@pytest.fixture(scope="module")
def rename_only_captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Single-toggle Fortran capture: `mdo_rename=1, others=0`.

    With gasaerexch off, `qaer_delsub_grow4rnam` is identically zero at
    the rename call site. In the canonical box-model fixture, the
    Aitken-mode `dgn_t_old` stays well below `dp_belowcut` because
    nothing else is growing the mode — so rename's optaa=40 guard at
    Fortran line 4141 trips and rename is a no-op. The JAX orchestration
    with `mdo_rename=1, others=0` must reproduce this passthrough.
    """
    before = {k: np.asarray(v)
              for k, v in np.load(RENAME_ONLY_REF_DIR / "amicphys_before.npz").items()}
    after  = {k: np.asarray(v)
              for k, v in np.load(RENAME_ONLY_REF_DIR / "amicphys_after.npz").items()}
    return before, after


def test_orchestration_rename_only_matches_fortran(rename_only_captured) -> None:
    """JAX `amicphys(state, mdo_rename=1, others=0)` reproduces the
    single-toggle Fortran capture's `amicphys_after` at machine epsilon.

    Validates the state-dict ↔ amicphys-local-view round-trip
    (unpack → rename → repack) plus the mmr↔vmr conversion (M3.6 PR-C).
    """
    before, after = rename_only_captured
    state = _build_state(before)
    new_state = amicphys(state,
                         mdo_gasaerexch=0, mdo_rename=1,
                         mdo_newnuc=0, mdo_coag=0)
    for key in ("q", "qqcw", "dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), after[key],
            rtol=1e-12, atol=1e-30,
            err_msg=f"rename-only orchestration diverged on {key!r}",
        )


GASAEREXCH_REF_DIR = (
    Path(__file__).resolve().parent / "reference" / "per_process_gasaerexch"
)


@pytest.fixture(scope="module")
def gasaerexch_captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Single-toggle Fortran capture: `mdo_gasaerexch=1, others=0`.

    Includes the full gasaerexch path (both `mam_soaexch_1subarea` for
    SOA exchange AND the H₂SO₄ analytical solver). Only the separate
    `mam_pcarbon_aging_1subarea` sub-process is skipped via overlay
    (out of M3.6 scope). Replaces the PR-D `per_process_gasaerexch_only`
    fixture (which additionally skipped soaexch).

    Use `amicphys_after_writeback.npz` for `q`/`qqcw` (the existing
    `amicphys_after` dump captures `q` before the driver's vmr→mmr
    writeback — see PR-D PROGRESS entry for context).
    """
    before = {k: np.asarray(v)
              for k, v in np.load(GASAEREXCH_REF_DIR / "amicphys_before.npz").items()}
    aw     = {k: np.asarray(v) for k, v in np.load(
                GASAEREXCH_REF_DIR / "amicphys_after_writeback.npz").items()}
    return before, aw


GASAEREXCH_AND_NEWNUC_REF_DIR = (
    Path(__file__).resolve().parent / "reference"
    / "per_process_gasaerexch_and_newnuc"
)


@pytest.fixture(scope="module")
def gasaerexch_and_newnuc_captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Single-toggle Fortran capture: `mdo_gasaerexch=1, mdo_newnuc=1,
    others=0` with `skip_pcarbon_aging.patch`. Newnuc needs gasaerexch's
    `qgas_avg` to fire — that's why both must be on. SOA exchange runs
    too (inside the unmodified gasaerexch call).
    """
    before = {k: np.asarray(v) for k, v in np.load(
        GASAEREXCH_AND_NEWNUC_REF_DIR / "amicphys_before.npz").items()}
    aw     = {k: np.asarray(v) for k, v in np.load(
        GASAEREXCH_AND_NEWNUC_REF_DIR / "amicphys_after_writeback.npz").items()}
    return before, aw


def test_orchestration_gasaerexch_and_newnuc_matches_fortran(
    gasaerexch_and_newnuc_captured,
) -> None:
    """JAX `amicphys(state, mdo_gasaerexch=1, mdo_newnuc=1, others=0)`
    reproduces the Fortran fixture at the ADR-015 diffrax-branch bar
    (M3.6 PR-F3 wiring + M7 PR-D1/D2 solver-port adjustments).

    PR-D1's test_sweep.py rewrite missed this test; M6 PR-J3 closes
    the gap. Empirical max rel-err on `q` is ~1e-2 (driven by the
    diffrax soaexch structural offset propagating through newnuc);
    abs diff stays under 1e-12 for near-zero tracers, so atol=1e-12
    avoids rtol blow-up at 1e-25-magnitude values.
    """
    before, aw = gasaerexch_and_newnuc_captured
    state = _build_state(before)
    new_state = amicphys(state,
                         mdo_gasaerexch=1, mdo_rename=0,
                         mdo_newnuc=1, mdo_coag=0)
    for key in ("q", "qqcw"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), aw[key],
            rtol=1e-2, atol=1e-12,
            err_msg=f"gasaerexch+newnuc orchestration diverged on {key!r}",
        )
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), aw[key],
            rtol=5e-3, atol=1e-15,
            err_msg=f"gasaerexch+newnuc orchestration drifted on {key!r}",
        )


COAG_REF_DIR = (
    Path(__file__).resolve().parent / "reference" / "per_process_coag"
)


@pytest.fixture(scope="module")
def coag_captured() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Single-toggle Fortran capture: `mdo_coag=1, others=0` with
    `skip_pcarbon_aging.patch`.

    Coag operates on the current state's `dgncur_a`, `dgncur_awet`,
    `wetdens` (set by calcsize + wateruptake upstream of amicphys); it
    does not need gasaerexch outputs, unlike newnuc.
    """
    before = {k: np.asarray(v) for k, v in np.load(
        COAG_REF_DIR / "amicphys_before.npz").items()}
    aw     = {k: np.asarray(v) for k, v in np.load(
        COAG_REF_DIR / "amicphys_after_writeback.npz").items()}
    return before, aw


def test_orchestration_coag_only_matches_fortran(coag_captured) -> None:
    """JAX `amicphys(state, mdo_coag=1, others=0)` reproduces the
    Fortran coag-only fixture at 1e-6 on the aerosol-tracer slots of
    `q` and `qqcw` (M3.6 PR-G3 wiring test). **Closes M3.6.**

    Gas-tracer slots (``LMAP_GAS``) are excluded from the comparison.
    Coag does not touch gases, but ``driver.F90:1249`` applies a
    ``vmr += 1e-16*dt`` gas-chem stub to H₂SO₄ *outside* amicphys, and
    that increment is captured in the Fortran writeback dump. The
    matching ``gasaerexch`` test absorbs this in the H₂SO₄ analytical
    solver (``_mam_gasaerexch_1subarea`` line ~594); coag-only has no
    such mechanism because gasaerexch is off, so the gas-chem-added
    H₂SO₄ shows up only in Fortran. The cleanest fix is to limit the
    coag test to coag's actual validation surface — the aerosol-tracer
    slots.
    """
    import mam4_jax.data as _data  # noqa
    gas_slots = set(int(i) for i in _data.LMAP_GAS)

    before, aw = coag_captured
    state = _build_state(before)
    new_state = amicphys(state,
                         mdo_gasaerexch=0, mdo_rename=0,
                         mdo_newnuc=0, mdo_coag=1)

    pcnst = before["q"].shape[-1]
    aerosol_slots = [i for i in range(pcnst) if i not in gas_slots]

    for key in ("q", "qqcw"):
        np.testing.assert_allclose(
            np.asarray(new_state[key])[..., aerosol_slots],
            aw[key][..., aerosol_slots],
            rtol=1e-6, atol=1e-20,
            err_msg=f"coag-only orchestration diverged on {key!r} "
                    "(aerosol slots)",
        )
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), aw[key],
            rtol=1e-3, atol=1e-15,
            err_msg=f"coag-only orchestration drifted on {key!r}",
        )


def test_orchestration_gasaerexch_matches_fortran(gasaerexch_captured) -> None:
    """JAX `amicphys(state, mdo_gasaerexch=1, others=0)` reproduces the
    Fortran gasaerexch+soaexch fixture (no skip patches besides pcarbon
    aging) at ADR-015's diffrax-branch bar.

    Historical M3.6 PR-E target was `rtol=1e-6` on `q`/`qqcw`, but
    M7 PR-D1's diffrax soaexch port produces O(dt²) per-step drift
    versus Fortran's semi-implicit (empirical worst rel-err on `q` is
    ~4e-3). PR-D1's test_sweep.py rewrite missed this test; M6 PR-J3
    closes the gap. atol=1e-12 keeps rtol from blowing up on
    1e-25-magnitude near-zero tracers (observed abs diff floor ~1.5e-13).
    """
    before, aw = gasaerexch_captured
    state = _build_state(before)
    new_state = amicphys(state,
                         mdo_gasaerexch=1, mdo_rename=0,
                         mdo_newnuc=0, mdo_coag=0)

    for key in ("q", "qqcw"):
        np.testing.assert_allclose(
            np.asarray(new_state[key]), aw[key],
            rtol=1e-2, atol=1e-12,
            err_msg=f"gasaerexch-only orchestration diverged on {key!r}",
        )
    for key in ("dgncur_a", "dgncur_awet", "qaerwat", "wetdens"):
        # Fortran's `update_aerosol_props` (called inside the cond
        # sub-stepping when `do_cond_wateruptake=.true.`) re-runs
        # wateruptake after every gasaerexch substep, so dgn_awet /
        # qaerwat / wetdens diverge slightly from the inputs. JAX
        # doesn't implement that re-uptake yet (Phase A only ports the
        # main sub-process), so we just confirm these fields are still
        # in the right ballpark — they are not part of gasaerexch's
        # validation surface for PR-D.
        np.testing.assert_allclose(
            np.asarray(new_state[key]), aw[key],
            rtol=1e-3, atol=1e-15,
            err_msg=f"gasaerexch-only orchestration drifted on {key!r}",
        )


def test_amicphys_returns_all_state_keys(captured) -> None:
    """The returned state preserves keys that weren't part of the
    aerosol-state inputs (meteorology, deltat, etc.)."""
    before, _ = captured
    state = _build_state(before)
    new_state = amicphys(state)
    for key in ("t", "pmid", "cldn", "deltat"):
        assert key in new_state, f"amicphys dropped state key {key!r}"


def test_condensation_backend_default_is_diffrax() -> None:
    """The substep backend is strictly opt-in: nothing changes unless a
    host calls ``configure_condensation``."""
    from mam4_jax.processes import amicphys as _amic
    assert _amic._COND["backend"] == "diffrax"


def test_configure_gas_netprod_default_and_override() -> None:
    """``configure_gas_netprod`` sets the other-process gas production rates.

    The default preserves the driver.F90:1248 stub (1e-16 mol/mol/s H2SO4,
    0 SOA); a host that prognoses its own sulfur/SOA chemistry — or runs a
    sulfur-free case — can zero either rate. ``None`` leaves a rate unchanged.
    Restores the process-global afterwards so it doesn't leak into other tests.
    """
    from mam4_jax.processes import amicphys as _amic
    saved = dict(_amic._GAS_NETPROD)
    try:
        assert _amic._GAS_NETPROD == {"h2so4": 1.0e-16, "soa": 0.0}
        _amic.configure_gas_netprod(h2so4=0.0)
        assert _amic._GAS_NETPROD["h2so4"] == 0.0
        assert _amic._GAS_NETPROD["soa"] == 0.0        # unchanged (None)
        _amic.configure_gas_netprod(soa=3.0e-17)
        assert _amic._GAS_NETPROD["h2so4"] == 0.0      # unchanged (None)
        assert _amic._GAS_NETPROD["soa"] == 3.0e-17
    finally:
        _amic.configure_gas_netprod(**saved)
    assert _amic._GAS_NETPROD == saved


def test_condensation_substep_matches_fortran(gasaerexch_captured) -> None:
    """The operator-split ``substep`` backend reproduces the Fortran
    gasaerexch+soaexch fixture at the SAME bar as the diffrax backend.

    The substep path replaces the adaptive Kvaerno5 SOA solve with an
    N-substep frozen-``g_star`` integrator and the H2SO4 solve with its
    exact closed form. It must be at least as faithful to Fortran as the
    diffrax path it replaces — Fortran itself is operator-split, so the
    substep scheme is structurally closer to it. We assert the existing
    diffrax-branch tolerance (``rtol=1e-2`` on ``q``/``qqcw``) holds.

    Restores the process-global backend afterwards so the opt-in default
    doesn't leak into other tests sharing this process.
    """
    from mam4_jax.processes import amicphys as _amic
    before, aw = gasaerexch_captured
    state = _build_state(before)
    saved = dict(_amic._COND)
    try:
        _amic.configure_condensation(backend="substep", n_substeps=4)
        new_state = amicphys(state,
                             mdo_gasaerexch=1, mdo_rename=0,
                             mdo_newnuc=0, mdo_coag=0)
        for key in ("q", "qqcw"):
            arr = np.asarray(new_state[key])
            assert np.all(np.isfinite(arr)), f"substep produced non-finite {key!r}"
            np.testing.assert_allclose(
                arr, aw[key], rtol=1e-2, atol=1e-12,
                err_msg=f"substep gasaerexch diverged from Fortran on {key!r}",
            )
    finally:
        _amic.configure_condensation(**saved)


def test_condensation_astem_matches_fortran(gasaerexch_captured) -> None:
    """The Fortran-faithful ``astem`` backend reproduces the gasaerexch
    fixture.

    ``astem`` IS the upstream's own adaptive semi-implicit step1/step2
    SOA scheme (plus the exact analytic H2SO4), so it should be at least
    as faithful as the diffrax path. We assert the diffrax-branch bar
    (``rtol=1e-2`` on ``q``/``qqcw``) holds and the result is finite,
    then restore the process-global default.
    """
    from mam4_jax.processes import amicphys as _amic
    before, aw = gasaerexch_captured
    state = _build_state(before)
    saved = dict(_amic._COND)
    try:
        _amic.configure_condensation(backend="astem")
        new_state = amicphys(state,
                             mdo_gasaerexch=1, mdo_rename=0,
                             mdo_newnuc=0, mdo_coag=0)
        for key in ("q", "qqcw"):
            arr = np.asarray(new_state[key])
            assert np.all(np.isfinite(arr)), f"astem produced non-finite {key!r}"
            np.testing.assert_allclose(
                arr, aw[key], rtol=1e-2, atol=1e-12,
                err_msg=f"astem gasaerexch diverged from Fortran on {key!r}",
            )
    finally:
        _amic.configure_condensation(**saved)


def test_substep_and_astem_agree_per_call(gasaerexch_captured) -> None:
    """Cross-validate substep vs astem at per-call level.

    Both opt-in backends are validated independently against the
    Fortran reference (the two tests above), but a regression in just
    one of them might still match Fortran if both drift together.
    This test asserts the two opt-in backends agree with each other at
    ``rtol=1e-2`` — the PR-59 measurement showed ~0.18 % agreement on
    the 3-day global aerosol burden (T21), so per-call agreement
    should be at least as tight.

    Catches a future regression in either backend that wouldn't surface
    via the Fortran-match tests alone.
    """
    from mam4_jax.processes import amicphys as _amic
    before, _aw = gasaerexch_captured
    state = _build_state(before)
    saved = dict(_amic._COND)
    try:
        _amic.configure_condensation(backend="substep", n_substeps=4)
        substep_out = amicphys(state,
                               mdo_gasaerexch=1, mdo_rename=0,
                               mdo_newnuc=0, mdo_coag=0)
        _amic.configure_condensation(backend="astem")
        astem_out = amicphys(state,
                             mdo_gasaerexch=1, mdo_rename=0,
                             mdo_newnuc=0, mdo_coag=0)
        for key in ("q", "qqcw"):
            np.testing.assert_allclose(
                np.asarray(substep_out[key]), np.asarray(astem_out[key]),
                rtol=1e-2, atol=1e-12,
                err_msg=f"substep and astem disagree on {key!r}",
            )
    finally:
        _amic.configure_condensation(**saved)


def test_astem_backend_not_grad_compatible(gasaerexch_captured) -> None:
    """``astem`` uses ``jax.lax.while_loop`` for its adaptive substep
    iteration, which is NOT reverse-mode differentiable.

    Locks in the documented contract: hosts using ``jax.grad`` (e.g.,
    M9 calibration workflows) must select ``"diffrax"`` or ``"substep"``
    — both grad-clean per PR-J5's audit. If a future "fix" silently
    swaps ``lax.while_loop`` for ``lax.fori_loop`` with a static cap
    (or some other grad-compatible construct), this test fails and
    the docstring contract should be reviewed.
    """
    import jax
    from mam4_jax.processes import amicphys as _amic
    before, _ = gasaerexch_captured
    state = _build_state(before)
    saved = dict(_amic._COND)
    try:
        _amic.configure_condensation(backend="astem")

        def loss(q):
            s = {**state, "q": q}
            new_state = amicphys(s,
                                 mdo_gasaerexch=1, mdo_rename=0,
                                 mdo_newnuc=0, mdo_coag=0)
            return jnp.sum(new_state["q"])

        # `jax.grad` through `lax.while_loop` raises at trace time. We
        # don't assert the exact exception type — diffrax/JAX may
        # change it — only that an exception is raised. The message
        # typically contains "while_loop" or "Reverse-mode".
        with pytest.raises(Exception):
            jax.grad(loss)(jnp.asarray(state["q"]))
    finally:
        _amic.configure_condensation(**saved)
