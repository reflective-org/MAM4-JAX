"""Operator-splitting time-loop driver — Milestone 4 PR-A scaffolding.

Composes the per-step microphysics sequence from
``mam4-original-src-code/test_drivers/driver.F90``'s ``main_time_loop``
(lines 1080–1367):

    1. ``calcsize``      — size redistribution + apply tendency to ``q``
    2. ``wateruptake``   — wet diameters, aerosol water, wet density
    3. *gas-chem*        — currently embedded inside ``gasaerexch``'s
                            H₂SO₄ analytical solver as ``pxt = 1e-16
                            mol/mol/s``; see "Gas-chem placement" below.
    4. ``cloudchem_simple_sub`` — parameterized aqueous SO₂→SO₄ via the
                            JAX port in ``mam4_jax.processes.cloudchem``
                            (M8 PR-K2). Conditional on ``state["cldn"]``:
                            cycles when ``cldn ≤ 0.009`` (bit-exact
                            identity), fires otherwise. Driver wrapper
                            handles the mmr↔vmr conversion.
    5. ``amicphys``      — gasaerexch → rename → newnuc → coag, all the
                            way through the vmr→mmr writeback (implicit
                            via ``_repack_amicphys_view_to_state``).

**Optimisation status.** ``run_step`` is ``@jax.jit``-compiled
(M6 PR-J1). ``run_timesteps`` uses ``jax.lax.scan`` (M6 PR-J2)
to amortise the body trace across long trajectories.

Gas-chem placement
------------------

Fortran's ``driver.F90:1249`` applies ``vmr[H₂SO₄] += 1e-16·deltat``
*outside* the amicphys call. Amicphys then *rolls back* to the
pre-gas-chem state when ``do_cond`` is on (``modal_aero_amicphys.F90:
2270-2289``), feeding the production rate as a continuous source term
to gasaerexch's analytical solver — which integrates production and
loss together over ``deltat`` (the operator-splitting-free path).

The JAX port collapses this into the gasaerexch solver directly (see
``processes/amicphys.py:594`` — ``qgas_netprod_h2so4 = 1e-16`` hard-
coded). The end-state matches Fortran at machine ε when gasaerexch is
on. When gasaerexch is *off* (e.g., the coag-only single-toggle
fixture), the gas-chem term is missing on the JAX side — which is why
``test_orchestration_coag_only_matches_fortran`` excludes gas-tracer
slots from its comparison.

This driver inherits the same convention: the full-physics path (all
``mdo_*=1``) has gas-chem absorbed inside gasaerexch, so no separate
gas-chem step is needed here. A follow-up PR (likely M5 work, when
running per-namelist sweeps) can lift the gas-chem stub to driver
layer if any future fixture toggles gasaerexch off.

Cloud-chem placement
--------------------

M8 PR-K3 wires ``cloudchem_simple_sub`` from
``mam4_jax.processes.cloudchem``. The driver wrapper below converts
``q``/``qqcw`` (mmr, pcnst=35) to ``vmr``/``vmrcw`` (gas_pcnst=30),
calls the ported physics, and applies the delta back to ``q``/``qqcw``
in mmr-space.

**Conditional firing.** No Python-static ``mdo_cloudchem`` flag —
the cloudchem function unconditionally executes but cycles internally
when ``state["cldn"] <= 0.009`` (matching Fortran's per-gridcell
``if (cldn <= 0.009) cycle``). The driver wrapper uses a
*delta-based* update (``q += vmr_to_mmr_factor * (vmr_out − vmr_in)``),
so cldn-zero state-dicts pass through with bit-exact identity (the
delta is exactly zero; no round-trip ULP drift). All pre-M8 tests
with cldn=0 stay byte-identical.

**Gas-chem ordering note.** Fortran applies the gas-chem source
(``vmr[H₂SO₄] += 1e-16·dt``, ``driver.F90:1249``) *before* cloudchem.
The current JAX path absorbs gas-chem into ``amicphys``'s H₂SO₄ ODE
*after* cloudchem. For ``cldn=0`` this is mathematically equivalent
(cloudchem is a no-op, so order doesn't matter). For ``cldn>0`` it
introduces a per-step ordering bias of ~``0.5 · 1e-16 · dt`` on
H₂SO₄ — small at low ``dt``, accumulates over a trajectory. Quantified
by the new ``test_run_step_with_cloudchem_matches_fortran`` test;
revisited in PR-K3b if the measured bar misses ADR-015's 3 %.
"""
from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp

from . import data
from .processes.amicphys import amicphys
from .processes.calcsize import calcsize
from .processes.cloudchem import (
    cloudchem_simple_sub as _cloudchem_simple_sub_vmr,
)
from .processes.wateruptake import wateruptake


#: Constant H₂SO₄ gas-chem source rate (mol/mol/s) from
#: ``driver.F90:1249``. M8 PR-K3: extracted to the driver level so it
#: runs *before* cloudchem (matching Fortran's per-step ordering), with
#: ``amicphys`` called with ``qgas_netprod_h2so4=0`` to avoid double-
#: counting the same source inside the H₂SO₄ ODE.
GAS_CHEM_H2SO4_RATE_VMR: float = 1.0e-16


def gas_chem_simple_step(state: dict[str, Any]) -> dict[str, Any]:
    """Driver-level gas-chem source — ``q[H₂SO₄] += rate · dt`` in mmr-space.

    Mirrors ``driver.F90:1249`` (``vmr(:,:,lmz_h2so4g) = vmr + 1e-16·dt``).
    Fortran applies this in vmr-space; JAX state-dict carries ``q`` in
    mmr-space, so we convert via ``data.VMR_TO_MMR[PCNST_H2SO4_GAS]``
    (= ``adv_mass[H₂SO₄] / mwdry``).

    Why driver-level (M8 PR-K3): the pre-M8 path absorbed this source
    into ``amicphys``'s H₂SO₄ ODE (``qgas_netprod_h2so4 = 1e-16``).
    That works at ``cldn = 0`` (cloudchem is a no-op) but mis-orders
    with cloudchem when ``cldn > 0`` — Fortran reduces the gas-chem-
    applied H₂SO₄ by ``tmpf`` before amicphys, whereas the in-ODE
    source would add ``1e-16·dt`` *after* cloudchem's halving. The
    structural fix is to apply gas-chem here (before cloudchem) and
    pass ``qgas_netprod_h2so4=0`` to amicphys.
    """
    q = state["q"]
    deltat = state["deltat"]
    factor_vmr_to_mmr = data.VMR_TO_MMR[data.PCNST_H2SO4_GAS]
    add_mmr = GAS_CHEM_H2SO4_RATE_VMR * deltat * factor_vmr_to_mmr
    new_q = q.at[..., data.PCNST_H2SO4_GAS].add(add_mmr)
    return {**state, "q": new_q}


def cloudchem_simple_sub(state: dict[str, Any]) -> dict[str, Any]:
    """Driver-level wrapper around ``processes.cloudchem.cloudchem_simple_sub``.

    Mirrors ``driver.F90:1263-1270`` (the call site bracketed by the
    ``mdo_cloudchem`` + ``maxval(cldn) > 1e-6`` gate). Converts the
    state-dict's ``q``/``qqcw`` (mmr, pcnst=35) to ``vmr``/``vmrcw``
    (gas_pcnst=30), invokes the physics, and projects the delta back
    into mmr-space — so:

    - When ``cldn <= 0.009`` everywhere, the cloudchem cycle mask
      returns ``vmr_out == vmr`` exactly; the projected delta is zero;
      ``q`` and ``qqcw`` are unchanged bit-for-bit. Pre-M8 tests pass
      through unaffected.
    - When ``cldn > 0.009``, the physics fires; only the slots cloudchem
      touches (H₂SO₄, SO₂ in ``q``; SO4_cw accum and aitken in
      ``qqcw``) move.

    The ``state["cldn"]`` field is required (already set by all
    fixtures and the run-state builders since M4 PR-A).
    """
    q       = state["q"]
    qqcw    = state["qqcw"]
    cldn    = state["cldn"]
    deltat  = state["deltat"]

    # Slice the chem-tracer portion [LOFFSET:] of q (pcnst=35 → gas_pcnst=30).
    factor_to_vmr = data.MMR_TO_VMR[data.AMICPHYS_LOFFSET:]   # (30,)
    factor_to_mmr = data.VMR_TO_MMR[data.AMICPHYS_LOFFSET:]   # (30,)

    q_chem    = q[...,    data.AMICPHYS_LOFFSET:]              # (..., 30)
    qqcw_chem = qqcw[..., data.AMICPHYS_LOFFSET:]

    vmr   = q_chem    * factor_to_vmr
    vmrcw = qqcw_chem * factor_to_vmr

    vmr_out, vmrcw_out = _cloudchem_simple_sub_vmr(
        vmr, vmrcw, cldn, deltat,
    )

    # Delta-based update so the cldn=0 path is bit-exact identity.
    # (vmr_out − vmr) is exactly zero when cloudchem cycles; nonzero
    # only on the H₂SO₄, SO₂, SO4_cw_accum, SO4_cw_aitken slots when
    # the body fires.
    new_q    = q.at[...,    data.AMICPHYS_LOFFSET:].add(
        (vmr_out   - vmr)   * factor_to_mmr,
    )
    new_qqcw = qqcw.at[..., data.AMICPHYS_LOFFSET:].add(
        (vmrcw_out - vmrcw) * factor_to_mmr,
    )

    return {**state, "q": new_q, "qqcw": new_qqcw}


@jax.jit
def run_step(state: dict[str, Any]) -> dict[str, Any]:
    """One operator-splitting timestep.

    Sequence mirrors ``driver.F90:1080-1367`` (``main_time_loop``):

    1. ``calcsize`` — size redistribution + apply tendency to ``q``.
    2. ``wateruptake`` — wet diameters, aerosol water, wet density.
    3. *gas-chem* — currently absorbed inside ``gasaerexch``'s analytical
       solver (see module docstring); no separate step here.
    4. ``cloud_chem_simple_sub`` — no-op on the box-model fixture.
    5. ``amicphys`` (``mdo_gasaerexch=mdo_rename=mdo_newnuc=mdo_coag=1``)
       — the full microphysics including the vmr↔mmr writeback.

    Returns the updated state dict (same keys as the input, with
    ``q``/``qqcw``/``dgncur_a``/``dgncur_awet``/``qaerwat``/``wetdens``
    advanced one timestep).
    """
    state = calcsize(state)
    state = wateruptake(state)
    state = gas_chem_simple_step(state)    # H2SO4 source: q += 1e-16·dt (vmr-space rate)
    state = cloudchem_simple_sub(state)    # bit-exact no-op when cldn ≤ 0.009
    state = amicphys(state, qgas_netprod_h2so4=0.0)  # source already applied above
    return state


#: Trajectory keys captured per step by :func:`run_timesteps`. Scalars
#: like ``deltat`` and met fields are echoed back unchanged each step
#: and stay out of the trajectory dict.
_TRAJ_KEYS = ("q", "qqcw", "dgncur_a", "dgncur_awet",
              "qaerwat", "wetdens")


@functools.partial(jax.jit, static_argnums=(1,))
def run_timesteps(state: dict[str, Any], n_steps: int) -> dict[str, Any]:
    """Run ``n_steps`` operator-splitting timesteps and return a
    stacked trajectory.

    For each step the returned dict's leading axis grows by one; the
    final entry along that axis is the state at the *end* of step
    ``n_steps`` (i.e. the input state is **not** included as step 0 —
    matches Fortran's ``do nstep = 1, nstop`` convention where the
    NetCDF output's step-1 entry is post-step-1, not the IC).

    Uses ``jax.lax.scan`` (M6 PR-J2): the JIT-compiled ``run_step``
    body is traced once and applied ``n_steps`` times inside the scan,
    with per-step snapshots stacked into the output trajectory. A new
    ``jax.lax.scan`` trace happens once per distinct ``n_steps`` value
    (Python-static length argument), but ``run_step`` itself reuses
    its JIT cache.

    ``calcsize`` adds three derived keys to the state on each call
    (``dgncur_c``, ``v2ncur_a``, ``v2ncur_c``). The scan carry must
    be pytree-stable, so this function pre-populates those keys **if
    missing** with zero placeholders (same shape/dtype as ``dgncur_a``)
    before entering scan; a caller that already supplies the keys
    keeps their values untouched. The first scan iteration overwrites
    the placeholders with the real values; downstream they're invisible
    because the scan output trajectory only captures :data:`_TRAJ_KEYS`.

    **Compile cost.** Scan trades a one-time body-trace cost for an
    asymptotic per-step speedup. Very short trajectories
    (``n_steps`` of a few thousand or less) may run slower than the
    previous Python ``for``-loop baseline — the M6 PR-J2 benchmark
    saw a 1.7× slowdown at ``n_steps = 2880`` (dt=30s 24h), offset by
    >1000× per-step amortisation at ``n_steps = 86400`` (dt=1s 24h).
    Don't read the per-step time as constant across ``n_steps``.

    **JIT cache.** ``run_timesteps`` itself is ``@jax.jit``-compiled
    with ``n_steps`` static (one cache entry per distinct ``n_steps``).
    This amortises the Python-side dispatch around ``jax.lax.scan``
    and the per-call abstractification of the 16-key carry pytree —
    without it, the inner ``scan`` cache hits but each call still
    paid ~1 s of Python overhead, which dominated benchmarks that
    invoke ``run_timesteps`` many times (e.g. 1000-sim wall-time
    studies). Inside the JIT'd ``run_timesteps``, scan calls
    ``run_step`` with the 16-key augmented carry; direct callers of
    ``run_step`` (e.g. ``tests/test_driver.py``) pass a 13-key state
    and get their own ``run_step`` cache entry. Both compiles are
    ~1-2 s each on this hardware.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    # Pre-augment the initial state with placeholder keys that calcsize
    # would add on its first call. The placeholder values are never
    # observed because: precondition — calcsize() *writes* but never
    # *reads* these three keys (they're computed from num / drv derived
    # from q / qqcw; see calcsize.py:545-565). If a future calcsize
    # change makes it read its own previous-step v2ncur_a, this becomes
    # unsafe: the first scan iteration would silently use the zero
    # placeholder, corrupting the trajectory's first step.
    #
    # Shape assumption: dgncur_c / v2ncur_a / v2ncur_c all share
    # dgncur_a's (..., NTOT_AMODE) shape today; if calcsize ever evolves
    # a per-moment axis, scan errors loudly at runtime (carry pytree
    # mismatch) — caught, not silently corrupted.
    dgncur_a = state["dgncur_a"]
    placeholder = jnp.zeros_like(dgncur_a)
    augmented = {**state}
    for k in ("dgncur_c", "v2ncur_a", "v2ncur_c"):
        augmented.setdefault(k, placeholder)

    def _scan_body(carry_state: dict[str, Any], _) -> tuple[
        dict[str, Any], dict[str, Any]
    ]:
        new_state = run_step(carry_state)
        output = {k: new_state[k] for k in _TRAJ_KEYS}
        return new_state, output

    _, trajectory = jax.lax.scan(
        _scan_body, augmented, xs=None, length=n_steps,
    )
    return trajectory
