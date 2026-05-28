"""Operator-splitting time-loop driver — Milestone 4 PR-A scaffolding.

Composes the per-step microphysics sequence from
``mam4-original-src-code/test_drivers/driver.F90``'s ``main_time_loop``
(lines 1080–1367):

    1. ``calcsize``      — size redistribution + apply tendency to ``q``
    2. ``wateruptake``   — wet diameters, aerosol water, wet density
    3. *gas-chem*        — currently embedded inside ``gasaerexch``'s
                            H₂SO₄ analytical solver as ``pxt = 1e-16
                            mol/mol/s``; see "Gas-chem placement" below.
    4. *cloud-chem*      — no-op for the box-model fixture (``cldn=0`` →
                            Fortran's ``if (cld > 1e-6)`` gate at
                            ``driver.F90:1263`` never fires).
    5. ``amicphys``      — gasaerexch → rename → newnuc → coag, all the
                            way through the vmr→mmr writeback (implicit
                            via ``_repack_amicphys_view_to_state``).

**Phase A only.** Plain Python ``for`` loop. ``jax.lax.scan`` is M6.

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

``cambox_config.box.in`` sets ``mdo_cloudchem=0``, and the box-model
fixture has ``cldn=0``. Even at ``mdo_cloudchem=1``, Fortran's
``if (mdo_cloudchem > 0 .and. maxval(cld_ncol) > 1.0e-6)`` gate at
``driver.F90:1263`` skips cloud-chem because ``cld=0``. So we don't
need cloud-chem in the box-model trajectory test. Stubbed as a no-op
to keep the operator-splitting sequence structurally faithful.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .processes.amicphys import amicphys
from .processes.calcsize import calcsize
from .processes.wateruptake import wateruptake


def cloud_chem_simple_sub(state: dict[str, Any]) -> dict[str, Any]:
    """No-op for the box-model fixture (``cldn=0``).

    Mirrors ``driver.F90:1263``'s gate: cloud chem only fires when
    ``mdo_cloudchem>0 AND maxval(cldn) > 1e-6``. The box-model namelist
    sets ``mdo_cloudchem=0`` and the initial state has ``cldn=0``, so
    this is structurally dead — leave as a stub so the operator-splitting
    sequence reads correctly. Implement when a future fixture demands it.
    """
    return state


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
    state = cloud_chem_simple_sub(state)   # currently a no-op
    state = amicphys(state)                # all four mdo_* default to 1
    return state


#: Trajectory keys captured per step by :func:`run_timesteps`. Scalars
#: like ``deltat`` and met fields are echoed back unchanged each step
#: and stay out of the trajectory dict.
_TRAJ_KEYS = ("q", "qqcw", "dgncur_a", "dgncur_awet",
              "qaerwat", "wetdens")


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
    be pytree-stable, so this function pre-populates those keys with
    zero placeholders (same shape/dtype as their final outputs) before
    entering scan. The first iteration overwrites them with the real
    values; the placeholders are never observed downstream because the
    scan output trajectory only captures :data:`_TRAJ_KEYS`.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    # Pre-augment the initial state with placeholder keys that calcsize
    # would add on its first call. Shapes mirror dgncur_a / v2ncur_a
    # (per-mode), and dtype mirrors dgncur_a's float64.
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
