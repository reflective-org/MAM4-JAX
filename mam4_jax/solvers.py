"""ODE solver strategy module — diffrax wrapper.

PR-I1 (M7) skeleton. `solve_ivp` raises NotImplementedError; PR-D1
wires the soaexch port through it, PR-D2 wires H2SO4.

Returns a `SolverResult(ts, ys, stats)` so call sites get a recorded
trajectory and adaptive-controller diagnostics without per-site
instrumentation. See `docs/plans/015-diffrax-infra.md` for the
rationale on the wider-than-narrow signature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import diffrax
import jax.numpy as jnp


@dataclass(frozen=True)
class SolverConfig:
    """Per-call solver configuration consumed by :func:`solve_ivp`.

    Call sites (e.g. ``_mam_soaexch_1subarea``, the H₂SO₄ analytical
    solver in ``_mam_gasaerexch_1subarea``) instantiate this with their
    own tolerances chosen to match the Fortran reference at machine ε.
    Hosts that want to relax those tolerances for speed at the cost of
    accuracy should use :func:`configure` instead — it overlays
    process-global overrides on top of whatever ``SolverConfig`` the
    call site uses, without requiring code changes in every call site.
    Overrides apply to ``rtol`` / ``atol`` / ``max_steps`` only; the
    ``solver`` choice (e.g. ``Kvaerno5``) and ``dt0`` (initial step
    hint) are intentionally not overridable — they're structural
    decisions per call site (different stability regions, different
    memory cost, see plan 021).
    """

    solver: str = "Kvaerno5"
    rtol: float = 1e-9
    atol: float = 1e-12
    max_steps: int = 4096
    dt0: Optional[float] = None


# Process-global overrides applied by `solve_ivp` on top of the per-call
# `SolverConfig`. Lets a host (e.g. jax-gcm) dial the speed/accuracy/robustness
# tradeoff without threading config through every call site:
#   * rtol/atol — looser tolerances => far fewer adaptive steps (the dominant
#     cost; atol=1e-20 default forces float64 + many tiny steps).
#   * throw=False — a cell that hits ``max_steps`` returns its best estimate
#     with a non-success ``result`` code instead of raising, so one pathological
#     cell can't abort (or, with a raised cap, crawl) the whole vmap batch.
# All ``None`` => upstream behaviour is unchanged.
#
# NOT thread-safe — single module-level dict, no lock. The "set once at
# startup" contract (see ``configure`` docstring) makes this a non-issue
# for single-process / process-per-device hosts. A single-process
# multi-threaded host calling ``configure`` from different threads can
# observe non-deterministic results; either coordinate or avoid the
# pattern.
_OVERRIDE: dict = {"rtol": None, "atol": None, "max_steps": None, "throw": None}


def configure(
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_steps: Optional[int] = None,
    throw: Optional[bool] = None,
    *,
    reset: bool = False,
) -> None:
    """Set process-global :func:`solve_ivp` overrides.

    Any argument left as ``None`` leaves the corresponding override
    unchanged. Pass ``reset=True`` to clear all overrides back to the
    ``SolverConfig`` per-call defaults (and optionally re-set specific
    fields in the same call: ``configure(reset=True, rtol=1e-6)``
    clears everything then sets rtol).

    Parameters
    ----------
    rtol, atol
        Adaptive PI-controller tolerances. The defaults baked into
        each call site's :class:`SolverConfig` are tight
        (rtol=1e-9, atol=1e-12 or atol=1e-20 for some sites) so the
        validation residual matches Fortran at machine ε. Hosts that
        run inside a larger model (e.g. a GCM) can loosen these for a
        speed/accuracy trade-off — empirically (jax-gcm A100 profile),
        ``rtol=1e-6 / atol=1e-15`` gives ~2.8× speedup with ~0.13 %
        per-step rel-err vs the tight defaults.
    max_steps
        Per-call cap on adaptive sub-step count. Raise to allow
        pathological cells to converge; combine with ``throw=False``
        to swallow non-convergent cells instead of aborting the batch.
    throw
        ``True`` (the upstream diffrax default) raises on
        ``max_steps`` exhaustion. ``False`` returns the best estimate
        with a non-success ``sol.result`` instead — for ``vmap``ed
        hosts where one cell hitting the cap shouldn't abort the
        batch.
    reset
        ``True`` clears every override (back to per-call
        :class:`SolverConfig` defaults). Applied before the
        ``rtol`` / ``atol`` / ``max_steps`` / ``throw`` kwargs, so
        ``configure(reset=True, rtol=1e-6)`` is "clear, then set
        rtol=1e-6."

    **JIT cache contract.** ``solve_ivp`` is called from inside
    ``@jax.jit``-decorated functions (e.g. ``_mam_amicphys_1subarea_clear``,
    ``run_step``). The overrides are read at *trace* time and baked
    into the cached JIT binary. Reconfiguring after a code path has
    been traced has no effect on the cached binary — only a new trace
    (e.g., a different array shape or first call after process start)
    picks up the new values. **Pattern**: call :func:`configure` once
    at process startup, before any traced path runs. Reconfiguring
    mid-run is supported semantically but only takes effect on
    uncompiled call sites.

    **Thread safety.** :data:`_OVERRIDE` is a module-level mutable
    dict with no locking. Safe for single-threaded use and for
    process-per-device parallelism (each process has its own
    ``_OVERRIDE``); a single-process multi-threaded host calling
    :func:`configure` concurrently can observe non-deterministic
    interleaving. Set once at startup.
    """
    if reset:
        for k in _OVERRIDE:
            _OVERRIDE[k] = None
    if rtol is not None:
        _OVERRIDE["rtol"] = float(rtol)
    if atol is not None:
        _OVERRIDE["atol"] = float(atol)
    if max_steps is not None:
        _OVERRIDE["max_steps"] = int(max_steps)
    if throw is not None:
        _OVERRIDE["throw"] = bool(throw)


@dataclass(frozen=True)
class SolverResult:
    """Standardized return from `solve_ivp`.

    `ts` and `ys` carry whatever the caller's `SaveAt` requested
    (default: endpoint only — `ts.shape == (1,)`, `ys[-1]` is the
    terminal state). `stats` carries diffrax's step counters so
    adaptive-controller behavior is observable.
    """

    ts: jnp.ndarray
    ys: Any
    stats: dict


def solve_ivp(
    rhs: Callable,
    y0: Any,
    t0: float,
    t1: float,
    args: Any = None,
    saveat: Optional[diffrax.SaveAt] = None,
    config: SolverConfig = SolverConfig(),
) -> SolverResult:
    """Integrate dy/dt = rhs(t, y, args) from t0 to t1.

    Default `saveat=None` records `t1` only (endpoint-fast path;
    read terminal state via `result.ys[-1]`). Pass
    `diffrax.SaveAt(ts=...)` to record a trajectory; pass
    `diffrax.SaveAt(t0=True, t1=True)` for trapezoidal-average
    use cases that need `ys[0]` as well.

    Inputs to the underlying diffrax solver are layered:

    1. **Per-call `SolverConfig`** — the tight defaults baked into each
       site (rtol=1e-9, atol=1e-12 typical).
    2. **Process-global overrides** set via :func:`configure` — a
       host-level speed/accuracy/robustness knob applied on top of
       (1). Read at trace time; see :func:`configure` for the JIT-cache
       contract.

    JIT-traceable in `y0` / `args`; not in `config`, `saveat`, or the
    `configure`-set overrides (those are captured into the trace).
    """
    rtol = _OVERRIDE["rtol"] if _OVERRIDE["rtol"] is not None else config.rtol
    atol = _OVERRIDE["atol"] if _OVERRIDE["atol"] is not None else config.atol
    max_steps = (_OVERRIDE["max_steps"] if _OVERRIDE["max_steps"] is not None
                 else config.max_steps)
    throw = _OVERRIDE["throw"] if _OVERRIDE["throw"] is not None else True
    solver_cls = getattr(diffrax, config.solver)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        solver_cls(),
        t0=t0,
        t1=t1,
        dt0=config.dt0,
        y0=y0,
        args=args,
        saveat=saveat if saveat is not None else diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        throw=throw,
    )
    return SolverResult(ts=sol.ts, ys=sol.ys, stats=sol.stats)
