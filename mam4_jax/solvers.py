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
    """Per-call solver configuration consumed by `solve_ivp`."""

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
_OVERRIDE: dict = {"rtol": None, "atol": None, "max_steps": None, "throw": None}


def configure(rtol=None, atol=None, max_steps=None, throw=None) -> None:
    """Set process-global solver overrides (any ``None`` leaves that unchanged)."""
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

    JIT-traceable in `y0` / `args`; not in `config` or `saveat`.
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
