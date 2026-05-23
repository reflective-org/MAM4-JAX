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
    `diffrax.SaveAt(ts=...)` to record a trajectory.

    JIT-traceable in `y0` / `args`; not in `config` or `saveat`.
    Body is filled in by PR-D1 (soaexch port).
    """
    raise NotImplementedError(
        "solve_ivp is a PR-I1 skeleton; the diffrax integration is "
        "wired in by PR-D1 (soaexch port). See "
        "docs/plans/015-diffrax-infra.md."
    )
