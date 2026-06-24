"""Tests for ``mam4_jax.solvers`` — process-global :func:`configure` hook.

Locks in the override layering so a future refactor of
:func:`solve_ivp` can't silently break the speed/accuracy knob
documented in plan 021.

The tests use a stiff ODE (``dy/dt = -1000·y`` over ``[0, 1]``) with
known analytical solution so the override behavior is verifiable
without depending on any MAM4 fixture.
"""
from __future__ import annotations

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401 — enables jax_enable_x64
from mam4_jax import solvers


@pytest.fixture(autouse=True)
def _reset_overrides_around_each_test():
    """Reset process-global overrides before AND after every test —
    so test order doesn't matter and a failure mid-test can't leak
    state into later tests."""
    solvers.configure(reset=True)
    yield
    solvers.configure(reset=True)


def _stiff_rhs(t, y, args):
    """Simple stiff scalar ODE: dy/dt = -1000·y. Analytical solution
    ``y(t) = y0 · exp(-1000·t)``."""
    return -1000.0 * y


def _solve_stiff(rtol_override=None, atol_override=None, max_steps_override=None,
                 throw_override=None,
                 t1: float = 0.01, y0: float = 1.0,
                 config: solvers.SolverConfig | None = None) -> solvers.SolverResult:
    """Helper: configure() overrides then solve the stiff ODE end-to-end."""
    if rtol_override is not None or atol_override is not None or \
       max_steps_override is not None or throw_override is not None:
        solvers.configure(
            rtol=rtol_override, atol=atol_override,
            max_steps=max_steps_override, throw=throw_override,
        )
    if config is None:
        config = solvers.SolverConfig()
    return solvers.solve_ivp(
        _stiff_rhs,
        y0=jnp.asarray(y0),
        t0=0.0, t1=t1,
        config=config,
    )


# ---------------------------------------------------------------------------
# Behavior: configure() actually affects the adaptive step count
# ---------------------------------------------------------------------------

def test_configure_rtol_reduces_step_count() -> None:
    """A looser ``rtol`` reduces the adaptive PI-controller's step
    count — the proximate observable that motivates the configure()
    hook (per plan 021's empirical 2.8× speedup at rtol=1e-6)."""
    result_tight = _solve_stiff()  # default SolverConfig: rtol=1e-9
    steps_tight = int(result_tight.stats["num_steps"])

    result_loose = _solve_stiff(rtol_override=1e-3, atol_override=1e-6)
    steps_loose = int(result_loose.stats["num_steps"])

    assert steps_loose < steps_tight, (
        f"loose tolerances should shorten step count, "
        f"got tight={steps_tight} loose={steps_loose}"
    )
    # Both should still hit the analytical answer at the requested
    # accuracy. t1=0.01 → exp(-10) ≈ 4.5e-5 (well-representable in
    # float64). Loose tolerances degrade accuracy but it should stay
    # within the loose-tolerance bound.
    expected = np.exp(-10.0)
    np.testing.assert_allclose(float(result_tight.ys[-1]), expected, rtol=1e-6)
    np.testing.assert_allclose(float(result_loose.ys[-1]), expected, rtol=5e-2)


def test_configure_default_is_no_op() -> None:
    """``configure()`` with all ``None`` leaves behavior identical to
    the per-call ``SolverConfig`` baseline. Locks in the safety
    property: a host that imports the module without calling
    ``configure`` gets unchanged upstream behavior."""
    result_a = _solve_stiff()
    solvers.configure()  # all None — explicit no-op
    result_b = _solve_stiff()
    # Same step count (exact). Same ys (exact — no source of nondeterminism).
    assert int(result_a.stats["num_steps"]) == int(result_b.stats["num_steps"])
    np.testing.assert_array_equal(np.asarray(result_a.ys), np.asarray(result_b.ys))


# ---------------------------------------------------------------------------
# Robustness: throw=False returns a non-success result instead of raising
# ---------------------------------------------------------------------------

def test_configure_throw_false_does_not_raise_on_step_exhaustion() -> None:
    """With ``throw=False`` AND a deliberately tiny ``max_steps``, the
    solver should exhaust its step budget and return without raising.
    The result's ``stats`` still surfaces the diagnostic.

    This is the load-bearing property for batched hosts (vmap): one
    pathological cell can no longer abort the whole batch. The host
    is responsible for gating/logging non-finite or non-success
    cells.
    """
    # Tight max_steps will be exhausted on this stiff problem
    # even at default tolerances.
    result = _solve_stiff(
        max_steps_override=4, throw_override=False,
        config=solvers.SolverConfig(rtol=1e-12, atol=1e-15, max_steps=4),
    )
    # Solver returned (didn't raise). The result may not match the
    # analytical answer — that's fine; the test is that we didn't crash.
    assert "num_steps" in result.stats
    assert int(result.stats["num_steps"]) <= 4


def test_configure_throw_true_default_raises_on_step_exhaustion() -> None:
    """Mirror of the above: with the default ``throw`` behavior (None →
    True inside ``solve_ivp``), exhausting ``max_steps`` raises. Locks
    in the upstream-diffrax default so a regression to "silently
    swallow" doesn't slip in."""
    with pytest.raises(Exception):  # noqa: BLE001 — diffrax raises a custom type
        _solve_stiff(
            config=solvers.SolverConfig(rtol=1e-12, atol=1e-15, max_steps=4),
        )


# ---------------------------------------------------------------------------
# Reset semantics
# ---------------------------------------------------------------------------

def test_configure_reset_clears_overrides() -> None:
    """``configure(reset=True)`` clears every override back to None.
    A subsequent ``solve_ivp`` call falls back to per-call
    ``SolverConfig`` defaults."""
    # Set a loose override, verify it takes effect.
    result_loose = _solve_stiff(rtol_override=1e-3, atol_override=1e-6)
    steps_loose = int(result_loose.stats["num_steps"])

    # Reset, verify the next call matches the default-tolerance baseline.
    solvers.configure(reset=True)
    result_reset = _solve_stiff()
    steps_reset = int(result_reset.stats["num_steps"])

    # After reset we expect the same step count as a fresh process
    # would see (no override). We can't know the exact value without
    # running it, so just verify it differs from the loose case.
    assert steps_reset != steps_loose, (
        f"reset didn't clear override: still {steps_reset} steps "
        f"(loose was {steps_loose})"
    )


def test_configure_reset_with_kwargs_applies_kwargs_after_reset() -> None:
    """``configure(reset=True, rtol=1e-6)`` clears everything then sets
    rtol — useful for "switch profile" patterns."""
    # Pre-set throw=False
    solvers.configure(throw=False)
    assert solvers._OVERRIDE["throw"] is False

    # Reset clears, then rtol kwarg applies
    solvers.configure(reset=True, rtol=1e-6)
    assert solvers._OVERRIDE["throw"] is None  # cleared
    assert solvers._OVERRIDE["rtol"] == 1e-6   # newly set
