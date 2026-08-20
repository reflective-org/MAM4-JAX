"""Topology must reach jitted kernels as a static argument, never a global.

THE HAZARD, reproduced before it was fixed. jit caches are not keyed on module
globals read inside a function body, so a kernel traced under one topology keeps
returning that topology's answer after ``set_topology``:

    call 1 (e3sm)       3.031219e+20
    call 2 (after swap) 3.031219e+20   <- stale
    correct for alt     3.789024e+19   <- 8x out

No error, no warning, wrong physics. These tests pin both halves of the fix:
``get_topology()`` now refuses to run inside a trace, and ``topology_jit``
provides the correct alternative.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.core import data
from mam4_jax.core.topology import (
    _REGISTRY,
    _TRACE_PROBE_AVAILABLE,
    get_topology,
    set_topology,
    topology_jit,
    trace_policy,
)


@pytest.fixture
def restore_policy():
    original = trace_policy()
    yield
    trace_policy(original)


@pytest.fixture
def restore_active():
    original = get_topology()
    yield
    set_topology(original)


def _alt() -> "data.Topology":     # type: ignore[name-defined]
    """A topology differing only in accum dgnum, so voltonumb differs ~8x."""
    return dataclasses.replace(
        data.E3SM_MAM4_MOM, name="alt_probe",
        dgnum_amode=(2.2e-07, 0.0260e-6, 2.000e-6, 0.050e-6),
    )


def test_trace_probe_is_available() -> None:
    """The guard leans on a private JAX API and fails OPEN if it moves.

    Failing open is right for a safety net, but a silent degradation would
    quietly remove the protection, so it is asserted rather than assumed. If
    this fails after a JAX upgrade, fix the probe -- do not delete the test.
    """
    assert _TRACE_PROBE_AVAILABLE, (
        "jax._src.core.trace_ctx moved; the jit-staleness guard is now a no-op"
    )
    # The flag alone is not enough: if a future JAX keeps trace_ctx but renames
    # EvalTrace, the flag stays True while the probe reports "tracing" in eager
    # mode -- and then EVERY get_topology() call raises. Name that directly.
    from mam4_jax.core.topology import _inside_jit_trace
    assert _inside_jit_trace() is False


def test_get_topology_inside_jit_raises() -> None:
    @jax.jit
    def unsafe(x):
        return x * jnp.asarray(get_topology().voltonumb_amode)[0]

    with pytest.raises(RuntimeError, match="called inside a jit trace"):
        unsafe(jnp.float64(1.0))


def test_get_topology_outside_jit_is_unaffected() -> None:
    assert get_topology() is data.E3SM_MAM4_MOM


def test_trace_policy_allow_restores_the_old_behaviour(restore_policy) -> None:
    """The escape hatch works -- and demonstrates the bug it protects against."""
    trace_policy("allow")

    @jax.jit
    def unsafe(x):
        return x * jnp.asarray(get_topology().voltonumb_amode)[0]

    first = float(unsafe(jnp.float64(1.0)))
    try:
        set_topology(_alt())
        second = float(unsafe(jnp.float64(1.0)))
    finally:
        set_topology(data.E3SM_MAM4_MOM)
        # set_topology registers by side effect, so drop the probe rather than
        # leaving available_topologies() polluted for the rest of the session.
        _REGISTRY.pop("alt_probe", None)

    # This IS the hazard: same cache entry, so the answer does not change.
    assert second == first
    assert second != pytest.approx(float(_alt().voltonumb_amode[0]))


def test_trace_policy_warn(restore_policy, restore_active) -> None:
    trace_policy("warn")

    @jax.jit
    def unsafe(x):
        return x * jnp.asarray(get_topology().voltonumb_amode)[0]

    with pytest.warns(RuntimeWarning, match="called inside a jit trace"):
        unsafe(jnp.float64(1.0))


def test_trace_policy_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="must be 'error', 'warn' or 'allow'"):
        trace_policy("off")


# ---------------------------------------------------------------------------
# The safe path. This is the test plan 024 requires of PR C: switching topology
# must CHANGE a jitted result.
# ---------------------------------------------------------------------------

def test_topology_jit_retraces_on_topology_change() -> None:
    @topology_jit
    def kernel(x, *, topology):
        return x * jnp.asarray(topology.voltonumb_amode)[0]

    alt = _alt()
    from_e3sm = float(kernel(jnp.float64(1.0), topology=data.E3SM_MAM4_MOM))
    from_alt  = float(kernel(jnp.float64(1.0), topology=alt))

    assert from_e3sm == pytest.approx(float(data.E3SM_MAM4_MOM.voltonumb_amode[0]))
    assert from_alt  == pytest.approx(float(alt.voltonumb_amode[0]))
    assert from_alt != from_e3sm, "topology switch must change the jitted result"


def test_topology_jit_is_stable_for_the_same_topology() -> None:
    """Same topology must hit the cache, not retrace -- otherwise the static
    argument is defeating jit rather than parameterising it."""
    calls = {"traces": 0}

    @topology_jit
    def kernel(x, *, topology):
        calls["traces"] += 1
        return x * jnp.asarray(topology.voltonumb_amode)[0]

    for _ in range(3):
        kernel(jnp.float64(1.0), topology=data.E3SM_MAM4_MOM)
    assert calls["traces"] == 1


def test_topology_jit_composes_with_other_static_argnames() -> None:
    @topology_jit(static_argnames=("mode",))
    def kernel(x, *, topology, mode):
        return x * jnp.asarray(topology.voltonumb_amode)[mode]

    assert float(kernel(jnp.float64(1.0), topology=data.E3SM_MAM4_MOM, mode=1)) \
        == pytest.approx(float(data.E3SM_MAM4_MOM.voltonumb_amode[1]))


def test_topology_jit_kernel_can_use_get_topology_at_the_call_site() -> None:
    """The intended idiom: resolve the global OUTSIDE the traced function."""
    @topology_jit
    def kernel(x, *, topology):
        return x * jnp.asarray(topology.voltonumb_amode)[0]

    result = float(kernel(jnp.float64(1.0), topology=get_topology()))
    assert result == pytest.approx(float(get_topology().voltonumb_amode[0]))


# ---------------------------------------------------------------------------
# Gaps found reviewing this branch. Both are cases where the failure is SILENT.
# ---------------------------------------------------------------------------

def test_topology_jit_factory_is_reusable() -> None:
    """A reused decorator factory must not lose the caller's static argnames.

    The first implementation used ``jit_kwargs.pop(...)``, which mutated the
    dict closed over by the decorator, so every function after the first got
    only ``('topology',)``. When the dropped argument is used for a Python
    branch that raises TracerBoolConversionError; when it is used only for
    indexing -- as in the composition test above -- it degrades quietly into a
    dynamic argument and nothing complains.
    """
    deco = topology_jit(static_argnames=("mode",))

    @deco
    def first(x, *, topology, mode):
        return x * jnp.asarray(topology.voltonumb_amode)[mode]

    @deco
    def second(x, *, topology, mode):
        return x * jnp.asarray(topology.voltonumb_amode)[mode]

    for fn in (first, second):
        assert "mode" in fn._jit_info.static_argnames
        assert "topology" in fn._jit_info.static_argnames

    # And behaviourally: `mode` must be usable in a Python branch.
    @deco
    def branching(x, *, topology, mode):
        if mode == 0:                      # requires a genuinely static `mode`
            return x * jnp.asarray(topology.voltonumb_amode)[0]
        return x

    assert float(branching(jnp.float64(1.0),
                           topology=data.E3SM_MAM4_MOM, mode=0)) \
        == pytest.approx(float(data.E3SM_MAM4_MOM.voltonumb_amode[0]))


def test_topology_passed_positionally_is_still_static() -> None:
    """`static_argnames` covers POSITIONAL_OR_KEYWORD params, so positional
    passing is static too -- asserted rather than assumed, since a silent
    downgrade to a dynamic argument would reintroduce the staleness class."""
    traces = {"n": 0}

    @topology_jit
    def kernel(x, topology):               # positional, not keyword-only
        traces["n"] += 1
        return x * jnp.asarray(topology.voltonumb_amode)[0]

    alt = _alt()
    a = float(kernel(jnp.float64(1.0), data.E3SM_MAM4_MOM))
    b = float(kernel(jnp.float64(1.0), data.E3SM_MAM4_MOM))
    c = float(kernel(jnp.float64(1.0), alt))

    assert a == b                                   # cache hit
    assert traces["n"] == 2                         # one trace per topology
    assert c == pytest.approx(float(alt.voltonumb_amode[0]))
    assert c != a


def test_structurally_identical_topologies_share_a_trace() -> None:
    """Distinct-but-equal instances must hit the same cache entry."""
    traces = {"n": 0}

    @topology_jit
    def kernel(x, *, topology):
        traces["n"] += 1
        return x * jnp.asarray(topology.voltonumb_amode)[0]

    twin = dataclasses.replace(data.E3SM_MAM4_MOM)
    assert twin == data.E3SM_MAM4_MOM and twin is not data.E3SM_MAM4_MOM
    kernel(jnp.float64(1.0), topology=data.E3SM_MAM4_MOM)
    kernel(jnp.float64(1.0), topology=twin)
    assert traces["n"] == 1
