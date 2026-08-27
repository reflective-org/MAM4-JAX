"""amicphys must refuse a cloud fraction it cannot represent.

Only the clear-sky sub-area is ported. With `cldn > 0` the Fortran splits the
cell and area-weights two different sub-area calculations; this port would
instead apply clear-sky physics to the whole cell and return a plausible answer.

The docstring on `_mam_amicphys_1gridcell` previously *claimed* such a call
"raises a clear error so future workflows don't silently get wrong physics" --
while the code immediately below declined to check and then never used `cldn` at
all. These tests exist so the claim and the behaviour cannot drift apart again.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.coupling.amicphys import _check_clear_sky, amicphys


def test_zero_cloud_fraction_is_accepted() -> None:
    """cldn = 0 is the supported case and must not be blocked."""
    _check_clear_sky(jnp.asarray(0.0))
    _check_clear_sky(0.0)
    _check_clear_sky(jnp.zeros(5))


def test_missing_cloud_fraction_is_accepted() -> None:
    """Absent means the caller never set it; the box driver defaults to clear."""
    _check_clear_sky(None)


@pytest.mark.parametrize("cldn", [0.3, 1.0, 1e-12])
def test_nonzero_cloud_fraction_is_refused(cldn: float) -> None:
    with pytest.raises(NotImplementedError, match="only the clear-sky sub-area"):
        _check_clear_sky(jnp.asarray(cldn))


def test_refusal_reports_the_magnitude() -> None:
    """The message should say what it saw -- a bare refusal sends the reader
    hunting for which field was wrong."""
    with pytest.raises(NotImplementedError, match="3.000e-01"):
        _check_clear_sky(jnp.asarray(0.3))


def test_any_nonzero_cell_refuses_not_just_the_mean() -> None:
    """A mostly-clear field with one cloudy cell is still unrepresentable."""
    field = jnp.zeros(10).at[7].set(0.5)
    with pytest.raises(NotImplementedError):
        _check_clear_sky(field)


def test_traced_cloud_fraction_returns_silently() -> None:
    """Under jit there is no readable magnitude, so nothing can be checked.

    It returns SILENTLY rather than warning. `driver.run_step` is jitted, so the
    traced case is the *normal* one -- a warning there would fire on every call,
    and a warning that always fires gets filtered, at which point it protects
    nothing while still reading as protection.

    The protection for that path lives outside the jit; see
    `test_the_driver_guards_before_tracing`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jax.jit(_check_clear_sky)(jnp.asarray(0.3))   # must not raise
    assert not caught, f"expected silence under trace, got {[str(w.message) for w in caught]}"


@pytest.mark.parametrize("entry", ["run_step", "run_timesteps"])
def test_the_driver_guards_before_tracing(entry: str) -> None:
    """This is where the guard has to work.

    `run_step` / `run_timesteps` are jitted, so a check *inside* them can never
    fire on a real call. The public names are therefore thin unjitted wrappers
    that validate first. Without them the main path had no protection at all --
    only direct `amicphys` calls did.
    """
    from mam4_jax import driver
    fn = getattr(driver, entry)
    args = ({"cldn": jnp.asarray(0.25)},) if entry == "run_step" else (
        {"cldn": jnp.asarray(0.25)}, 2)
    with pytest.raises(NotImplementedError, match="clear-sky"):
        fn(*args)


def test_the_public_names_are_not_themselves_jitted() -> None:
    """If these were jitted the guard would be inside the trace again, which is
    the bug this arrangement exists to avoid."""
    from mam4_jax import driver
    for name in ("run_step", "run_timesteps"):
        fn = getattr(driver, name)
        assert not hasattr(fn, "_jit_info"), (
            f"driver.{name} is jitted; the cldn guard would be unreachable"
        )


def test_the_guard_is_wired_into_the_public_entry() -> None:
    """It must run from amicphys(), not only when called directly."""
    with pytest.raises(NotImplementedError, match="clear-sky"):
        amicphys({"cldn": jnp.asarray(0.42)})
