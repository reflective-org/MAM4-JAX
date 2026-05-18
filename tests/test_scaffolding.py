"""M1 acceptance test: the package imports, float64 is on, stubs raise."""
from __future__ import annotations

import importlib

import jax
import pytest


# Process modules whose stubs must raise NotImplementedError.
PROCESS_MODULES: tuple[str, ...] = (
    "calcsize",
    "wateruptake",
    "gasaerexch",
    "newnuc",
    "coag",
    "rename",
    "amicphys",
)


def test_package_imports_cleanly() -> None:
    import mam4_jax

    assert mam4_jax.__version__ == "0.0.1"


def test_x64_enabled() -> None:
    import mam4_jax

    assert mam4_jax.x64_enabled is True
    assert jax.config.read("jax_enable_x64") is True


def test_default_dtype_is_float64() -> None:
    import mam4_jax  # noqa: F401  - import for the side-effect of enabling x64

    import jax.numpy as jnp

    assert jnp.array(1.0).dtype == jnp.float64


def test_constants_match_mam4_mom_config() -> None:
    from mam4_jax import data

    assert data.PCNST == 35
    assert data.NTOT_AMODE == 4
    assert data.NTOT_ASPECTYPE == 9
    assert data.MAXD_ASPECTYPE == 14
    assert data.NSPEC_AMODE == (7, 4, 7, 3)
    assert data.MODE_NAMES == ("accum", "aitken", "coarse", "primary_carbon")


def test_sentinel_tables_raise_on_access() -> None:
    import jax.numpy as jnp

    from mam4_jax import data

    tables = data.make_sentinel_tables()
    q = jnp.zeros(data.PCNST)

    with pytest.raises(NotImplementedError):
        data.get_number(q, mode=0, tables=tables)
    with pytest.raises(NotImplementedError):
        data.get_mass(q, mode=0, species_slot=0, tables=tables)


@pytest.mark.parametrize("module_name", PROCESS_MODULES)
def test_process_stub_raises(module_name: str) -> None:
    mod = importlib.import_module(f"mam4_jax.processes.{module_name}")
    fn = getattr(mod, module_name)

    with pytest.raises(NotImplementedError):
        fn(state=None, params=None, config=None)
