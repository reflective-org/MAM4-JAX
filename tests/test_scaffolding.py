"""M1 acceptance test: the package imports, float64 is on, stubs raise."""
from __future__ import annotations

import importlib
from pathlib import Path

import jax
import numpy as np
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

INDICES_NPZ = (
    Path(__file__).resolve().parent / "reference" / "indices" / "reference.npz"
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


def test_index_tables_populated() -> None:
    """The hard-coded INDEX_TABLES contain the canonical MAM4-MOM values."""
    from mam4_jax import data

    # numptr_amode: one number-tracer index per mode.
    assert tuple(data.INDEX_TABLES.numptr_amode.tolist()) == (17, 22, 30, 34)

    # nspec_amode constraint: lmassptr_amode has valid (≠ -1) entries in
    # exactly the first nspec_amode[mode] slots, and -1 elsewhere.
    for mode, n in enumerate(data.NSPEC_AMODE):
        row = data.INDEX_TABLES.lmassptr_amode[mode]
        valid = (row >= 0).sum().item()
        assert valid == n, f"mode {mode}: expected {n} valid slots, got {valid}"


def test_index_tables_match_npz_reference() -> None:
    """The hard-coded constants match the committed Fortran capture."""
    from mam4_jax import data

    ref = np.load(INDICES_NPZ, allow_pickle=False)
    assert int(ref["ntot_amode"]) == data.NTOT_AMODE
    assert int(ref["ntot_aspectype"]) == data.NTOT_ASPECTYPE
    assert int(ref["maxd_aspectype"]) == data.MAXD_ASPECTYPE
    np.testing.assert_array_equal(ref["numptr_amode"],     data.INDEX_TABLES.numptr_amode)
    np.testing.assert_array_equal(ref["numptrcw_amode"],   data.INDEX_TABLES.numptrcw_amode)
    np.testing.assert_array_equal(ref["lmassptr_amode"],   data.INDEX_TABLES.lmassptr_amode)
    np.testing.assert_array_equal(ref["lmassptrcw_amode"], data.INDEX_TABLES.lmassptrcw_amode)
    np.testing.assert_array_equal(ref["nspec_amode"], np.asarray(data.NSPEC_AMODE, dtype=np.int32))
    np.testing.assert_array_equal(ref["modename_amode"], np.asarray(data.MODE_NAMES))
    np.testing.assert_array_equal(ref["specname_amode"], np.asarray(data.SPECNAME_AMODE))


def test_get_number_returns_slice() -> None:
    import jax.numpy as jnp

    from mam4_jax import data

    q = jnp.arange(data.PCNST, dtype=jnp.float64)
    assert float(data.get_number(q, mode=0)) == 17.0  # NUMPTR_AMODE[0]
    assert float(data.get_number(q, mode=3)) == 34.0  # NUMPTR_AMODE[3]


def test_get_mass_returns_slice() -> None:
    import jax.numpy as jnp

    from mam4_jax import data

    q = jnp.arange(data.PCNST, dtype=jnp.float64)
    # Mode 0 (accum), slot 0: lmassptr_amode[0, 0] = 10 (sulfate)
    assert float(data.get_mass(q, mode=0, species_slot=0)) == 10.0
    # Mode 3 (primary_carbon), slot 1: lmassptr_amode[3, 1] = 32 (bc)
    assert float(data.get_mass(q, mode=3, species_slot=1)) == 32.0


def test_get_mass_raises_on_unused_slot() -> None:
    import jax.numpy as jnp

    from mam4_jax import data

    q = jnp.zeros(data.PCNST, dtype=jnp.float64)
    # Mode 1 (aitken) has nspec=4, so slot 5 is unused.
    with pytest.raises(NotImplementedError):
        data.get_mass(q, mode=1, species_slot=5)


def test_get_mass_by_species_name() -> None:
    import jax.numpy as jnp

    from mam4_jax import data

    q = jnp.arange(data.PCNST, dtype=jnp.float64)
    # Sulfate ('sulfate') in mode 0 (accum) is at slot 0 → pcnst index 10.
    assert float(data.get_mass_by_species_name(q, 0, "sulfate")) == 10.0
    # Sulfate in mode 3 (primary_carbon) doesn't exist.
    with pytest.raises(KeyError):
        data.get_mass_by_species_name(q, 3, "sulfate")
    with pytest.raises(KeyError):
        data.get_mass_by_species_name(q, 0, "not-a-species")


@pytest.mark.parametrize("module_name", PROCESS_MODULES)
def test_process_stub_raises(module_name: str) -> None:
    mod = importlib.import_module(f"mam4_jax.processes.{module_name}")
    fn = getattr(mod, module_name)

    with pytest.raises(NotImplementedError):
        fn(state=None, params=None, config=None)
