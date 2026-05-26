"""M1 acceptance test: the package imports, float64 is on, stubs raise."""
from __future__ import annotations

import importlib
from pathlib import Path

import jax
import numpy as np
import pytest


# Process modules whose stubs must raise NotImplementedError. Real
# implementations have their own coverage:
#   - wateruptake → tests/test_wateruptake.py (filled by M3.4 PR-C)
#   - calcsize    → tests/test_calcsize.py    (filled by M3.5 PR-A)
#   - amicphys    → tests/test_amicphys.py    (filled by M3.6 PR-A as
#                   an orchestration shell with stub sub-processes;
#                   real physics lands in M3.6 PR-B through PR-E)
#
# `gasaerexch`, `newnuc`, `coag`, `rename` are NOT in this list because
# the standalone modules with those names are dead code in the box
# model (see docs/ARCHITECTURE.md). The corresponding physics lives
# inside `mam4_jax/processes/amicphys.py` as `_mam_*_1subarea` helpers.
PROCESS_MODULES: tuple[str, ...] = (
    "gasaerexch",
    "newnuc",
    "coag",
    "rename",
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


def test_amicphys_init_tables_match_npz_reference() -> None:
    """The hard-coded amicphys init constants match the captured tables (M3.6 PR-C)."""
    from mam4_jax import data

    ref = np.load(INDICES_NPZ, allow_pickle=False)
    assert int(ref["amicphys_ngas"])    == data.AMICPHYS_NGAS
    assert int(ref["amicphys_naer"])    == data.AMICPHYS_NAER
    assert int(ref["amicphys_max_gas"]) == data.AMICPHYS_MAX_GAS
    assert int(ref["amicphys_max_aer"]) == data.AMICPHYS_MAX_AER
    np.testing.assert_array_equal(ref["pcnst_lmap_gas"],   data.LMAP_GAS)
    np.testing.assert_array_equal(ref["pcnst_lmap_num"],   data.LMAP_NUM)
    np.testing.assert_array_equal(ref["pcnst_lmap_numcw"], data.LMAP_NUMCW)
    np.testing.assert_array_equal(ref["pcnst_lmap_aer"],   data.LMAP_AER)
    np.testing.assert_array_equal(ref["pcnst_lmap_aercw"], data.LMAP_AERCW)
    np.testing.assert_allclose(ref["fcvt_gas"], data.FCVT_GAS, rtol=1e-14)
    np.testing.assert_array_equal(ref["fcvt_aer"], data.FCVT_AER)
    assert float(ref["fcvt_num"]) == data.FCVT_NUM
    assert float(ref["fcvt_wtr"]) == data.FCVT_WTR

    # Cross-check: amicphys's lmap_num must equal modal_aero_data's
    # numptr_amode (different tables, same physical content).
    np.testing.assert_array_equal(data.LMAP_NUM, data.NUMPTR_AMODE)

    # FAC_M2V_AER parity: PR-B captured it per-record in the rename
    # snapshot. Constant across the run, must match the hard-coded constant.
    rename_ref = np.load(
        INDICES_NPZ.parent.parent / "per_process" / "rename_before.npz",
        allow_pickle=False,
    )
    np.testing.assert_allclose(rename_ref["fac_m2v_aer"][0], data.FAC_M2V_AER,
                               rtol=1e-14, atol=0.0)

    # mwdry and adv_mass (driver-side mmr→vmr conversion factors).
    assert float(ref["mwdry"]) == data.MWDRY
    np.testing.assert_array_equal(ref["adv_mass"], data.ADV_MASS)

    # Gas-property constants (M3.6 PR-D — for gasaerexch).
    assert float(ref["vmdry"]) == data.VMDRY
    np.testing.assert_array_equal(ref["mw_gas"],         data.MW_GAS)
    np.testing.assert_array_equal(ref["vol_molar_gas"],  data.VOL_MOLAR_GAS)
    np.testing.assert_array_equal(ref["accom_coef_gas"], data.ACCOM_COEF_GAS)

    # SOA-specific constants (M3.6 PR-E — for soaexch).
    assert int(ref["amicphys_npoa"]) == data.AMICPHYS_NPOA
    assert int(ref["amicphys_nsoa"]) == data.AMICPHYS_NSOA
    # Fortran 1-based → 0-based: subtract 1.
    assert int(ref["amicphys_iaer_pom"]) - 1 == data.AMICPHYS_IAER_POM
    assert int(ref["amicphys_iaer_soa"]) - 1 == data.AMICPHYS_IAER_SOA
    assert int(ref["amicphys_npca"])     - 1 == data.AMICPHYS_NPCA
    np.testing.assert_array_equal(ref["mode_aging_optaa"], data.MODE_AGING_OPTAA)
    # Boolean form of lptr2_soa_a_amode (only the > 0 check is used in soaexch).
    lptr2_present = ref["lptr2_soa_a_amode"] > 0
    np.testing.assert_array_equal(lptr2_present, data.LPTR2_SOA_A_AMODE_PRESENT)

    # Host molecular weights and density (M3.6 PR-F2 — for newnuc dispatcher).
    assert float(ref["mw_so4a_host"])   == data.MW_SO4A_HOST
    assert float(ref["mw_nh4a_host"])   == data.MW_NH4A_HOST
    assert float(ref["dens_so4a_host"]) == data.DENS_SO4A_HOST


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


def test_solvers_smoke() -> None:
    """solve_ivp integrates dy/dt = -y over [0, 1]; result ~= exp(-1)."""
    import jax.numpy as jnp
    from mam4_jax import solvers

    assert isinstance(solvers.SolverConfig(), solvers.SolverConfig)

    def rhs(t, y, args):
        return -y

    result = solvers.solve_ivp(rhs, y0=jnp.array(1.0), t0=0.0, t1=1.0)
    np.testing.assert_allclose(result.ys[-1], np.exp(-1.0), rtol=1e-8)
    assert "num_accepted_steps" in result.stats
