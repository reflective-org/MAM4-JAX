"""The `Topology` parameter axis (plan 024 revision 2, PR B).

Two jobs:

1. **Bit-identity.** Every derived quantity the kernels consume must come out of
   the registered E3SM instance exactly as `data.py`'s module constants have it.
   This is what backs the claim that introducing the axis leaves the E3SM path
   unchanged, and it is what will catch a mistranscription when the CAM and MAM5
   instances arrive.
2. **The validation actually rejects.** A validator nobody has seen fail is not
   evidence of anything, so each rule gets a case that trips it.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax import data
from mam4_jax.topology import (
    Topology,
    available_topologies,
    get_topology,
    set_topology,
)


# ---------------------------------------------------------------------------
# 1. Bit-identity with the module constants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_name,module_name", [
    ("nmodes",           "NTOT_AMODE"),
    ("pcnst",            "PCNST"),
    ("mode_names",       "MODE_NAMES"),
    ("specname_amode",   "SPECNAME_AMODE"),
    ("nspec_amode",      "NSPEC_AMODE"),
    ("numptr_amode",     "NUMPTR_AMODE"),
    ("numptrcw_amode",   "NUMPTRCW_AMODE"),
    ("lspectype_amode",  "LSPECTYPE_AMODE"),
    ("lmassptr_amode",   "LMASSPTR_AMODE"),
    ("lmassptrcw_amode", "LMASSPTRCW_AMODE"),
    ("specdens_amode",   "SPECDENS_AMODE"),
    ("spechygro_amode",  "SPECHYGRO_AMODE"),
    ("sigmag_amode",     "SIGMAG_AMODE"),
    ("dgnum_amode",      "DGNUM_AMODE"),
    ("dgnumlo_amode",    "DGNUMLO_AMODE"),
    ("dgnumhi_amode",    "DGNUMHI_AMODE"),
    ("rhcrystal_amode",  "RHCRYSTAL_AMODE"),
    ("rhdeliques_amode", "RHDELIQUES_AMODE"),
])
def test_primitive_fields_match_module_constants(field_name, module_name) -> None:
    assert getattr(data.E3SM_MAM4_MOM, field_name) == getattr(data, module_name)


@pytest.mark.parametrize("prop_name,module_name", [
    ("alnsg_amode",       "ALNSG_AMODE"),
    ("dumfac_amode",      "DUMFAC_AMODE"),
    ("voltonumb_amode",   "VOLTONUMB_AMODE"),
    ("voltonumblo_amode", "VOLTONUMBLO_AMODE"),
    ("voltonumbhi_amode", "VOLTONUMBHI_AMODE"),
])
def test_derived_quantities_are_bit_identical(prop_name, module_name) -> None:
    """Bitwise, not approximate.

    The derived values are recomputed from the same primitives by the same
    expression, so anything short of bit-equality means the expression drifted.
    """
    from_topology = np.asarray(getattr(data.E3SM_MAM4_MOM, prop_name))
    from_module   = np.asarray(getattr(data, module_name))
    assert from_topology.shape == from_module.shape
    assert np.array_equal(from_topology, from_module), (
        f"{prop_name} differs from {module_name}:\n"
        f"  topology: {from_topology!r}\n  module:   {from_module!r}"
    )


def test_registry_and_active_default() -> None:
    assert "e3sm_mam4_mom" in available_topologies()
    assert get_topology() is data.E3SM_MAM4_MOM
    assert data.E3SM_MAM4_MOM.variant == "e3sm"
    assert data.E3SM_MAM4_MOM.is_mam5 is False


def test_set_topology_by_name_round_trips() -> None:
    original = get_topology()
    try:
        assert set_topology("e3sm_mam4_mom") is data.E3SM_MAM4_MOM
        assert get_topology() is data.E3SM_MAM4_MOM
    finally:
        set_topology(original)


def test_set_topology_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown topology"):
        set_topology("cam_mam9")


def test_mode_index_and_missing_mode() -> None:
    t = data.E3SM_MAM4_MOM
    assert t.mode_index("accum") == 0
    assert t.mode_index("aitken") == 1
    assert t.has_mode("coarse_strat") is False
    with pytest.raises(KeyError, match="no mode 'coarse_strat'"):
        t.mode_index("coarse_strat")


# ---------------------------------------------------------------------------
# 2. The validation rejects what it claims to
# ---------------------------------------------------------------------------

def _mutate(**overrides) -> Topology:
    """A copy of the E3SM instance with fields replaced -- re-runs validation."""
    return dataclasses.replace(data.E3SM_MAM4_MOM, **overrides)


def test_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant must be one of"):
        _mutate(variant="cesm2")


def test_rejects_e3sm_with_five_modes() -> None:
    """E3SMv1 has no coarse_strat, so ("e3sm", 5) must not be constructible."""
    with pytest.raises(ValueError, match="no 5-mode configuration"):
        _mutate(nmodes=5)


def test_rejects_bad_mode_count() -> None:
    with pytest.raises(ValueError, match="nmodes must be 4 or 5"):
        _mutate(variant="cesm", nmodes=7)


def test_rejects_per_mode_length_mismatch() -> None:
    with pytest.raises(ValueError, match="sigmag_amode has 3 entries"):
        _mutate(sigmag_amode=(1.8, 1.6, 1.8))


def test_rejects_species_property_length_mismatch() -> None:
    with pytest.raises(ValueError, match="specdens_amode has 3 entries"):
        _mutate(specdens_amode=(1770.0, 1770.0, 1770.0))


def test_rejects_sentinel_inside_declared_species_count() -> None:
    """A -1 in a populated slot yields zero-valued physics, not an error.

    That is the failure this rule exists to convert into a loud one, so it is
    worth a test of its own.
    """
    broken = list(data.E3SM_MAM4_MOM.lspectype_amode)
    row = list(broken[0])
    row[2] = -1                      # inside nspec_amode[0] = 7
    broken[0] = tuple(row)
    with pytest.raises(ValueError, match="within nspec_amode.*lspectype_amode is -1"):
        _mutate(lspectype_amode=tuple(broken))


def test_rejects_mass_pointer_outside_pcnst() -> None:
    broken = list(data.E3SM_MAM4_MOM.lmassptr_amode)
    row = list(broken[1])
    row[0] = data.PCNST + 5
    broken[1] = tuple(row)
    with pytest.raises(ValueError, match="points at pcnst index"):
        _mutate(lmassptr_amode=tuple(broken))


def test_rejects_number_pointer_outside_pcnst() -> None:
    with pytest.raises(ValueError, match="outside .0, pcnst"):
        _mutate(numptr_amode=(17, 22, 30, data.PCNST))


def test_rejects_inverted_size_bounds() -> None:
    with pytest.raises(ValueError, match="dgnumlo < dgnum < dgnumhi"):
        _mutate(dgnumhi_amode=(1.0e-9, 0.0520e-6, 4.000e-6, 0.100e-6))


def test_rejects_degenerate_sigmag() -> None:
    with pytest.raises(ValueError, match="sigmag = 1.0 must exceed 1"):
        _mutate(sigmag_amode=(1.0, 1.6, 1.8, 1.6))


def test_topology_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.E3SM_MAM4_MOM.nmodes = 5   # type: ignore[misc]
