"""The `Topology` parameter axis (plan 024 revision 2, PR B).

Two jobs:

1. **Bit-identity.** Every derived quantity the kernels consume must come out of
   the registered E3SM instance exactly as `data.py`'s module constants have it.
   This backs the claim that introducing the axis leaves the E3SM path unchanged.

   Scope of that claim, stated precisely because an earlier version of this
   docstring overstated it: these cases are parameterized against `data.py`'s
   E3SM module constants, so they say nothing about the CAM and MAM5 instances,
   which have no module-level counterpart to compare against. Those need their
   own tests against the extracted CAM tables (PRs C-E). The primitive-field
   cases compare objects that are literally identical (`data.py` passes the same
   tuple objects to the constructor), so they verify constructor WIRING -- a
   swapped `dgnumlo`/`dgnumhi` keyword -- not values.

2. **The validation actually rejects.** A validator nobody has seen fail is not
   evidence of anything, so every rule gets a case that trips it.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.core import data
from mam4_jax.core.topology import (
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
    """Byte-for-byte, not merely equal.

    The derived values are recomputed from the same primitives by the same
    expression, so anything short of bit-equality means the expression drifted.
    Compared via ``tobytes()`` rather than ``np.array_equal``: the latter is
    value equality, which treats -0.0 and 0.0 as equal and NaN as unequal. Moot
    for these strictly-positive finite quantities, but the point of the test is
    to be the strict check, so it should actually be one.
    """
    from_topology = np.asarray(getattr(data.E3SM_MAM4_MOM, prop_name))
    from_module   = np.asarray(getattr(data, module_name))
    assert from_topology.shape == from_module.shape
    assert from_topology.dtype == from_module.dtype
    assert from_topology.tobytes() == from_module.tobytes(), (
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


# ---------------------------------------------------------------------------
# 3. Rules that had no test until the PR-B review pointed it out. All three are
#    exactly what a hand-extracted CAM index table gets wrong, so they matter
#    more than the ones that were covered.
# ---------------------------------------------------------------------------

def test_rejects_slot_row_shorter_than_declared_species() -> None:
    """A short lmassptr row used to raise IndexError, not ValueError."""
    broken = list(data.E3SM_MAM4_MOM.lmassptr_amode)
    broken[0] = (10, 11)                       # nspec_amode[0] is 7
    with pytest.raises(ValueError, match=r"lmassptr_amode\[0\] has only 2 slots"):
        _mutate(lmassptr_amode=tuple(broken))


def test_rejects_species_index_past_end_of_species_list() -> None:
    broken = list(data.E3SM_MAM4_MOM.lspectype_amode)
    row = list(broken[0]); row[1] = len(data.SPECNAME_AMODE)
    broken[0] = tuple(row)
    with pytest.raises(ValueError, match="only 9 exist"):
        _mutate(lspectype_amode=tuple(broken))


def test_rejects_negative_species_count() -> None:
    """range(-1) is empty, so a negative count silently skips ALL validation
    for that mode -- the failure mode runs the wrong way."""
    with pytest.raises(ValueError, match=r"nspec_amode\[3\] = -1 outside"):
        _mutate(nspec_amode=(7, 4, 7, -1))


@pytest.mark.parametrize("label,kwargs", [
    ("two modes sharing a mass tracer",
     {"lmassptr_amode": (
         (10, 11, 12, 13, 14, 15, 16, -1, -1, -1, -1, -1, -1, -1),
         (10, 19, 20, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),   # 10 reused
         (23, 24, 25, 26, 27, 28, 29, -1, -1, -1, -1, -1, -1, -1),
         (31, 32, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))}),
    ("number pointer colliding with a mass tracer",
     {"numptr_amode": (10, 22, 30, 34)}),                            # 10 is mass
])
def test_rejects_duplicate_pcnst_claims(label, kwargs) -> None:
    """Double-claimed tracers silently double-count mass.

    Same class of quiet-wrong-physics as the -1 rule, and the specific thing
    hand-extracting CAM's tables produces.
    """
    with pytest.raises(ValueError, match="claimed by both"):
        _mutate(**kwargs)


def test_registration_does_not_implicitly_activate() -> None:
    """register_topology must not silently steal `active`.

    Registration used to activate whenever nothing was active, which meant the
    first import of any registering module could revert an explicit
    set_topology choice.
    """
    from mam4_jax.core.topology import register_topology
    original = get_topology()
    try:
        other = dataclasses.replace(data.E3SM_MAM4_MOM, name="e3sm_probe")
        register_topology(other)
        assert get_topology() is original
        assert "e3sm_probe" in available_topologies()
    finally:
        set_topology(original)


def test_topology_module_does_not_export_a_none_instance() -> None:
    """Regression: topology.E3SM_MAM4_MOM was declared None for data.py to
    populate, and data.py never assigned back -- so importing it from here
    silently yielded None."""
    import mam4_jax.core.topology as topo
    assert not hasattr(topo, "E3SM_MAM4_MOM")
    assert "E3SM_MAM4_MOM" not in topo.__all__


def test_topology_is_hashable_for_use_as_a_jit_static_arg() -> None:
    """Every field must stay hashable.

    The intended way to thread topology into jitted kernels (PR C) is as a
    static argument, which requires hashability. A list-valued field validates
    fine -- len() works -- and only fails at the eventual jit call site, far
    from construction.
    """
    assert isinstance(hash(data.E3SM_MAM4_MOM), int)
    for f in dataclasses.fields(data.E3SM_MAM4_MOM):
        value = getattr(data.E3SM_MAM4_MOM, f.name)
        assert not isinstance(value, list), (
            f"field {f.name} is a list; Topology must stay hashable"
        )


# ---------------------------------------------------------------------------
# 4. The whole point: a real MAM5 configuration must validate.
#
# The validator is only worth having if it accepts the configuration this axis
# exists to support. Numbers are the actual CAM MAM5 values from the sibling
# Fortran repo, not invented ones:
#     mam-box-fortran/box-src/mam_physprop_table.F90:101-104  (sigmag/dgnum/lo/hi)
#     mam-box-fortran/docs/MAM5_PLAN.md                       (nspec_amode, mode 5 so4-only)
# ---------------------------------------------------------------------------

def _cam_mam5_probe() -> Topology:
    """A CAM MAM5 topology with real size parameters and plausible indices.

    The index tables are illustrative -- the real ones come from CAM's chemistry
    preprocessor in PRs C-E -- but they are structurally faithful: mode 5 carries
    exactly one species (sulfate), and every pointer is unique and within pcnst.
    """
    nspec = (6, 4, 3, 2, 1)
    # Contiguous, non-overlapping blocks: number then that mode's masses.
    numptr, lmass, cursor = [], [], 0
    for n in nspec:
        numptr.append(cursor)
        lmass.append(tuple(range(cursor + 1, cursor + 1 + n)) + (-1,) * (14 - n))
        cursor += 1 + n
    return Topology(
        name="cam_mam5_probe",
        variant="cesm",
        nmodes=5,
        pcnst=cursor,
        mode_names=("accum", "aitken", "coarse", "primary_carbon", "coarse_strat"),
        specname_amode=data.SPECNAME_AMODE,
        nspec_amode=nspec,
        numptr_amode=tuple(numptr),
        numptrcw_amode=tuple(numptr),
        lspectype_amode=(
            (0, 3, 4, 5, 7, 6) + (-1,) * 8,
            (0, 4, 6, 7)       + (-1,) * 10,
            (7, 6, 0)          + (-1,) * 11,
            (3, 5)             + (-1,) * 12,
            (0,)               + (-1,) * 13,   # coarse_strat: sulfate only
        ),
        lmassptr_amode=tuple(lmass),
        lmassptrcw_amode=tuple(lmass),
        specdens_amode=data.SPECDENS_AMODE,
        spechygro_amode=data.SPECHYGRO_AMODE,
        sigmag_amode=(1.6, 1.6, 1.8, 1.600000023841858, 1.2),
        dgnum_amode=(1.1e-07, 2.6e-08, 2e-06, 5.000000058430487e-08, 9e-07),
        dgnumlo_amode=(5.35e-08, 8.7e-09, 1e-06, 9.99999993922529e-09, 4e-07),
        dgnumhi_amode=(4.8e-07, 5.2e-08, 4e-06, 1.0000000116860974e-07, 4e-05),
        rhcrystal_amode=(0.35,) * 5,
        rhdeliques_amode=(0.80,) * 5,
        provenance="probe only -- real tables land in PRs C-E",
    )


def test_a_real_mam5_topology_validates() -> None:
    t = _cam_mam5_probe()
    assert t.nmodes == 5
    assert t.variant == "cesm"
    assert t.is_mam5 is True
    assert t.mode_index("coarse_strat") == 4
    assert t.nspec_amode[4] == 1, "coarse_strat carries sulfate only"


def test_mam5_derived_quantities_are_finite_and_ordered() -> None:
    """voltonumb must fall monotonically with diameter, mode 5 included."""
    t = _cam_mam5_probe()
    for name in ("alnsg_amode", "dumfac_amode", "voltonumb_amode",
                 "voltonumblo_amode", "voltonumbhi_amode"):
        arr = np.asarray(getattr(t, name))
        assert arr.shape == (5,)
        assert np.isfinite(arr).all(), f"{name} not finite: {arr!r}"
    # Bigger particles -> fewer per unit volume.
    assert (t.voltonumbhi_amode < t.voltonumb_amode).all()
    assert (t.voltonumb_amode < t.voltonumblo_amode).all()


def test_mam5_probe_does_not_disturb_the_active_topology() -> None:
    """Constructing a topology must not register or activate it."""
    before = get_topology()
    _cam_mam5_probe()
    assert get_topology() is before
    assert "cam_mam5_probe" not in available_topologies()
