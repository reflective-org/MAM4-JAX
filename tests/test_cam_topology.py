"""The CAM MAM4 and MAM5 topology instances.

These are the payload of the MAM5 work: `Topology` already accepted `nmodes=5`,
but nothing was registered, so "MAM5-JAX" was a validated shape with no data.

`mam4_jax/core/cam_topologies.py` is GENERATED in the sibling repo
`mam-box-fortran` by reading the index tables out of an initialised CAM box
model. That indirection is not laziness: `numptr_amode`, `lmassptr_amode` and
the per-slot species properties are not tabulated in any source file --
`modal_aero_data.F90` builds them at init from the chemistry preprocessor's
species list -- so they cannot be transcribed, only observed.

The interesting assertion here is the LOSSLESSNESS of `lspectype_amode`. CAM has
no such array; it resolves each (mode, slot) straight to `specdens_amode(l,m)`
and `spechygro(l,m)`. Rather than give `Topology` two shapes, the generator
synthesises a per-type list from species-name prefixes. That is only legitimate
if it round-trips, and the generator asserting it at generation time is not
evidence for the committed data -- so it is re-checked here.
"""
from __future__ import annotations

import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.core import data
from mam4_jax.core.cam_topologies import CAM_MAM4, CAM_MAM5
from mam4_jax.core.topology import available_topologies, get_topology

CAM = {"cam_mam4": CAM_MAM4, "cam_mam5": CAM_MAM5}


# --- registration -----------------------------------------------------------

def test_both_cam_topologies_are_registered() -> None:
    for name in CAM:
        assert name in available_topologies()


def test_importing_them_does_not_change_the_active_topology() -> None:
    """Registration must not activate. E3SM stays the default so that importing
    the CAM tables cannot silently repoint existing code."""
    assert get_topology() is data.E3SM_MAM4_MOM


# --- what MAM5 actually is --------------------------------------------------

def test_mam5_has_the_coarse_strat_mode() -> None:
    t = CAM_MAM5
    assert t.nmodes == 5
    assert t.variant == "cesm"
    assert t.is_mam5 is True
    assert t.mode_names[4] == "coarse_strat"
    assert t.mode_index("coarse_strat") == 4


def test_coarse_strat_carries_sulfate_only() -> None:
    """Mode 5 is the stratospheric sulfate mode: one species, and it is so4."""
    t = CAM_MAM5
    assert t.nspec_amode[4] == 1
    only = t.lspectype_amode[4][0]
    assert t.specname_amode[only] == "so4"
    assert t.specdens_amode[only] == pytest.approx(1770.0)
    assert t.spechygro_amode[only] == pytest.approx(0.507)


def test_coarse_strat_size_distribution() -> None:
    """Narrower and smaller than the tropospheric coarse mode -- which is why it
    takes up sulfate faster in the Fortran (3.4 % vs 0.22 % over an hour)."""
    t = CAM_MAM5
    assert t.sigmag_amode[4] == pytest.approx(1.2)
    assert t.dgnum_amode[4] == pytest.approx(9e-7)
    assert t.dgnumlo_amode[4] == pytest.approx(4e-7)
    assert t.dgnumhi_amode[4] == pytest.approx(4e-5)
    # Asymmetric about dgnum: 2.25x headroom below, 44.4x above. This is why
    # calcsize only ever responds on the low side for mode 5.
    assert t.dgnum_amode[4] / t.dgnumlo_amode[4] == pytest.approx(2.25)
    assert t.dgnumhi_amode[4] / t.dgnum_amode[4] == pytest.approx(44.44, rel=1e-3)


def test_mam4_and_mam5_agree_on_the_first_four_modes_where_they_should() -> None:
    """MAM5 is MAM4 plus a mode -- except accum's width, which the strat size
    variant changes from 1.8 to 1.6. Asserted rather than assumed, because
    'MAM5 = MAM4 + 1' is the intuition and it is wrong here."""
    assert CAM_MAM4.mode_names == CAM_MAM5.mode_names[:4]
    assert CAM_MAM4.nspec_amode == CAM_MAM5.nspec_amode[:4]
    assert CAM_MAM4.sigmag_amode[0] == pytest.approx(1.8)
    assert CAM_MAM5.sigmag_amode[0] == pytest.approx(1.6)


# --- the synthesis must be lossless ----------------------------------------

# CAM's own per-(mode, slot) arrays, as read out of the running model. Kept
# here rather than imported so the check compares two independent copies.
CAM_MAM5_SPECDENS_BY_SLOT = (
    (1770.0, 1000.0, 1000.0, 1700.0, 2600.0, 1900.0),
    (1770.0, 1000.0, 1900.0, 2600.0),
    (2600.0, 1900.0, 1770.0),
    (1000.0, 1700.0),
    (1770.0,),
)
CAM_MAM5_SPEC_NAMES_BY_SLOT = (
    ("so4_a1", "pom_a1", "soa_a1", "bc_a1", "dst_a1", "ncl_a1"),
    ("so4_a2", "soa_a2", "ncl_a2", "dst_a2"),
    ("dst_a3", "ncl_a3", "so4_a3"),
    ("pom_a4", "bc_a4"),
    ("so4_a5",),
)


def test_per_type_table_reproduces_cams_per_slot_densities() -> None:
    """The round-trip that justifies synthesising lspectype_amode at all."""
    t = CAM_MAM5
    for m, row in enumerate(CAM_MAM5_SPECDENS_BY_SLOT):
        for s, expected in enumerate(row):
            got = t.specdens_amode[t.lspectype_amode[m][s]]
            assert got == pytest.approx(expected), (
                f"mode {m} slot {s}: per-type gives {got}, CAM has {expected}"
            )


def test_synthesised_types_match_the_species_names() -> None:
    """lspectype_amode[m][s] must point at the type named by the species in
    that slot -- otherwise the indices are self-consistent but wrong."""
    t = CAM_MAM5
    for m, row in enumerate(CAM_MAM5_SPEC_NAMES_BY_SLOT):
        for s, name in enumerate(row):
            prefix = name.rsplit("_", 1)[0]
            assert t.specname_amode[t.lspectype_amode[m][s]] == prefix, (
                f"mode {m} slot {s} holds {name} but lspectype points at "
                f"{t.specname_amode[t.lspectype_amode[m][s]]}"
            )


def test_pom_and_bc_are_distinct_types_despite_both_reading_1e_minus_10() -> None:
    """Both hygroscopicities print as 1.0e-10 at low precision but are distinct
    float32 round-trips. If the generator had emitted them rounded, the two
    types would have collapsed and the synthesis would have been wrong."""
    t = CAM_MAM5
    i_pom = t.specname_amode.index("pom")
    i_bc = t.specname_amode.index("bc")
    assert t.spechygro_amode[i_pom] != t.spechygro_amode[i_bc]
    assert t.specdens_amode[i_pom] != t.specdens_amode[i_bc]
    for i in (i_pom, i_bc):
        assert t.spechygro_amode[i] == pytest.approx(1e-10, rel=1e-6)


# --- derived quantities -----------------------------------------------------

@pytest.mark.parametrize("name", sorted(CAM))
def test_derived_quantities_are_finite_and_ordered(name: str) -> None:
    t = CAM[name]
    for prop in ("alnsg_amode", "dumfac_amode", "voltonumb_amode",
                 "voltonumblo_amode", "voltonumbhi_amode"):
        arr = np.asarray(getattr(t, prop))
        assert arr.shape == (t.nmodes,)
        assert np.isfinite(arr).all(), f"{name}.{prop} not finite: {arr!r}"
    # Bigger particles -> fewer per unit volume.
    assert (t.voltonumbhi_amode < t.voltonumb_amode).all()
    assert (t.voltonumb_amode < t.voltonumblo_amode).all()


def test_mam5_dumfac_is_what_the_upstream_bug_reads() -> None:
    """The stale-dumfac bug (calcsize) reads the LAST mode's value. Under MAM5
    that is coarse_strat at sigma_g = 1.2, giving +32.5 % on both transfer modes
    -- versus +20.6 % on accum alone under MAM4, whose last mode happens to
    share aitken's width. This pins the numbers the bug report quotes."""
    import math
    stale5 = float(CAM_MAM5.dumfac_amode[-1])
    correct5 = float(CAM_MAM5.dumfac_amode[0])
    assert (correct5 / stale5) ** (1 / 3) == pytest.approx(1.3251, rel=1e-3)

    stale4 = float(CAM_MAM4.dumfac_amode[-1])
    correct4 = float(CAM_MAM4.dumfac_amode[0])
    assert (correct4 / stale4) ** (1 / 3) == pytest.approx(1.2055, rel=1e-3)
