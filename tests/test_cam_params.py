"""Validate the generated CAM parameter layer (core/cam_params.py).

The values are READ OUT of an initialised CAM box model (sibling repo
`mam-box-fortran`, `tools/dump_tables/to_params.py`) because `specmw_amode`
and `adv_mass` come from the chemistry preprocessor and are not tabulated in
any source file. These tests re-check, from the committed data, the
invariants the generator asserts at generation time — a generator-time
assertion says nothing about what actually landed here — plus the
cross-checks against plan 024 §3's independently-sourced census.
"""
from __future__ import annotations

import mam4_jax  # noqa: F401
from mam4_jax.core.cam_params import CAM_PARAMS
from mam4_jax.core.cam_topologies import CAM_MAM4, CAM_MAM5


def test_params_align_with_topologies() -> None:
    """specmw is per TYPE aligned with Topology.specname_amode; adv_mass
    and cnst_names tile the gas window exactly; pcnst = loffset +
    gas_pcnst."""
    for topo in (CAM_MAM4, CAM_MAM5):
        p = CAM_PARAMS[topo.name]
        assert len(p["specmw_amode"]) == len(topo.specname_amode)
        assert len(p["adv_mass"]) == p["gas_pcnst"] == len(p["cnst_names"])
        assert topo.pcnst == p["loffset"] + p["gas_pcnst"]


def test_tracer_pointers_resolve_to_the_right_names() -> None:
    """The topology's lmassptr/numptr indices, shifted into the gas
    window, must land on tracers whose NAMES carry the right species
    prefix and mode suffix — the strongest committed-data consistency
    check available between the two generated files."""
    for topo in (CAM_MAM4, CAM_MAM5):
        p = CAM_PARAMS[topo.name]
        names, off = p["cnst_names"], p["loffset"]
        for m in range(topo.nmodes):
            suffix = f"_a{m + 1}"
            num_name = names[topo.numptr_amode[m] - off]
            assert num_name == "num" + suffix, (topo.name, m, num_name)
            for s in range(topo.nspec_amode[m]):
                lm = topo.lmassptr_amode[m][s]
                t = topo.lspectype_amode[m][s]
                spec_name = names[lm - off]
                expected = topo.specname_amode[t] + suffix
                assert spec_name == expected, (topo.name, m, s, spec_name)


def test_specmw_matches_the_plan024_census() -> None:
    """Cross-check against the independently-sourced values in plan 024
    §3 (from mam-box-fortran/docs/CESM_VS_E3SM.md): CAM so4 115.107340,
    organics/bc 12.011, dust 135.064039, seasalt 58.442468."""
    for topo in (CAM_MAM4, CAM_MAM5):
        p = CAM_PARAMS[topo.name]
        mw = dict(zip(topo.specname_amode, p["specmw_amode"]))
        assert mw["so4"] == 115.10734
        assert mw["pom"] == mw["soa"] == mw["bc"] == 12.011
        assert mw["dst"] == 135.064039
        assert mw["ncl"] == 58.442468


def test_gas_species_present_with_mechanism_mws() -> None:
    """The SO4-only driver needs H2SO4 and SO2 by name; their adv_mass
    values are the mechanism's (H2SO4 98.0784, SO2 64.0648)."""
    for topo in (CAM_MAM4, CAM_MAM5):
        p = CAM_PARAMS[topo.name]
        names = p["cnst_names"]
        assert p["adv_mass"][names.index("H2SO4")] == 98.0784
        assert p["adv_mass"][names.index("SO2")] == 64.0648
