"""CAM mode topologies — GENERATED, do not hand-edit.

Produced by `tools/dump_tables/to_topology.py` in the sibling repo
`mam-box-fortran`, from index tables read out of an initialised CAM MAM box
model. Those tables are not transcribable from source: `modal_aero_data.F90`
builds them at init from the chemistry preprocessor's species list.

Regenerate with:
    cd mam-box-fortran
    ./build/build_cam.sh && ./build/build_cam.sh --mam5
    tools/dump_tables/run.sh mam4 > outputs/tables/cam_mam4_indices.py
    tools/dump_tables/run.sh mam5 > outputs/tables/cam_mam5_indices.py
    python3 tools/dump_tables/to_topology.py

`lspectype_amode` here is SYNTHESISED: CAM has no such array, resolving each
(mode, slot) straight to its density and hygroscopicity. The generator derives
the type list from species-name prefixes and verifies it reproduces CAM's
per-slot arrays exactly; `tests/test_cam_topology.py` re-checks that from the
committed data, so the claim is not merely a generator-time assertion.
"""
from __future__ import annotations

from mam4_jax.core.topology import Topology, register_topology

# --- CAM MAM4 --------------------------------------------------------
# source: cam_mam4_indices.py  sha256: d51d3379199331e4…
CAM_MAM4 = register_topology(Topology(
    name="cam_mam4",
    variant="cesm",
    nmodes=4,
    pcnst=31,
    mode_names=('accum', 'aitken', 'coarse', 'primary_carbon'),
    specname_amode=('so4', 'pom', 'soa', 'bc', 'dst', 'ncl'),
    nspec_amode=(6, 4, 3, 2),
    numptr_amode=(16, 17, 18, 19),
    numptrcw_amode=(16, 17, 18, 19),
    lspectype_amode=(
        (0, 1, 2, 3, 4, 5),
        (0, 2, 5, 4, -1, -1),
        (4, 5, 0, -1, -1, -1),
        (1, 3, -1, -1, -1, -1),
    ),
    lmassptr_amode=(
        (23, 20, 26, 5, 8, 13),
        (24, 27, 14, 9, -1, -1),
        (10, 15, 25, -1, -1, -1),
        (21, 6, -1, -1, -1, -1),
    ),
    lmassptrcw_amode=(
        (23, 20, 26, 5, 8, 13),
        (24, 27, 14, 9, -1, -1),
        (10, 15, 25, -1, -1, -1),
        (21, 6, -1, -1, -1, -1),
    ),
    specdens_amode=(1770.0, 1000.0, 1000.0, 1700.0, 2600.0, 1900.0),
    spechygro_amode=(0.507, 1.000000082740371e-10, 0.14, 1.000000013351432e-10, 0.068, 1.16),
    sigmag_amode=(1.8, 1.6, 1.8, 1.600000023841858),
    dgnum_amode=(1.1e-07, 2.6e-08, 2e-06, 5.000000058430487e-08),
    dgnumlo_amode=(5.35e-08, 8.7e-09, 1e-06, 9.99999993922529e-09),
    dgnumhi_amode=(4.4e-07, 5.2e-08, 4e-06, 1.0000000116860974e-07),
    rhcrystal_amode=(0.35, 0.35, 0.35, 0.35),
    rhdeliques_amode=(0.8, 0.8, 0.8, 0.8),
    provenance="CAM cam6_4_187, MAM4. Index tables and per-slot properties read out of an initialised box model "
))

# --- CAM MAM5 --------------------------------------------------------
# source: cam_mam5_indices.py  sha256: 989ae4e263afe9f0…
CAM_MAM5 = register_topology(Topology(
    name="cam_mam5",
    variant="cesm",
    nmodes=5,
    pcnst=108,
    mode_names=('accum', 'aitken', 'coarse', 'primary_carbon', 'coarse_strat'),
    specname_amode=('so4', 'pom', 'soa', 'bc', 'dst', 'ncl'),
    nspec_amode=(6, 4, 3, 2, 1),
    numptr_amode=(68, 69, 70, 71, 72),
    numptrcw_amode=(68, 69, 70, 71, 72),
    lspectype_amode=(
        (0, 1, 2, 3, 4, 5),
        (0, 2, 5, 4, -1, -1),
        (4, 5, 0, -1, -1, -1),
        (1, 3, -1, -1, -1, -1),
        (0, -1, -1, -1, -1, -1),
    ),
    lmassptr_amode=(
        (86, 79, 90, 5, 40, 62),
        (87, 91, 63, 41, -1, -1),
        (42, 64, 88, -1, -1, -1),
        (80, 6, -1, -1, -1, -1),
        (89, -1, -1, -1, -1, -1),
    ),
    lmassptrcw_amode=(
        (86, 79, 90, 5, 40, 62),
        (87, 91, 63, 41, -1, -1),
        (42, 64, 88, -1, -1, -1),
        (80, 6, -1, -1, -1, -1),
        (89, -1, -1, -1, -1, -1),
    ),
    specdens_amode=(1770.0, 1000.0, 1000.0, 1700.0, 2600.0, 1900.0),
    spechygro_amode=(0.507, 1.000000082740371e-10, 0.14, 1.000000013351432e-10, 0.068, 1.16),
    sigmag_amode=(1.6, 1.6, 1.8, 1.600000023841858, 1.2),
    dgnum_amode=(1.1e-07, 2.6e-08, 2e-06, 5.000000058430487e-08, 9e-07),
    dgnumlo_amode=(5.35e-08, 8.7e-09, 1e-06, 9.99999993922529e-09, 4e-07),
    dgnumhi_amode=(4.8e-07, 5.2e-08, 4e-06, 1.0000000116860974e-07, 4e-05),
    rhcrystal_amode=(0.35, 0.35, 0.35, 0.35, 0.35),
    rhdeliques_amode=(0.8, 0.8, 0.8, 0.8, 0.8),
    provenance="CAM cam6_4_187, MAM5. Index tables and per-slot properties read out of an initialised box model "
))
