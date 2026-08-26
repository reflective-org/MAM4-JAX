"""CAM per-configuration parameters — GENERATED, do not hand-edit.

Produced by `tools/dump_tables/to_params.py` in the sibling repo
`mam-box-fortran`; same provenance chain as `cam_topologies.py`
(read out of an initialised CAM MAM box model — `specmw_amode` and
`adv_mass` come from the chemistry preprocessor and are not
transcribable from source). Regenerate alongside the topologies:
    tools/dump_tables/run.sh mam4 > outputs/tables/cam_mam4_indices.py
    tools/dump_tables/run.sh mam5 > outputs/tables/cam_mam5_indices.py
    python3 tools/dump_tables/to_params.py
"""
from __future__ import annotations

CAM_PARAMS: dict = {
    # source: cam_mam4_indices.py  sha256: 1dd51e380a38ea35…
    "cam_mam4": {
        "loffset": 5,
        "gas_pcnst": 26,
        "mwdry": 28.966,
        # per species TYPE, aligned with Topology.specname_amode ('so4', 'pom', 'soa', 'bc', 'dst', 'ncl')
        "specmw_amode": (115.10734, 12.011, 12.011, 12.011, 135.064039, 58.442468),
        # chemistry-mechanism molecular weights, gas window (tracer slots loffset..pcnst-1)
        "adv_mass": (
            12.011,   # bc_a1
            12.011,   # bc_a4
            62.1324,   # DMS
            135.064039,   # dst_a1
            135.064039,   # dst_a2
            135.064039,   # dst_a3
            34.0136,   # H2O2
            98.0784,   # H2SO4
            58.442468,   # ncl_a1
            58.442468,   # ncl_a2
            58.442468,   # ncl_a3
            1.0074,   # num_a1
            1.0074,   # num_a2
            1.0074,   # num_a3
            1.0074,   # num_a4
            12.011,   # pom_a1
            12.011,   # pom_a4
            64.0648,   # SO2
            115.10734,   # so4_a1
            115.10734,   # so4_a2
            115.10734,   # so4_a3
            12.011,   # soa_a1
            12.011,   # soa_a2
            12.011,   # SOAE
            12.011,   # SOAG
            18.0142,   # H2O
        ),
        "cnst_names": ('bc_a1', 'bc_a4', 'DMS', 'dst_a1', 'dst_a2', 'dst_a3', 'H2O2', 'H2SO4', 'ncl_a1', 'ncl_a2', 'ncl_a3', 'num_a1', 'num_a2', 'num_a3', 'num_a4', 'pom_a1', 'pom_a4', 'SO2', 'so4_a1', 'so4_a2', 'so4_a3', 'soa_a1', 'soa_a2', 'SOAE', 'SOAG', 'H2O'),
    },
    # source: cam_mam5_indices.py  sha256: 9a4b4a28e7206bc7…
    "cam_mam5": {
        "loffset": 5,
        "gas_pcnst": 103,
        "mwdry": 28.966,
        # per species TYPE, aligned with Topology.specname_amode ('so4', 'pom', 'soa', 'bc', 'dst', 'ncl')
        "specmw_amode": (115.10734, 12.011, 12.011, 12.011, 135.064039, 58.442468),
        # chemistry-mechanism molecular weights, gas window (tracer slots loffset..pcnst-1)
        "adv_mass": (
            12.011,   # bc_a1
            12.011,   # bc_a4
            79.904,   # BR
            115.3567,   # BRCL
            95.9034,   # BRO
            141.90894,   # BRONO2
            99.71685,   # BRY
            153.8218,   # CCL4
            165.364506,   # CF2CLBR
            148.91021,   # CF3BR
            137.367503,   # CFC11
            187.37531,   # CFC113
            170.921013,   # CFC114
            154.466716,   # CFC115
            120.913206,   # CFC12
            173.8338,   # CH2BR2
            30.0252,   # CH2O
            94.9372,   # CH3BR
            133.4023,   # CH3CCL3
            50.4859,   # CH3CL
            47.032,   # CH3O2
            48.0394,   # CH3OOH
            16.0406,   # CH4
            252.7304,   # CHBR3
            35.4527,   # CL
            70.9054,   # CL2
            102.9042,   # CL2O2
            51.4521,   # CLO
            97.45764,   # CLONO2
            100.91685,   # CLY
            28.0104,   # CO
            44.0098,   # CO2
            66.007206,   # COF2
            82.461503,   # COFCL
            62.1324,   # DMS
            135.064039,   # dst_a1
            135.064039,   # dst_a2
            135.064039,   # dst_a3
            18.998403,   # F
            1.0074,   # H
            2.0148,   # H2
            259.823613,   # H2402
            34.0136,   # H2O2
            98.0784,   # H2SO4
            80.9114,   # HBR
            116.948003,   # HCFC141B
            100.493706,   # HCFC142B
            86.467906,   # HCFC22
            36.4601,   # HCL
            20.005803,   # HF
            63.01234,   # HNO3
            79.01174,   # HO2NO2
            96.9108,   # HOBR
            52.4595,   # HOCL
            14.00674,   # N
            44.01288,   # N2O
            108.01048,   # N2O5
            58.442468,   # ncl_a1
            58.442468,   # ncl_a2
            58.442468,   # ncl_a3
            30.00614,   # NO
            46.00554,   # NO2
            62.00494,   # NO3
            1.0074,   # num_a1
            1.0074,   # num_a2
            1.0074,   # num_a3
            1.0074,   # num_a4
            1.0074,   # num_a5
            15.9994,   # O
            31.9988,   # O2
            47.9982,   # O3
            47.9982,   # O3S
            67.4515,   # OCLO
            60.0764,   # OCS
            12.011,   # pom_a1
            12.011,   # pom_a4
            32.066,   # S
            146.056419,   # SF6
            48.0654,   # SO
            64.0648,   # SO2
            80.0642,   # SO3
            115.10734,   # so4_a1
            115.10734,   # so4_a2
            115.10734,   # so4_a3
            115.10734,   # so4_a5
            12.011,   # soa_a1
            12.011,   # soa_a2
            12.011,   # SOAG
            0.000548567,   # e
            33.0062,   # HO2
            14.00674,   # N2D
            28.01348,   # N2p
            30.00614,   # NOp
            14.00674,   # Np
            15.9994,   # O1D
            31.9988,   # O2_1D
            31.9988,   # O2_1S
            31.9988,   # O2p
            17.0068,   # OH
            15.9994,   # Op
            15.9994,   # Op2D
            15.9994,   # Op2P
            18.0142,   # H2O
        ),
        "cnst_names": ('bc_a1', 'bc_a4', 'BR', 'BRCL', 'BRO', 'BRONO2', 'BRY', 'CCL4', 'CF2CLBR', 'CF3BR', 'CFC11', 'CFC113', 'CFC114', 'CFC115', 'CFC12', 'CH2BR2', 'CH2O', 'CH3BR', 'CH3CCL3', 'CH3CL', 'CH3O2', 'CH3OOH', 'CH4', 'CHBR3', 'CL', 'CL2', 'CL2O2', 'CLO', 'CLONO2', 'CLY', 'CO', 'CO2', 'COF2', 'COFCL', 'DMS', 'dst_a1', 'dst_a2', 'dst_a3', 'F', 'H', 'H2', 'H2402', 'H2O2', 'H2SO4', 'HBR', 'HCFC141B', 'HCFC142B', 'HCFC22', 'HCL', 'HF', 'HNO3', 'HO2NO2', 'HOBR', 'HOCL', 'N', 'N2O', 'N2O5', 'ncl_a1', 'ncl_a2', 'ncl_a3', 'NO', 'NO2', 'NO3', 'num_a1', 'num_a2', 'num_a3', 'num_a4', 'num_a5', 'O', 'O2', 'O3', 'O3S', 'OCLO', 'OCS', 'pom_a1', 'pom_a4', 'S', 'SF6', 'SO', 'SO2', 'SO3', 'so4_a1', 'so4_a2', 'so4_a3', 'so4_a5', 'soa_a1', 'soa_a2', 'SOAG', 'e', 'HO2', 'N2D', 'N2p', 'NOp', 'Np', 'O1D', 'O2_1D', 'O2_1S', 'O2p', 'OH', 'Op', 'Op2D', 'Op2P', 'H2O'),
    },
}
