# Features

Catalog of what the Fortran reference offers and the corresponding status in the JAX port. The Fortran column is factual (extracted from the reference under `mam4-original-src-code/`); the JAX column tracks porting progress.

Status legend: **planned**, **in progress**, **ported (validated)**, **deferred** (see `DEFERRED.md`), **not planned**.

---

## Microphysical processes

| Process | Fortran module | JAX status |
| --- | --- | --- |
| Size redistribution (`calcsize`) | `box_model_utils/modal_aero_calcsize.F90` | stub in `mam4_jax/processes/calcsize.py`; physics port planned (M3) |
| Water uptake (`wateruptake`) | `e3sm_src_modified/modal_aero_wateruptake.F90` | stub in `mam4_jax/processes/wateruptake.py`; physics port planned (M3) |
| Gas–aerosol exchange (condensation) | `e3sm_src/modal_aero_gasaerexch.F90` | stub in `mam4_jax/processes/gasaerexch.py`; physics port planned (M3) |
| New-particle nucleation (binary H2SO4–H2O) | `e3sm_src/modal_aero_newnuc.F90` | stub in `mam4_jax/processes/newnuc.py`; physics port planned (M3) |
| Coagulation (Brownian, intra/inter-modal) | `e3sm_src/modal_aero_coag.F90` | stub in `mam4_jax/processes/coag.py`; physics port planned (M3) |
| Rename (Aitken → accumulation) | `e3sm_src/modal_aero_rename.F90` | stub in `mam4_jax/processes/rename.py`; physics port planned (M3) |
| Umbrella orchestrator (`amicphys`) | `e3sm_src_modified/modal_aero_amicphys.F90` | stub in `mam4_jax/processes/amicphys.py`; orchestration planned (M4) |

## Supporting physics

| Capability | Fortran module | JAX status |
| --- | --- | --- |
| Saturation vapor pressure (`polysvp`) | `box_model_utils/wv_saturation.F90:699-736` (Goff–Gratch) | **ported (validated)** in `mam4_jax/saturation.py`; max rel-err 4e-15. |
| Saturation specific humidity (`qsat_water`, `qsat_ice`) | `wv_saturation.F90:758-862` (Goff–Gratch / Clausius–Clapeyron mix) | **ported (validated)** in `mam4_jax/saturation.py`; max rel-err 9e-14 / 8e-15. |
| Physical constants (RGAS, MWDAIR, MWWV, LATVAP, LATICE, EPSQS, …) | `e3sm_src/shr_const_mod.F90:33-61`, `box_model_utils/physconst.F90` | **transcribed** in `mam4_jax/constants.py`. |
| Constants and species table | `e3sm_src/modal_aero_data.F90`, `e3sm_src/shr_const_mod.F90` | compile-time + runtime indices hard-coded in `mam4_jax/data.py` (0-based, with sentinel `-1` for unused slots); provenance at `tests/reference/indices/reference.npz` |
| Error function / special functions | `box_model_utils/error_function.F90`, `e3sm_src/shr_spfn_mod.F90` | use `jax.scipy.special` if available; otherwise port closed-form |

## Modes and species

The MAM4-MOM (with `RAIN_EVAP_TO_COARSE_AERO`) reference configuration has four modes in this Fortran order (from `modal_aero_data.F90:104-109, 121-123`):

| Mode index | Mode name | `nspec_amode` |
| --- | --- | --- |
| 1 | `accum` | 7 |
| 2 | `aitken` | 4 |
| 3 | `coarse` | 7 |
| 4 | `primary_carbon` | 3 |

The nine aerosol species *types* available across modes (`specname_amode`, `modal_aero_data.F90:49-52`):
`sulfate`, `ammonium`, `nitrate`, `p-organic`, `s-organic`, `black-c`, `seasalt`, `dust`, `m-organic`.

The exact species-to-mode assignment is set at initialization in `modal_aero_initialize_data.F90:250-309` and will be captured authoritatively when M2 lands the instrumented reference. Until then, `mam4_jax/data.py` keeps `IndexTables` sentinel-filled (-1).

Reference build flags: `-DMODAL_AERO_4MODE_MOM -DRAIN_EVAP_TO_COARSE_AERO -DPCNST=35 -DPCOLS=1 -DPVER=1 -DNBC=1 -DNPOA=1 -DNSOA=1` (see `mam4-original-src-code/test_drivers/cambox_config.cpp.in`).

## I/O

| Capability | Fortran | JAX status |
| --- | --- | --- |
| Input via namelist | `driver.F90` (`&time_input`, `&cntl_input`, `&met_input`, `&chem_input`) | scaffolded — `mam4_jax/config.py` exposes `TimeConfig` / `ControlConfig` / `MetConfig` / `ChemConfig` + `load_yaml` (ADR-010) |
| Output as NetCDF | `driver.F90` writes `mam_output.nc` | reference captures committed under `tests/reference/sweep/`; JAX-side NetCDF output still planned |
| Per-process input/output capture (for validation) | not in Fortran natively — instrumented via the ADR-012 overlay | `scripts/patches/` + `scripts/capture_reference.py --mode instrumented`; outputs under `tests/reference/per_process/*.npz` |

## Validation features

| Capability | Status |
| --- | --- |
| Element-wise `1e-6` rel-err assertion (ADR-003) | scaffolding test (`tests/test_scaffolding.py`) live; rel-err assertion harness planned for first M3 port |
| 12-point convergence sweep matching `run_test.csh` | captured (`tests/reference/sweep/*.nc`); JAX reproduction planned for M5 |
| Per-process reference data for M3 port validation | captured (`tests/reference/per_process/*.npz`); schema in `tests/reference/SCHEMA.md` |
| Residual / convergence diagnostic plots | planned (rule #6 — figures are first-class deliverables) |

## Out of scope (deferred or not planned)

See `DEFERRED.md` for: multi-column/multi-level execution, GPU/TPU sharding, end-to-end differentiability claims, CI, license selection.

Explicitly **not planned** at this time:
- Coupling to a host atmosphere model (E3SM, CESM). The port targets the *box model* configuration only.
- Sulfur chemistry beyond the placeholder `gaschem_simple` / `cloudchem_simple` stubs.
- Sea-salt emissions, aerosol deposition, convective processing — the Fortran modules `seasalt_model.F90`, `modal_aero_deposition.F90`, `modal_aero_convproc.F90`, and `aerodep_flx.F90` are stubs in the box model and will remain stubs in the JAX port.
