# Features

Catalog of what the Fortran reference offers and the corresponding status in the JAX port. The Fortran column is factual (extracted from the reference under `mam4-original-src-code/`); the JAX column tracks porting progress.

Status legend: **planned**, **in progress**, **ported (validated)**, **deferred** (see `DEFERRED.md`), **not planned**.

---

## Microphysical processes

| Process | Fortran module | JAX status |
| --- | --- | --- |
| Size redistribution (`calcsize`) | `box_model_utils/modal_aero_calcsize.F90` | **ported (validated)** end-to-end in `mam4_jax/processes/calcsize.py` (M3.5 PR-A + PR-B). Per-mode bounds-adjustment + Aitken↔accum transfer both implemented; dgncur_a matches Fortran at machine ε. Transfer code paths are dead in the canonical box-model fixture (`docs/DEFERRED.md`) but the port is structurally faithful. |
| Water uptake (`wateruptake`) | `e3sm_src_modified/modal_aero_wateruptake.F90` | **ported (validated)** end-to-end. `makoh_cubic`/`makoh_quartic`/`modal_aero_kohler` in `mam4_jax/kohler.py`; `_sub`/`_dr` orchestration in `mam4_jax/processes/wateruptake.py`. dgncur_awet/wetdens match Fortran at machine ε. |
| Gas–aerosol exchange (condensation) | `e3sm_src_modified/modal_aero_amicphys.F90` (`mam_gasaerexch_1subarea`, lines 3279–3584) | stub `mam4_jax/processes/gasaerexch.py` is **dead code** in the box-model build — the standalone module isn't called from the driver. Active port target is `_mam_gasaerexch_1subarea` inside `amicphys.py` (currently a no-op stub after M3.6 PR-A), scheduled for M3.6 PR-C. See `docs/ARCHITECTURE.md`. |
| New-particle nucleation (binary H₂SO₄–H₂O) | `e3sm_src_modified/modal_aero_amicphys.F90` (`mam_newnuc_1subarea`, lines 4251–4665) | stub `mam4_jax/processes/newnuc.py` is dead code. Active port target is `_mam_newnuc_1subarea` inside `amicphys.py` (currently a no-op stub after M3.6 PR-A), scheduled for M3.6 PR-D. |
| Coagulation (Brownian, intra/inter-modal) | `e3sm_src_modified/modal_aero_amicphys.F90` (`mam_coag_1subarea`, lines 4670–5106) | stub `mam4_jax/processes/coag.py` is dead code. Active port target is `_mam_coag_1subarea` inside `amicphys.py` (currently a no-op stub after M3.6 PR-A), scheduled for M3.6 PR-E. |
| Rename (Aitken → accumulation) | `e3sm_src_modified/modal_aero_amicphys.F90` (`mam_rename_1subarea`, lines 3923–4246) | stub `mam4_jax/processes/rename.py` is dead code. Active port target is `_mam_rename_1subarea` inside `amicphys.py` (currently a no-op stub after M3.6 PR-A), scheduled for M3.6 PR-B. |
| Umbrella orchestrator (`amicphys`) | `e3sm_src_modified/modal_aero_amicphys.F90` | **orchestration shell ported** in `mam4_jax/processes/amicphys.py` (M3.6 PR-A); four sub-process stubs land in PR-B (`rename`), PR-C (`gasaerexch`), PR-D (`newnuc`), PR-E (`coag`). All-mdo-off passthrough validated bit-exact. |

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

The exact species-to-mode assignment is set at initialization in `modal_aero_initialize_data.F90:250-309`. M3.3 captured these indices via the instrumented `dump_indices()` overlay and hard-coded them into `mam4_jax/data.py` (`NUMPTR_AMODE`, `LMASSPTR_AMODE`, etc.); the canonical values live at `tests/reference/indices/reference.npz` with `tests/test_scaffolding.py::test_index_tables_match_npz_reference` enforcing parity.

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
| Element-wise `1e-6` rel-err assertion (ADR-003) | **in use** — every M3 port PR has an end-to-end test asserting max rel-err < 1e-6 against a committed Fortran capture. As of M3.5, eight ports (polysvp, qsat_water/ice, IndexTables, makoh_cubic/quartic, modal_aero_kohler, wateruptake_dr, calcsize_sub) all match at machine ε. |
| 12-point convergence sweep matching `run_test.csh` | captured (`tests/reference/sweep/*.nc`); JAX reproduction planned for M5 |
| Per-process reference data for M3 port validation | captured (`tests/reference/per_process/*.npz` and siblings); schema in `tests/reference/SCHEMA.md` |
| Residual / convergence diagnostic plots | **in use** — seven plots committed under `docs/figures/` (polysvp, qsat, makoh, kohler, wateruptake, calcsize residuals + the upstream flowchart). New ports add their plot per the validation workflow in `CLAUDE.md`. |

## Out of scope (deferred or not planned)

See `DEFERRED.md` for: multi-column/multi-level execution, GPU/TPU sharding, end-to-end differentiability claims, CI, license selection.

Explicitly **not planned** at this time:
- Coupling to a host atmosphere model (E3SM, CESM). The port targets the *box model* configuration only.
- Sulfur chemistry beyond the placeholder `gaschem_simple` / `cloudchem_simple` stubs.
- Sea-salt emissions, aerosol deposition, convective processing — the Fortran modules `seasalt_model.F90`, `modal_aero_deposition.F90`, `modal_aero_convproc.F90`, and `aerodep_flx.F90` are stubs in the box model and will remain stubs in the JAX port.
