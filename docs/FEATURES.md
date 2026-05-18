# Features

Catalog of what the Fortran reference offers and the corresponding status in the JAX port. The Fortran column is factual (extracted from the reference under `mam4-original-src-code/`); the JAX column tracks porting progress.

Status legend: **planned**, **in progress**, **ported (validated)**, **deferred** (see `DEFERRED.md`), **not planned**.

---

## Microphysical processes

| Process | Fortran module | JAX status |
| --- | --- | --- |
| Size redistribution (`calcsize`) | `box_model_utils/modal_aero_calcsize.F90` | planned (`PLANS.md` M3) |
| Water uptake (`wateruptake`) | `e3sm_src_modified/modal_aero_wateruptake.F90` | planned (`PLANS.md` M3) |
| Gas–aerosol exchange (condensation) | `e3sm_src/modal_aero_gasaerexch.F90` | planned (`PLANS.md` M3) |
| New-particle nucleation (binary H2SO4–H2O) | `e3sm_src/modal_aero_newnuc.F90` | planned (`PLANS.md` M3) |
| Coagulation (Brownian, intra/inter-modal) | `e3sm_src/modal_aero_coag.F90` | planned (`PLANS.md` M3) |
| Rename (Aitken → accumulation) | `e3sm_src/modal_aero_rename.F90` | planned (`PLANS.md` M3) |
| Umbrella orchestrator (`amicphys`) | `e3sm_src_modified/modal_aero_amicphys.F90` | planned (`PLANS.md` M4) |

## Supporting physics

| Capability | Fortran module | JAX status |
| --- | --- | --- |
| Saturation vapor pressure | `box_model_utils/wv_saturation.F90` (Goff-Gratch / Flatau) | planned (`PLANS.md` M3 candidate first port) |
| Constants and species table | `e3sm_src/modal_aero_data.F90`, `e3sm_src/shr_const_mod.F90` | planned (`PLANS.md` M1 — transcribe into `mam4_jax/data.py`) |
| Error function / special functions | `box_model_utils/error_function.F90`, `e3sm_src/shr_spfn_mod.F90` | use `jax.scipy.special` if available; otherwise port closed-form |

## Modes and species

| Mode | Species in MAM4-MOM (reference config) |
| --- | --- |
| Aitken (mode 1) | so4, soa, ncl, (mom) |
| Accumulation (mode 2) | so4, soa, ncl, (mom), pom, bc, dst |
| Coarse (mode 3) | so4, soa, ncl, dst, (mom), pom, bc |
| Primary carbon (mode 4) | pom, bc, (mom) |

Counts and the exact species per mode are authoritative in `modal_aero_data.F90` — the table above is a navigation aid, not a substitute. Reference build uses `-DMODAL_AERO_4MODE_MOM -DPCNST=35 -DNBC=1 -DNPOA=1 -DNSOA=1`.

## I/O

| Capability | Fortran | JAX status |
| --- | --- | --- |
| Input via namelist | `driver.F90` (`&time_input`, `&cntl_input`, `&met_input`, `&chem_input`) | TBD — see `ARCHITECTURE.md` open questions (dataclass / dict / YAML) |
| Output as NetCDF | `driver.F90` writes `mam_output.nc` | planned — match Fortran NetCDF layout for diffability |
| Per-process input/output capture (for validation) | not in Fortran natively — must instrument | planned (`PLANS.md` M2) |

## Validation features

| Capability | Status |
| --- | --- |
| Element-wise `1e-6` rel-err assertion (ADR-003) | planned (`PLANS.md` M1 test harness) |
| 12-point convergence sweep matching `run_test.csh` | planned (`PLANS.md` M5) |
| Residual / convergence diagnostic plots | planned (rule #6 — figures are first-class deliverables) |

## Out of scope (deferred or not planned)

See `DEFERRED.md` for: multi-column/multi-level execution, GPU/TPU sharding, end-to-end differentiability claims, CI, license selection.

Explicitly **not planned** at this time:
- Coupling to a host atmosphere model (E3SM, CESM). The port targets the *box model* configuration only.
- Sulfur chemistry beyond the placeholder `gaschem_simple` / `cloudchem_simple` stubs.
- Sea-salt emissions, aerosol deposition, convective processing — the Fortran modules `seasalt_model.F90`, `modal_aero_deposition.F90`, `modal_aero_convproc.F90`, and `aerodep_flx.F90` are stubs in the box model and will remain stubs in the JAX port.
