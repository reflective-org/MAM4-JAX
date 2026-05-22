# Features

Catalog of what the Fortran reference offers and the corresponding status in the JAX port. The Fortran column is factual (extracted from the reference under `mam4-original-src-code/`); the JAX column tracks porting progress.

Status legend: **planned**, **in progress**, **ported (validated)**, **deferred** (see `DEFERRED.md`), **not planned**.

---

## Microphysical processes

| Process | Fortran module | JAX status |
| --- | --- | --- |
| Size redistribution (`calcsize`) | `box_model_utils/modal_aero_calcsize.F90` | **ported (validated)** end-to-end in `mam4_jax/processes/calcsize.py` (M3.5 PR-A + PR-B). Per-mode bounds-adjustment + Aitken↔accum transfer both implemented; dgncur_a matches Fortran at machine ε. Transfer code paths are dead in the canonical box-model fixture (`docs/DEFERRED.md`) but the port is structurally faithful. |
| Water uptake (`wateruptake`) | `e3sm_src_modified/modal_aero_wateruptake.F90` | **ported (validated)** end-to-end. `makoh_cubic`/`makoh_quartic`/`modal_aero_kohler` in `mam4_jax/kohler.py`; `_sub`/`_dr` orchestration in `mam4_jax/processes/wateruptake.py`. dgncur_awet/wetdens match Fortran at machine ε. |
| Gas–aerosol exchange (condensation) | `e3sm_src_modified/modal_aero_amicphys.F90` (`mam_gasaerexch_1subarea`, lines 3279–3584, with `mam_soaexch_1subarea` 3589–3918) | **ported (validated) and wired into the orchestration** as `_mam_gasaerexch_1subarea` + `_mam_soaexch_1subarea` in `mam4_jax/processes/amicphys.py` (M3.6 PR-D for H₂SO₄ + helpers, PR-E for SOA exchange). SOA path uses the single-substep assumption (`dtcur = dtfull`); adaptive sub-stepping deferred to PR-E2 if a fixture ever triggers it. Max rel-err **4.77e-15** (machine ε) on all SOA/H₂SO₄/so4 tracers across 60 timesteps. RK4 branch still out of scope. Stub `mam4_jax/processes/gasaerexch.py` remains dead code. |
| New-particle nucleation (binary H₂SO₄–H₂O) | `e3sm_src_modified/modal_aero_amicphys.F90` (`mam_newnuc_1subarea`, lines 4251–4665) | **ported (validated) and wired into the orchestration** as `_mam_newnuc_1subarea` inside `mam4_jax/processes/amicphys.py` (M3.6 PR-F3). Built on PR-F1's leaf parameterizations + PR-F2's dispatcher; the orchestration glue pulls `qh2so4_avg` from gasaerexch, calls the dispatcher, applies particle-size constraints, deposits new-particle mass+number into Aitken mode. Max rel-err **3.9e-16** (machine ε) on the 3 newnuc-affected tracers across 60 steps. Stub `mam4_jax/processes/newnuc.py` remains dead code in the box-model build. |
| Coagulation (Brownian, intra/inter-modal) | `e3sm_src_modified/modal_aero_amicphys.F90` (`mam_coag_1subarea`, lines 4670–5106) + `e3sm_src/modal_aero_coag.F90` (`getcoags`, `getcoags_wrapper_f`) | **ported (validated) and wired into the orchestration** as `_mam_coag_1subarea` inside `mam4_jax/processes/amicphys.py` (M3.6 PR-G3); composes PR-G1's `getcoags` + PR-G2's `getcoags_wrapper_f` from `mam4_jax/coag.py`. 3 active MAM4-MOM coag pairs (Aitken→accum, pcarbon→accum, Aitken→pcarbon); marine-organics modes absent. Max rel-err **4.1e-13** across 33 aerosol-slot tracers × 60 timesteps. Stub `mam4_jax/processes/coag.py` remains dead code in the box-model build. |
| Rename (Aitken → accumulation) | `e3sm_src_modified/modal_aero_amicphys.F90` (`mam_rename_1subarea`, lines 3923–4246) | **ported (validated) and wired into the orchestration** as `_mam_rename_1subarea` inside `mam4_jax/processes/amicphys.py` (M3.6 PR-B for the physics, PR-C for the orchestration wiring). PR-B max rel-err vs full-physics fixture: 2.5e-9 (qnum) / 7.0e-10 (qaer) across 60 timesteps. PR-C wired it through `_mam_amicphys_1subarea_clear` via the state-dict ↔ amicphys-local-view unpacking layer (`MMR_TO_VMR` + `FCVT_*`); validated against the rename-only single-toggle Fortran capture at machine epsilon. Stub `mam4_jax/processes/rename.py` remains dead code in the box-model build. |
| Umbrella orchestrator (`amicphys`) | `e3sm_src_modified/modal_aero_amicphys.F90` | **orchestration shell ported** in `mam4_jax/processes/amicphys.py` (M3.6 PR-A); four sub-process stubs land in PR-B (`rename`), PR-C (`gasaerexch`), PR-D (`newnuc`), PR-E (`coag`). All-mdo-off passthrough validated bit-exact. **M3.6 complete (2026-05-22).** |
| Operator-splitting time loop (driver) | `test_drivers/driver.F90:1080-1367` (`main_time_loop`) | **ported and validated end-to-end (60 steps + 6-of-12 convergence sweep)** in `mam4_jax/driver.py` (M4 PR-A scaffold + PR-B trajectory + M5 convergence). `run_step(state)` chains `calcsize → wateruptake → cloud_chem (no-op) → amicphys`; `run_timesteps(state, n_steps)` is a plain Python `for` loop returning a stacked-snapshot trajectory (`jax.lax.scan` deferred to M6). **Max rel-err 1.97e-8** on `q` across 60 steps × 35 tracers — 50× under ADR-003. M5 convergence sweep validates `nstep ∈ {60, 120, 180, 360, 900, 1800}` against Fortran NetCDF outputs; `nstep ≤ 30` deferred to PR-E2 (adaptive SOA substepping). Cloud-chem stub fires the no-op branch (box-model fixture has `cldn=0`); pcarbon-aging remains out of scope; gas-chem term stays inside gasaerexch's analytical solver. Figures: `docs/figures/driver_60step_trajectory.png`, `docs/figures/sweep_convergence.png`. |

## Supporting physics

| Capability | Fortran module | JAX status |
| --- | --- | --- |
| Saturation vapor pressure (`polysvp`) | `box_model_utils/wv_saturation.F90:699-736` (Goff–Gratch) | **ported (validated)** in `mam4_jax/saturation.py`; max rel-err 4e-15. |
| Saturation specific humidity (`qsat_water`, `qsat_ice`) | `wv_saturation.F90:758-862` (Goff–Gratch / Clausius–Clapeyron mix) | **ported (validated)** in `mam4_jax/saturation.py`; max rel-err 9e-14 / 8e-15. |
| Binary H₂SO₄–H₂O nucleation parameterization | `modal_aero_newnuc.F90:1256-1448` (Vehkamäki 2002) | **ported (validated)** in `mam4_jax/newnuc.py` (M3.6 PR-F1); max rel-err **6.4e-11** on `rateloge` across 1920 (T, RH, [H₂SO₄]) records. |
| Boundary-layer nucleation overlay | `modal_aero_newnuc.F90:1179-1255` (Wang 2008 first/second order) | **ported (validated)** in `mam4_jax/newnuc.py` (M3.6 PR-F1); both flagaa=11 (first-order) and flagaa=12 (second-order) paths match at machine ε. |
| Newnuc dispatcher (binary path + KK2002 size correction + grown-particle composition) | `modal_aero_newnuc.F90:598-1173` (`mer07_veh02_nuc_mosaic_1box`) | **ported (validated)** as `mer07_veh02_nuc_mosaic_1box` in `mam4_jax/newnuc.py` (M3.6 PR-F2). MAM4-MOM-specific simplifications: no ternary, `nsize=1` hardcoded, no NH₃-aware composition. Max rel-err **2.27e-12** on `qnuma_del`, `qso4a_del`, `qh2so4_del`, `dnclusterdt` across 2160 records covering 5 regimes (subcutoff / low-rate / active no-PBL / active PBL / gas-limited). |
| Whitby coagulation coefficients (closed-form intra/intermodal, 0th/2nd/3rd moments) | `modal_aero_coag.F90:1177-2858` (`getcoags`) + lookup tables in `modal_aero_coag.F90:1306-2540` | **ported (validated)** as `getcoags` in `mam4_jax/coag.py` (M3.6 PR-G1). Lookup tables (`bm0`, `bm0ij`, `bm3i`, `bm2ii`, `bm2iitt`, `bm2ij`, `bm2ji`) extracted once by `scripts/extract_coag_tables.py` into `mam4_jax/_coag_tables.npz`. Max rel-err **6.5e-9** on all 8 coefficients across 240 (T, P, dgnumA, dgnumB) records. |
| Coagulation-coefficient wrapper (CMAQ→MIRAGE2 conversion: prep `lamda` / `knc` / `kfm*` + clamp + divide) | `modal_aero_coag.F90:999-1129` (`getcoags_wrapper_f`) | **ported (validated)** as `getcoags_wrapper_f` in `mam4_jax/coag.py` (M3.6 PR-G2). Composes `getcoags`; reuses the PR-G1 fixture for validation. 7/8 outputs at machine ε; `betaij2j` inherits 6.5e-9 from PR-G1's `qs21`. |
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
| Element-wise `1e-6` rel-err assertion (ADR-003) | **in use** — every M3 port PR has an end-to-end test asserting max rel-err < 1e-6 against a committed Fortran capture. As of M3.6 PR-F3, **fifteen** ports plus the orchestration unpack/repack round-trip all pass. Most ports match at machine ε; rename matches at ~1e-9; orchestration's gasaerexch+soaexch test matches at < 1e-14; orchestration's gasaerexch+newnuc test matches at < 4e-16; binary_nuc_vehk2002's polynomial-accumulated rateloge matches at < 6.5e-11; the mer07_veh02 dispatcher matches at < 2.3e-12 across 5 regimes. |
| 12-point convergence sweep matching `run_test.csh` | captured (`tests/reference/sweep/*.nc`); JAX reproduction planned for M5 |
| Per-process reference data for M3 port validation | captured (`tests/reference/per_process/*.npz` and siblings); schema in `tests/reference/SCHEMA.md` |
| Residual / convergence diagnostic plots | **in use** — thirteen plots committed under `docs/figures/` (polysvp, qsat, makoh, kohler, wateruptake, calcsize, rename, gasaerexch, soaexch, newnuc_helpers, mer07_veh02, newnuc_orchestration residuals + the upstream flowchart). New ports add their plot per the validation workflow in `CLAUDE.md`. |

## Out of scope (deferred or not planned)

See `DEFERRED.md` for: multi-column/multi-level execution, GPU/TPU sharding, end-to-end differentiability claims, CI, license selection.

Explicitly **not planned** at this time:
- Coupling to a host atmosphere model (E3SM, CESM). The port targets the *box model* configuration only.
- Sulfur chemistry beyond the placeholder `gaschem_simple` / `cloudchem_simple` stubs.
- Sea-salt emissions, aerosol deposition, convective processing — the Fortran modules `seasalt_model.F90`, `modal_aero_deposition.F90`, `modal_aero_convproc.F90`, and `aerodep_flx.F90` are stubs in the box model and will remain stubs in the JAX port.
