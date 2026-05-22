# Architecture

This document describes the architecture of the MAM4 aerosol-microphysics model as represented in the Fortran reference under `mam4-original-src-code/`, and the structure of the JAX port as currently implemented. Both sections are factual; outstanding architectural decisions are captured as ADRs in [`KEY_DECISIONS.md`](KEY_DECISIONS.md).

## Glossary

- **MAM4** — Modal Aerosol Module, 4-mode version. Represents the aerosol population as four log-normal *modes* (size distributions), each carrying a number concentration and per-species mass concentrations.
- **Mode** — one of Aitken (small), accumulation (medium), coarse (large), and primary-carbon. The reference configuration uses the *marine-organics* variant (`MOM`).
- **Species** — chemical components carried by a mode (e.g., sulfate `so4`, primary organic matter `pom`, secondary organic aerosol `soa`, black carbon `bc`, dust `dst`, sea-salt `ncl`, marine organic matter `mom`).
- **Interstitial vs. cloud-borne** — every interstitial tracer has a `qqcw` cloud-borne counterpart that mirrors it. The Fortran code passes them as separate arrays.
- **Tracer array `q`** — flat `(pcols, pver, pcnst)` array packing all mode-number and mode-species mass mixing ratios. Mapping is via `numptr_amode`, `lmassptr_amode` in `modal_aero_data`.
- **Time step `mam_dt`** — outer time step within which all microphysical processes are applied via sequential operator splitting.

## Fortran reference: operator-splitting time loop

The driver applies the following processes in order on each `mam_dt` step:

1. **`modal_aero_calcsize`** — recompute dry diameters from number + mass; enforce per-mode size bounds; transfer particles between modes when bounds are violated.
2. **`modal_aero_wateruptake`** — equilibrium water uptake (Köhler-like); produces wet diameter and wet density.
3. **`modal_aero_amicphys`** — umbrella subroutine that internally runs the following in order:
   - **`modal_aero_gasaerexch`** — H2SO4 / SOAG condensation onto modes.
   - **`modal_aero_newnuc`** — binary H2SO4–H2O nucleation (Vehkamäki-style).
   - **`modal_aero_coag`** — intra- and inter-modal Brownian coagulation.
   - **`modal_aero_rename`** — transfer aged Aitken-mode particles to accumulation mode when size criteria are met.
4. **`gaschem_simple`, `cloudchem_simple`** — placeholder simple chemistry hooks in the box model.

Each sub-process is a Fortran subroutine that mutates `q(:,:,pcnst)` in place. The index bookkeeping in `modal_aero_data` (`lmassptr_amode`, `numptr_amode`, `lmassptrcw_amode`, `numptrcw_amode`, etc.) is the **single largest porting hazard** — it must surface explicitly in the JAX data model rather than hiding behind integer indirections.

## Fortran reference: heaviest modules

Useful for scoping porting effort:

| Module | Lines | Notes |
| --- | --- | --- |
| `box_model_utils/physics_buffer.F90` | ~6500 | E3SM physics-buffer infrastructure; mostly stub-able in JAX. |
| `test_drivers/driver.F90` | ~1600 | Namelist parsing, NetCDF I/O, time loop. |
| `box_model_utils/modal_aero_calcsize.F90` | ~1500 | Non-trivial size/mass redistribution physics. Port carefully. |
| `box_model_utils/wv_saturation.F90` | ~1400 | Goff-Gratch / Flatau saturation vapor pressure; candidate for direct closed-form port. |
| `e3sm_src_modified/modal_aero_amicphys.F90` | (large) | Orchestrates gas-aerosol exchange, nucleation, coagulation, rename. |

## amicphys is self-contained — the standalone process modules are dead code

This caught us by surprise during M3.5 PR-B planning, so it's worth pinning down. The box-model `driver.F90` calls `modal_aero_amicphys_intr` (defined in `e3sm_src_modified/modal_aero_amicphys.F90:310`), and **this single module contains its own copies of all four sub-processes plus the orchestration**:

| `modal_aero_amicphys.F90` symbol | Lines | Role |
| --- | --- | --- |
| `modal_aero_amicphys_intr` | 310–1185 | Entry point. Called by `driver.F90:1283`. |
| `mam_amicphys_1gridcell` | 1190–1499 | Per-(col, level) orchestrator. |
| `mam_amicphys_1subarea_clear` | 2064–2626 | Clear-sky sub-area handler. |
| `mam_amicphys_1subarea_cloudy` | 1504–2059 | Cloudy sub-area handler (unused when `cldn=0`). |
| `mam_gasaerexch_1subarea` | 3279–3584 | Gas–aerosol exchange. |
| `mam_rename_1subarea` | 3923–4246 | Aitken → accum mode-transfer (renaming). |
| `mam_newnuc_1subarea` | 4251–4665 | Binary H₂SO₄–H₂O nucleation. |
| `mam_coag_1subarea` | 4670–5106 | Brownian coagulation. |

The standalone files `modal_aero_rename.F90`, `modal_aero_gasaerexch.F90`, `modal_aero_newnuc.F90`, `modal_aero_coag.F90` are **not invoked** by the box-model driver. They are real implementations of the same physics but are reachable only via a different orchestration path that the box-model build does not exercise. (`modal_aero_rename_sub` is called solely from `modal_aero_gasaerexch.F90:685`, which itself is unreachable from this driver.)

Implications for the JAX port:

- M3 ports targeting the box-model fixture must port the `mam_*_1subarea` versions inside `modal_aero_amicphys.F90`, not the standalone modules.
- "Smallest module by line count" is a misleading scoping heuristic: the standalone `modal_aero_rename.F90` is ~682 LOC but irrelevant; the active `mam_rename_1subarea` is ~323 LOC and inside a tightly-coupled orchestration.
- The `tests/reference/per_process/amicphys_{before,after}.npz` captures already bundle all four sub-processes' contributions (per ADR-012). Validating any single sub-routine in isolation requires a single-toggle re-run (e.g., `mdo_newnuc=1, others=0`) or mid-routine instrumentation.

## Fortran reference: E3SM infrastructure that can be short-circuited

The following modules are mostly empty shims in the box model and should be replaced with minimal JAX equivalents (or eliminated) rather than ported faithfully:

`ppgrid`, `pmgrid`, `spmd_utils`, `ref_pres`, `units`, `time_manager`, `cam_history`, `cam_logfile`, `cam_abortutils`, `dyn_grid`, `seasalt_model`, `modal_aero_convproc`, `modal_aero_deposition`, `aerodep_flx`, `phys_control`.

Re-implement only what `driver.F90` and the active microphysics actually call.

## Reference configuration

The Fortran build is configured via `test_drivers/cambox_config.cpp.in`:

```
-DMODAL_AERO -DMODAL_AERO_4MODE_MOM -DRAIN_EVAP_TO_COARSE_AERO
-DPCNST=35 -DPCOLS=1 -DPVER=1 -DNBC=1 -DNPOA=1 -DNSOA=1
```

This means: MAM4 with marine organics, 35 total tracers, single column, single vertical level. The JAX port targets this configuration as its primary validation surface.

## JAX port: actual layout

The four architectural questions that were "TBD" in the initial draft have all been resolved (see `docs/KEY_DECISIONS.md`):

| Question | Resolution | ADR |
| --- | --- | --- |
| Tracer representation | Flat `(pcols, pver, pcnst)` array mirroring Fortran, with named accessors on top (`get_number`, `get_mass`, `get_mass_by_species_name`). | ADR-008 |
| Process signature | Pure functional `process_fn(state, params, config) -> new_state`. | ADR-009 |
| Time-loop expression | Python `for` loop initially; `jax.lax.scan` deferred to Milestone 6. | ADR-004 |
| Configuration | Frozen `@dataclass` per namelist group + `RunConfig` + YAML loader. | ADR-010 |

**Package name:** `mam4_jax` (importable as `from mam4_jax import ...`).

**Current directory layout** (everything below exists in the repo; status reflects what is filled in vs. stubbed):

```
mam4_jax/
  __init__.py               # enables jax_enable_x64 at import
  config.py                 # @dataclass configs + load_yaml (ADR-010)
  constants.py              # physical constants from shr_const_mod.F90 / physconst.F90
  data.py                   # MAM4-MOM constants + IndexTables + accessors (ADR-008)
  saturation.py             # polysvp_water/ice, qsat_water/ice (ported, M3.1/M3.2)
  kohler.py                 # makoh_cubic/quartic + modal_aero_kohler (ported, M3.4 A/B)
  newnuc.py                 # binary_nuc_vehk2002, pbl_nuc_wang2008, mer07_veh02_nuc_mosaic_1box (ported, M3.6 PR-F1/F2)
  coag.py                   # getcoags, getcoags_wrapper_f (ported, M3.6 PR-G1/G2)
  _coag_tables.npz          # Whitby correction-factor tables (extracted from upstream data declarations)
  driver.py                 # run_step / run_timesteps — operator-splitting time loop (ported, M4)
  processes/
    calcsize.py             # modal_aero_calcsize_sub (ported, M3.5 A+B)
    wateruptake.py          # modal_aero_wateruptake_sub/_dr (ported, M3.4 C)
    amicphys.py             # orchestration shell (M3.6 PR-A); sub-routines pending
    gasaerexch.py           # M1 stub — dead code (see "amicphys is self-contained" above)
    newnuc.py               # M1 stub — dead code
    coag.py                 # M1 stub — dead code
    rename.py               # M1 stub — dead code
tests/
  reference/                # committed Fortran captures (NetCDF + .npz); see tests/reference/SCHEMA.md
  test_scaffolding.py       # M1 acceptance
  test_saturation.py        # M3.1 + M3.2 rel-err harness
  test_kohler.py            # M3.4 A+B
  test_wateruptake.py       # M3.4 C
  test_calcsize.py          # M3.5 A+B
scripts/
  build_reference.sh        # builds the Fortran reference (Homebrew/Linux auto-detect)
  capture_reference.py      # 7 modes — sweep / instrumented(+no-aitacc) / polysvp / qsat / makoh / kohler
  patches/                  # ADR-012 patch overlay (mam4_dump_state.F90 + driver instrumentation + transfer-off)
  reference_drivers/        # standalone Fortran main programs for the leaf-function ports (polysvp, qsat, makoh, kohler)
docs/
  ARCHITECTURE.md PROGRESS.md PLANS.md KEY_DECISIONS.md DEFERRED.md FEATURES.md REFERENCE_BUILD.md
  figures/                  # residual / diagnostic plots committed alongside each port PR
  plans/                    # archived approved plans (ADR-007)
```

Notable structural decisions that emerged during the port and weren't in the original sketch:

- **`constants.py` and `saturation.py` are package-level**, not under `processes/`. They are leaf math (no aerosol state), reused by multiple processes.
- **`kohler.py`, `newnuc.py`, `coag.py` are package-level** for the same reason — they're equilibrium / parameterization solvers consumed by `wateruptake` / `amicphys`, and are independently testable. `mam4_jax/coag.py` ships with a sibling `_coag_tables.npz` (extracted once from the upstream Fortran `data` declarations by `scripts/extract_coag_tables.py`).
- **`driver.py` landed in M4** (2026-05-22). Exposes `run_step(state)` and `run_timesteps(state, n_steps)`. Chains `calcsize → wateruptake → cloud_chem (no-op) → amicphys`. Validated end-to-end against a 60-step Fortran capture at max rel-err 1.97e-8.
- **The four `processes/{gasaerexch,newnuc,coag,rename}.py` stubs are kept** as dead M1 scaffolding, even though the box-model build never reaches them. Removing them would be a no-op refactor; they get deleted (or repurposed) when the corresponding `_mam_*_1subarea` helpers inside `amicphys.py` land in M3.6 PR-B/C/D/E.

See [PLANS.md](PLANS.md) for milestone sequencing and [KEY_DECISIONS.md](KEY_DECISIONS.md) for the architectural ADRs.
