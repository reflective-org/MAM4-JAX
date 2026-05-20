# Architecture

This document describes the architecture of the MAM4 aerosol-microphysics model as represented in the Fortran reference under `mam4-original-src-code/`, and the **proposed** structure of the JAX port. The Fortran section is factual. The JAX section is a proposal that has **not** yet been approved by the owner — items marked **TBD** await explicit decisions before scaffolding begins.

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

## JAX port: proposed layout (TBD — pending owner approval)

> Nothing in this section has been built yet. Any module name, signature, or directory below is a proposal. Confirm with the owner before scaffolding.

**Proposed package name:** `mam4_jax` (importable as `from mam4_jax import ...`).

**Proposed directory layout (TBD):**

```
mam4_jax/                  # JAX package source — TBD
  __init__.py
  config.py                # float64 enable, mode/species constants
  data.py                  # mode/species index bookkeeping, tracer layout
  processes/
    calcsize.py
    wateruptake.py
    gasaerexch.py
    newnuc.py
    coag.py
    rename.py
    amicphys.py            # composes the four above in order
  driver.py                # operator-splitting time loop
tests/                     # validation harness — TBD
  reference/               # captured Fortran outputs (NetCDF or .npz)
  test_<process>.py        # one per process, asserts rel-err < 1e-6
scripts/                   # CLI utilities — TBD
  capture_reference.py     # run Fortran + dump per-process inputs/outputs
  plot_residuals.py        # diagnostic figures
```

**Open architectural questions (TBD, owner decision needed):**
- Tracer representation: flat `(pcols, pver, pcnst)` array (mirror Fortran) **or** a structured `pytree`/`dataclass` per (mode, species)? The Fortran approach is bug-prone but enables exact diffing; the pytree approach is JAX-native.
- Pure-functional process signatures (return new state) vs. in-place semantics (return only deltas).
- Whether to express the operator-splitting loop with `jax.lax.scan` from the start, or keep it as a Python `for` loop until correctness lands (rule #8 phase-A says the latter).
- How to expose namelist-style configuration: dataclass, dict, or YAML config files.

See [PLANS.md](PLANS.md) for milestone sequencing once these are resolved.
