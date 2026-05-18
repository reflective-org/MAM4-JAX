# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mission

Port the **MAM4 (Modal Aerosol Module, 4-mode)** aerosol-microphysics box model from Fortran 90 to **JAX**, preserving scientific behavior to within a configurable relative tolerance (start at `1e-6`).

The original Fortran reference implementation lives in `mam4-original-src-code/` (an upstream snapshot of the kaizhangpnl/MAM_box_model repository). The JAX port has not been started yet — when scaffolding the Python package, treat the Fortran tree as **read-only reference**.

## Non-negotiable working rules (from the project owner)

These override default Claude Code behavior. Re-read them before starting any task.

1. **Plan first, always.** Every task begins with a written plan: logical tasks broken into commit-sized subtasks. No code changes without an agreed plan.
2. **Each meaningful unit of progress becomes its own PR** so the owner can review before the next step.
3. **No silent assumptions.** Before making any modeling, numerical, or API choice (even small ones — tolerances, dtype, function signatures, file layout), ask. When in doubt, ask.
4. **Do not mention Claude, AI, or auto-generation in commit messages or PR descriptions.** Write them as if a human engineer authored them.
5. **Maintain living docs in `docs/`**: `README.md`, `ARCHITECTURE.md`, `PROGRESS.md`, `PLANS.md`, `KEY_DECISIONS.md`, `DEFERRED.md`, `FEATURES.md`. Update them as part of the same PR that introduces the change they describe.
6. **Scientific integrity comes first.** Every ported component must be validated against the Fortran reference. Add plots whenever a visualization clarifies a discrepancy, convergence behavior, or process result — diagnostic figures are first-class deliverables, not decoration.
7. **Target relative error: `1e-6`.** This threshold may be relaxed *only* with explicit owner approval, documented in `KEY_DECISIONS.md` with a justification.
8. **Port properly, in two phases.**
   - Phase A — **scaffold** the JAX package structure end-to-end (modules, signatures, data flow, tests) before filling in physics.
   - Phase B — fill in physics, validate, then perform a **code audit + JAX-idiom optimization pass** (vectorization, `jit`, `vmap`, `scan`, sharding decisions) after correctness is established. Do **not** prematurely optimize during initial porting.
9. **Default precision is `float64`** everywhere. Aerosol microphysics is stiff and quantity ratios span many orders of magnitude; do not silently downcast. If JAX requires `jax.config.update("jax_enable_x64", True)`, set it at package import.
10. **Ask, ask, ask.** Even for small tweaks. Better to ask one extra question than to commit a wrong assumption.
11. **Project values:** scientific integrity, modern language idioms, thorough documentation, collaborative review, reproducibility, transparency. Trade off against these explicitly when forced to.

## Behavioral guardrails

These bias toward caution over speed. For trivial tasks (typo fixes, one-line doc edits), use judgment. For everything else, treat them as binding.

### Think before coding — don't assume, don't hide confusion, surface tradeoffs

Before implementing:
- **State assumptions explicitly.** If uncertain, ask.
- **If multiple interpretations exist, present them** — don't pick silently.
- **If a simpler approach exists, say so.** Push back when warranted.
- **If something is unclear, stop.** Name what's confusing. Ask.

### Simplicity first — minimum code that solves the problem, nothing speculative

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you wrote 200 lines and it could be 50, rewrite it.

Senior-engineer test: would they call this overcomplicated? If yes, simplify.

### Surgical changes — touch only what you must; clean up only your own mess

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, **mention it — don't delete it**.

Orphans from your edits:
- Remove imports / variables / functions that **your** changes made unused.
- Do **not** remove pre-existing dead code unless asked.

Acceptance test: every changed line traces directly to the request.

### Goal-driven execution — define success criteria, then loop until verified

Translate vague tasks into verifiable goals before writing code:
- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after."
- "Port subroutine Y to JAX" → "Diff JAX output against captured Fortran reference; max rel-err < 1e-6."

For multi-step work, state the plan up front in the form:

```
1. <step>  → verify: <check>
2. <step>  → verify: <check>
3. <step>  → verify: <check>
```

Strong, checkable criteria let you iterate without re-asking. Weak ones ("make it work") leak ambiguity into the diff.

**These guardrails are working when:** diffs contain fewer drive-by changes, fewer rewrites are needed due to overcomplication, and clarifying questions arrive **before** implementation rather than after mistakes ship.

## Repository layout

```
mam4-jax/                              # repo root (this directory)
  CLAUDE.md                            # this file
  mam4-original-src-code/              # READ-ONLY Fortran reference (upstream snapshot)
    e3sm_src/                          # MAM4 modules identical to E3SMv1
    e3sm_src_modified/                 # E3SMv1 MAM4 modules with box-model edits
    box_model_utils/                   # Box-model shims for E3SM infrastructure (ppgrid, physconst, wv_saturation, ...)
    test_drivers/                      # driver.F90 + cambox_config.*.in build config
    postprocess/                       # postprocess.ipynb (NetCDF -> figures)
    Makefile, run_test.csh             # Fortran build + convergence test harness
```

The JAX package, test harness, validation data, and docs do not exist yet — propose their layout in `PLANS.md` and confirm with the owner before scaffolding.

## How to build & run the Fortran reference (for validation)

The Fortran reference produces the "ground truth" the JAX port must reproduce. To regenerate reference outputs:

```bash
cd mam4-original-src-code/
./run_test.csh        # csh script: builds in build/, runs in run/, writes mam_output.nc
```

Notes before running:
- Requires `gfortran` (default) or `ifort`, plus a NetCDF Fortran library. Set `NETCDF_LIB` and `NETCDF_INCLUDE` env vars (see `test_drivers/cambox_config.make.in`).
- `run_test.csh` currently hard-codes an `outpath` belonging to a previous developer (`/Users/sunj695/...`). Either edit it or remove the post-run `mv` before running.
- Default build flags (`test_drivers/cambox_config.cpp.in`) configure **MAM4 with marine organics** (`-DMODAL_AERO_4MODE_MOM`), `PCNST=35`, `PCOLS=1`, `PVER=1` — single-column, single-level box. Do not change these flags without owner approval; they define the reference configuration the port targets.
- The script sweeps timestep counts `(1 2 4 9 18 30 60 120 180 360 900 1800)` over a fixed 1800 s window — this is a **convergence test** suite. Reproducing this sweep is a key validation milestone for the JAX port.

Namelist (built inline by `run_test.csh`) controls process toggles (`mdo_gaschem`, `mdo_gasaerexch`, `mdo_rename`, `mdo_newnuc`, `mdo_coag`), meteorology (`temp`, `press`, `RH_CLEA`), and initial aerosol/gas mixing ratios. Use the same namelist values when generating reference data for any given JAX test.

## MAM4 architecture in one screen

Understanding this before touching code saves hours.

**Domain.** Aerosols are represented as four log-normal **modes** (Aitken, accumulation, coarse, primary-carbon), each carrying a number concentration and per-species mass concentrations. The 4-mode-with-marine-organics (`MOM`) variant is the configured reference. There are also **cloud-borne** counterparts (`qqcw`) that mirror the interstitial tracers.

**Time-stepping pattern.** The driver applies **sequential operator splitting** over each `mam_dt` step. The microphysical processes, in order, are:

1. `modal_aero_calcsize` — recompute dry diameters from number + mass; enforce mode size bounds, transfer particles between modes when violated.
2. `modal_aero_wateruptake` — equilibrium water uptake (Köhler-ish), gives wet diameter & wet density.
3. `modal_aero_amicphys` — the umbrella "amicphys" call that internally runs:
   - `modal_aero_gasaerexch` (H2SO4 / SOAG condensation onto modes)
   - `modal_aero_newnuc` (binary H2SO4–H2O nucleation, Vehkamäki-style)
   - `modal_aero_coag` (intra- and inter-modal Brownian coagulation)
   - `modal_aero_rename` (transfer aged Aitken → accumulation when size criteria met)
4. Simple gas/cloud chemistry (`gaschem_simple`, `cloudchem_simple`) — placeholders in the box model.

Each sub-process is called as a Fortran subroutine that mutates the `q(:,:,pcnst)` tracer array in place. The mapping from `pcnst` indices to (mode, species) is held in `modal_aero_data` (`lmassptr_amode`, `numptr_amode`, etc.) — **this index bookkeeping is the single largest source of porting bugs**; surface it explicitly in the JAX data model rather than hiding it behind integer indirections.

**Heaviest reference modules** (line counts, useful for scoping):
- `box_model_utils/physics_buffer.F90` (~6500) — E3SM physics-buffer infrastructure; mostly stub-able in JAX.
- `test_drivers/driver.F90` (~1600) — namelist parsing, I/O, time loop.
- `box_model_utils/modal_aero_calcsize.F90` (~1500) — non-trivial physics, port carefully.
- `box_model_utils/wv_saturation.F90` (~1400) — Goff-Gratch / Flatau saturation vapor pressure; consider replacing with a direct closed-form port.

**E3SM infrastructure to short-circuit in JAX.** `ppgrid`, `pmgrid`, `spmd_utils`, `ref_pres`, `units`, `time_manager`, `cam_history`, `cam_logfile`, `cam_abortutils`, `dyn_grid`, `seasalt_model`, `modal_aero_convproc`, `modal_aero_deposition`, `aerodep_flx`, `phys_control` are mostly empty shims in the box model — re-implement only what `driver.F90` and the active microphysics actually call.

## Validation workflow (apply to every port PR)

1. Identify the smallest Fortran subroutine being ported.
2. Generate reference inputs/outputs by instrumenting the Fortran code or capturing from a `run_test.csh` run.
3. Port to JAX, `float64`, no `jit` yet.
4. Diff against reference; require `max relative error < 1e-6` element-wise.
5. Produce a plot (when meaningful: time series, scatter of JAX-vs-Fortran, residual histogram). Save under `docs/figures/`.
6. Record the result in `PROGRESS.md` with the PR number.
7. Only after correctness lands: open a follow-up PR for `jit`/`vmap` and update `PROGRESS.md`.

## Git, commits, PRs

- `mam4-original-src-code/` is a **vendored** frozen snapshot — no nested `.git/`, no submodule. Provenance (upstream URL, commit SHA, date) is recorded in `README.md`. Do not modify files under that directory; to refresh the snapshot, replace it wholesale and update the provenance table in the same PR.
- Branch per task; PR per logical task as defined in `PLANS.md`.
- Commit messages: imperative mood, scope-prefixed (`scaffold:`, `port:`, `docs:`, `test:`, `fix:`). No Claude/AI attribution. No emojis unless the owner requests them.
- Open PRs against `main` and tag with the matching `PLANS.md` task ID.
