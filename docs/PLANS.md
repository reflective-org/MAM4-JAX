# Plans

The forward-looking roadmap. Each milestone is broken into commit-sized subtasks. Status uses **proposed**, **in progress**, **done**, **deferred**. **Nothing should move from "proposed" to "in progress" without the owner's explicit approval** (rule #3).

When a milestone is in progress, its subtasks become the working task list. As subtasks complete they get a commit/PR link inline.

---

## Milestone 0 — Repo + documentation scaffold

**Status:** done.

- [x] Vendor Fortran reference, write `README.md` + initial `CLAUDE.md`. (`22f212d`)
- [x] Extract architecture and decisions out of `CLAUDE.md` into `docs/`; create `PROGRESS.md`, `PLANS.md`, `KEY_DECISIONS.md`, `DEFERRED.md`, `FEATURES.md`. (`a82e42d`)
- [x] Establish `docs/plans/` convention (ADR-007) and archive plan 001. (PR [#1](https://github.com/reflective-org/MAM4-JAX/pull/1))

---

## Milestone 1 — JAX package scaffolding

**Status:** done. See `docs/plans/001-scaffold-and-reference-capture.md` and ADRs 008–011.

- [x] Resolve open architectural ADRs: tracer representation (ADR-008), pure-functional signatures (ADR-009), dataclass+YAML config (ADR-010).
- [x] Tighten process discipline: all changes via PR, supersede ADR-006 (ADR-011).
- [x] `pyproject.toml` with top-level `mam4_jax/` layout and pinned floors for jax, jaxlib, numpy, netCDF4, pyyaml, matplotlib, pytest.
- [x] `mam4_jax/__init__.py` enables `jax_enable_x64` at import.
- [x] `mam4_jax/config.py`: four namelist-equivalent dataclasses + `RunConfig` + `load_yaml`.
- [x] `mam4_jax/data.py`: MAM4-MOM compile-time constants + sentinel-filled `IndexTables` + accessor helpers.
- [x] Seven `NotImplementedError`-raising stubs under `mam4_jax/processes/`.
- [x] `tests/test_scaffolding.py` with 12 assertions (all pass against jax 0.9.2 / pytest 9.0.2).

---

## Milestone 2 — Reference output capture

**Status:** done. See `docs/plans/001-scaffold-and-reference-capture.md`, ADR-011 (now superseded — used during planning) and ADR-012.

- [x] Build the Fortran reference locally via `scripts/build_reference.sh` (detects gfortran + NetCDF, applies `-fallow-invalid-boz` and the two-prefix `-L` paths).
- [x] Run the canonical 12-point convergence sweep; archive NetCDFs under `tests/reference/sweep/` (~1.7 MB).
- [x] Patch-overlay instrumentation (ADR-012): `scripts/patches/mam4_dump_state.F90` + `scripts/patches/driver_instrumentation.patch`, applied to the transient build copy of `driver.F90`. Hooks six points around `calcsize`, `wateruptake`, `amicphys`.
- [x] `scripts/capture_reference.py --mode instrumented` builds with overlay, runs, parses `.bin` dumps into `tests/reference/per_process/*.npz`.
- [x] `tests/reference/SCHEMA.md` documents both the NetCDF sweep contract and the `.npz` per-process contract.
- [x] `docs/REFERENCE_BUILD.md` documents prerequisites, build flag rationale, the missing-from-upstream `&size_parameters` namelist, and why `run_test.csh` is bypassed.

---

## Milestone 3 — First process ports (in progress)

**Status:** in progress.

1. [x] **`polysvp`** (within `wv_saturation.F90:699-736`) — Goff–Gratch saturation vapor pressure. Ported to `mam4_jax/saturation.py`; validated at max rel-err ~4e-15 (water and ice), eleven orders below ADR-003's 1e-6 tolerance. Reference: standalone Fortran driver (`scripts/reference_drivers/polysvp_driver.F90`) + `tests/reference/polysvp/reference.npz`. Plot: `docs/figures/polysvp_residuals.png`.
2. [x] **`qsat_water` and `qsat_ice`** — saturation specific humidity (`wv_saturation.F90:758-862`). Ported to `mam4_jax/saturation.py` alongside `qs_from_es` helper and `mam4_jax/constants.py` (physical constants from `shr_const_mod.F90`). Max rel-err 9.4e-14 / 7.8e-15. Reference: `scripts/reference_drivers/qsat_driver.F90` + `tests/reference/qsat/reference.npz`. Plot: `docs/figures/qsat_residuals.png`. **Note**: `qsat_ice` uses Clausius–Clapeyron (Fortran convention), not `polysvp_ice` — documented in the saturation module.
2.5. [x] **`IndexTables` populated** — extended the M2 instrumentation overlay with `dump_indices()`; captured to `tests/reference/indices/reference.npz`; hard-coded into `mam4_jax/data.py` as 0-based constants. `make_sentinel_tables()` retained for sentinel-raise tests; new `get_mass_by_species_name` accessor. Unblocks the aerosol-state-aware ports below.
3. **Water uptake port chain** — bottom-up split across three PRs:
   - 3a. [x] `makoh_cubic` + `makoh_quartic` — Cardano / Ferrari polynomial root finders (`modal_aero_wateruptake.F90:684-793`). Ported to `mam4_jax/kohler.py`; rel-err ~1e-14. Reference: `scripts/reference_drivers/makoh_driver.F90` + `tests/reference/makoh/reference.npz`. Plot: `docs/figures/makoh_residuals.png`.
   - 3b. [x] `modal_aero_kohler` (`modal_aero_wateruptake.F90:488-680`) — equilibrium solver consuming the polynomial root finders. Ported to `mam4_jax/kohler.py`; rel-err 9.8e-14 across a 168-point (rdry, hygro, s) grid. Reference: `scripts/reference_drivers/kohler_driver.F90` + `tests/reference/kohler/reference.npz`. Plot: `docs/figures/kohler_residuals.png`.
   - 3c. [x] `modal_aero_wateruptake_sub` + `_dr` (`:130-485`) — driver + per-column workhorse. Ported into `mam4_jax/processes/wateruptake.py` (replaces the M1 stub). Validated end-to-end against `tests/reference/per_process/wateruptake_{before,after}.npz`: `dgncur_awet` rel-err 4.5e-16, `wetdens` rel-err 2.1e-16, `qaerwat` rel-err 1.9e-7 at the 10⁻²⁰ floor (essentially zero qaerwat for the primary-carbon mode). Plot: `docs/figures/wateruptake_residuals.png`. **Wateruptake port complete.**
4. **`modal_aero_calcsize_sub`** (`modal_aero_calcsize.F90`) — size redistribution. Two-PR bottom-up split:
   - 4a. [x] **PR-A**: per-mode bounds adjustment + dgncur_a recomputation. Ported to `mam4_jax/processes/calcsize.py` (replaces the M1 stub); rel-err 2.1e-16 across 60 timesteps × 4 modes. Reference: new `tests/reference/per_process_no_aitacc/` captured with `do_aitacc_transfer_in=.false.` via `scripts/patches/disable_aitacc_transfer.patch`. Plot: `docs/figures/calcsize_residuals.png`.
   - 4b. [x] **PR-B**: Aitken ↔ accumulation mode-transfer block (Fortran lines 944–1294). Ported in `mam4_jax/processes/calcsize.py`; ``do_aitacc_transfer`` keyword (defaults to True, matching the box-model call). Validated against the refreshed `tests/reference/per_process/calcsize_{before,after}.npz` (nstep=60, full-transfer enabled): dgncur_a rel-err 2.1e-16. The transfer is a no-op in this fixture (see `docs/DEFERRED.md`); a structural test confirms `do_aitacc_transfer=True` ≡ `=False` on this fixture. **`modal_aero_calcsize_sub` is fully ported.**
5. **Amicphys** — the remaining microphysics. **Status (2026-05-22): all 7 sub-PRs merged; M3.6 complete.** **Scope correction (2026-05-19):** the standalone modules `modal_aero_newnuc.F90`, `modal_aero_coag.F90`, `modal_aero_gasaerexch.F90`, and `modal_aero_rename.F90` are **not invoked** by the box-model driver — `modal_aero_amicphys_intr` (in `e3sm_src_modified/modal_aero_amicphys.F90:310`) contains its own self-contained orchestration and four sub-routines (`mam_gasaerexch_1subarea`, `mam_rename_1subarea`, `mam_newnuc_1subarea`, `mam_coag_1subarea`). See `docs/ARCHITECTURE.md` for the full module map. The M3 amicphys port targets those internal sub-routines, not the standalone files. Multi-PR plan:
   - 5a. [x] **Orchestration shell**: ported in `mam4_jax/processes/amicphys.py` (replaces M1 stub). Implements `_mam_amicphys_1gridcell` → `_mam_amicphys_1subarea_clear` → four sub-process stubs in the Fortran order (gasaerexch → rename → newnuc → coag). Cloudy path not implemented (unreachable for `cldn=0`). Validation: bit-exact passthrough vs `tests/reference/per_process_amicphys_off/amicphys_{before,after}.npz` (captured with all `mdo_*=0`). Capture via `scripts/capture_reference.py --mode instrumented-amicphys-off`.
   - 5b. [x] `mam_rename_1subarea` (~323 LOC) — Aitken → accum mode-transfer. Ported in `mam4_jax/processes/amicphys.py` against the amicphys-local view (`qnum_cur`, `qaer_cur`, `qaer_delsub_grow4rnam`, `qwtr_cur`, `fac_m2v_aer`). Validated against `tests/reference/per_process/rename_{before,after}.npz` captured via the new `scripts/patches/rename_hook.patch` overlay. Max rel-err: qnum 2.5e-9, qaer 7.0e-10 across all 60 timesteps. The orchestration shell's call to rename is deferred to PR-C — wiring requires the state-dict ↔ amicphys-local-view unpacking that lands alongside `_mam_gasaerexch_1subarea`. Plot: `docs/figures/rename_residuals.png`. Plan: `docs/plans/002-rename-port.md`.
   - 5c. [x] **Foundation + wire rename**: state-dict ↔ amicphys-local-view unpacking layer (`_unpack_state_to_amicphys_view`, `_repack_amicphys_view_to_state` in `mam4_jax/processes/amicphys.py`) using a two-stage conversion (driver-side mmr→vmr via `MWDRY/ADV_MASS` + amicphys-internal vmr→local via `FCVT_*`). Wires `_mam_rename_1subarea` into the orchestration shell. Validated via new single-toggle Fortran capture `tests/reference/per_process_rename_only/` and the new test `test_orchestration_rename_only_matches_fortran` (rel-err < 1e-12 across 60 steps). Empirical finding: with gasaerexch off, the Fortran rename's optaa=40 guard trips and rename is a no-op every step, so the orchestration test is a full unpack/repack passthrough check. Plan: `docs/plans/003-foundation-rename-wiring.md`. **Scope expansion (2026-05-20):** reading `mam_gasaerexch_1subarea`'s source revealed it calls `mam_soaexch_1subarea` (~330 LOC) plus `gas_aer_uptkrates_1box1gas` (~148 LOC), so the original 4-PR remainder is now a 5-PR remainder.
   - 5d. [x] `mam_gasaerexch_1subarea` proper (~306 LOC) — H₂SO₄ analytical solver + leaf helpers (`gas_diffusivity`, `mean_molecular_speed`, `gas_aer_uptkrates_1box1gas`). Ported in `mam4_jax/processes/amicphys.py`. Wired into the orchestration via PR-C's unpack/repack scaffold. Validated against `tests/reference/per_process_gasaerexch_only/amicphys_after_writeback.npz` (single-toggle Fortran capture with SOA and pcarbon-aging both skipped via overlays). Max rel-err **7.8e-16** (machine ε) on the 5 gasaerexch-modified tracers across 60 timesteps. Plot: `docs/figures/gasaerexch_residuals.png`. Plan: `docs/plans/004-gasaerexch-no-soa-port.md`.
   - 5e. [x] `mam_soaexch_1subarea` (~330 LOC) — secondary-organic-aerosol condensation/evaporation (called from gasaerexch's body). Ported in `mam4_jax/processes/amicphys.py` under the single-substep assumption (`dtcur = dtfull`; adaptive sub-stepping deferred to a follow-up PR-E2 if a fixture ever triggers it). Wired unconditionally into `_mam_gasaerexch_1subarea`. Validated against the new `tests/reference/per_process_gasaerexch/` fixture (no `gasaerexch_skip_soaexch.patch`, only `skip_pcarbon_aging.patch`) — max rel-err **4.77e-15** (machine ε) on the 4 SOA tracers across 60 timesteps. Plot: `docs/figures/soaexch_residuals.png`. Plan: `docs/plans/005-soaexch-port.md`.
   - 5f. **`mam_newnuc_1subarea` (~415 LOC) — binary H₂SO₄–H₂O nucleation.** Owner-approved 3-PR split (2026-05-21) because the dependency chain adds another ~850 LOC (the standalone `modal_aero_newnuc.F90` helpers it delegates to):
     - 5f.PR-F1. [x] **Leaf parameterizations**: `binary_nuc_vehk2002` + `pbl_nuc_wang2008` ported to new `mam4_jax/newnuc.py`. Validated via the new standalone Fortran driver `scripts/reference_drivers/newnuc_helpers_driver.F90` (16 × 10 × 12 = 1920-record sweep across T, RH, [H₂SO₄], both PBL flagaa branches). Max rel-err **6.4e-11** on `binary rateloge`; everything else at ≤ 1.4e-14 (machine ε). Plan: `docs/plans/006-newnuc-helpers-port.md`. Plot: `docs/figures/newnuc_helpers_residuals.png`.
     - 5f.PR-F2. [x] **`mer07_veh02_nuc_mosaic_1box` dispatcher** (~580 LOC) — case-dispatch on `newnuc_method_flagaa` + Kerminen-Kulmala 2002 size correction + grown-particle composition + final deltas. Ported in `mam4_jax/newnuc.py` (~150 LOC after MAM4-MOM-specific simplifications: no ternary, no `nsize>1`, no NH₃-aware composition). Validated against a new standalone Fortran driver sweep (2160 records across 5 regimes: subcutoff / low-rate / active no-PBL / active PBL / gas-limited). Max rel-err **2.27e-12** on `qnuma_del`, `qso4a_del`, `qh2so4_del`, `dnclusterdt`. Plot: `docs/figures/mer07_veh02_residuals.png`. Plan: `docs/plans/007-mer07-veh02-dispatcher-port.md`.
     - 5f.PR-F3. [x] **`mam_newnuc_1subarea` orchestration** (~415 LOC) — amicphys glue wiring the PR-F2 dispatcher into `_mam_amicphys_1subarea_clear`. Extended `_mam_gasaerexch_1subarea` to return `qgas_avg` (so newnuc can consume the time-averaged H₂SO₄ vmr). Added `zmid` / `pblh` / `relhum` to the state-dict contract. Validated against new `tests/reference/per_process_gasaerexch_and_newnuc/` single-toggle fixture (mdo_gasaerexch=1, mdo_newnuc=1, others=0 + skip_pcarbon_aging). Max rel-err **3.9e-16** (machine ε) on the 3 newnuc-affected tracers (H₂SO₄ gas, Aitken number, Aitken so4 mass). **M3.6 PR-F (newnuc) complete.** Plot: `docs/figures/newnuc_orchestration_residuals.png`. Plan: `docs/plans/008-newnuc-orchestration-port.md`.
   - 5g. **`mam_coag_1subarea` (~437 LOC) — Brownian coagulation kernels.** Owner-approved 3-PR split (2026-05-21) because the closed-form coagulation-coefficient leaf (`getcoags`, ~1685 LOC including ~1200 LOC of correction-factor lookup tables) plus its wrapper plus the subarea orchestration cleanly factor into three reviewable chunks:
     - 5g.PR-G1. [x] **`getcoags` leaf** (~250 LOC of physics + extracted lookup tables): closed-form Whitby coagulation coefficients. Ported in new module `mam4_jax/coag.py`. Lookup tables extracted once from the Fortran `data` declarations into `mam4_jax/_coag_tables.npz` by `scripts/extract_coag_tables.py`. Validated against a new standalone Fortran driver `coag_coefficients_driver.F90` sweeping (4 T × 2 P × 5 dgnumA × 6 dgnumB = 240 records); the fixture also captures `getcoags_wrapper_f` outputs so PR-G2 reuses the same `.npz`. Max rel-err **6.5e-9** across all 8 outputs. Plot: `docs/figures/getcoags_residuals.png`. Plan: `docs/plans/009-getcoags-port.md`.
     - 5g.PR-G2. [x] **`getcoags_wrapper_f`** (~130 LOC): wraps `getcoags` with the prep math (lamda / amu / knc / kfmat* from T, P, densities) and post-processes the 8 raw coefficients into the 8 `betaij*` / `betaii*` / `betajj*` coefficients consumed by `mam_coag_1subarea`. Ported in `mam4_jax/coag.py` (~70 new LOC); added `PSTD` and `TMELT` to `mam4_jax/constants.py`. Validated against the wrapper-output section of the PR-G1 fixture (no new Fortran capture needed) — 7/8 outputs at machine ε, `betaij2j` at 6.5e-9 inherited from `qs21`. Plot: `docs/figures/getcoags_wrapper_residuals.png`. Plan: `docs/plans/010-getcoags-wrapper-port.md`. Folded into PR #23 alongside PR-G1.
     - 5g.PR-G3. [x] **`mam_coag_1subarea` orchestration** (~437 LOC Fortran → ~140 LOC JAX after MAM4-MOM trimming): amicphys glue wiring the PR-G2 wrapper into `_mam_amicphys_1subarea_clear`. 3 active coag pairs (Aitken→accum, pcarbon→accum, Aitken→pcarbon); marine-organics blocks dropped (modes absent); pcarbon-aging input not accumulated (matching capture applies `skip_pcarbon_aging.patch`). Two-branch number-loss `if (tmpa < 1e-5)` and mass-transfer `if (tmpc > epsilonx2)` guards reformulated as `jnp.where` with safe-division. New capture mode `instrumented-coag-only` → `tests/reference/per_process_coag/`. Max rel-err **4.1e-13** across 33 aerosol-slot tracers × 60 timesteps. Gas-tracer slots excluded from comparison (driver-level gas-chem stub at `driver.F90:1249` adds H₂SO₄ outside amicphys — coag doesn't touch gases so this is not part of coag's validation surface). Plot: `docs/figures/coag_orchestration_residuals.png`. Plan: `docs/plans/011-coag-orchestration-port.md`. **M3.6 (amicphys) complete.**

Each sub-routine port (5d–5g) needs a single-toggle capture (e.g., `mdo_gasaerexch=1, others=0`) so its effect can be isolated from the others. Final validation reuses the existing `tests/reference/per_process/amicphys_{before,after}.npz` (full-bundle, 60-step) once all sub-processes are in place.

Each port lands as its own PR following the validation workflow in `CLAUDE.md` (capture reference, port, diff to `1e-6`, plot residuals, log in `PROGRESS.md`).

---

## Milestone 4 — Operator-splitting time loop (done)

**Status:** done (2026-05-22). M3.6 complete; M4 landed across two PRs.

- **PR-M4-A** [x]: scaffold `mam4_jax/driver.py` with `run_step` and `run_timesteps`; 1-step test against a new `instrumented-full-minus-pcarbon-aging` Fortran capture (max rel-err **2.5e-9** on `q`). Cloud-chem stubbed as a no-op (box-model fixture has `cldn=0`); gas-chem term stays inside gasaerexch's analytical solver (no operator-splitting refactor). Plan: `docs/plans/012-driver-scaffold.md`.
- **PR-M4-B** [x]: 60-step trajectory test against the same fixture — **max rel-err 1.97e-8** at step 29 on Aitken-mode number (tracer 17), 50× under ADR-003. Errors flatten by step ~5; no runaway accumulation. Mode-by-mode size-distribution comparison figure shows JAX (dashed) overlaying Fortran (solid) across the full integration. Plan: `docs/plans/013-driver-trajectory-and-figure.md`. Plot: `docs/figures/driver_60step_trajectory.png`.

Initial implementation is a Python `for` loop (rule #8 phase A); `jax.lax.scan` is deferred to Milestone 6.

---

## Milestone 5 — Convergence test reproduction (partial, accepted as final-on-main)

**Status:** partial-and-final on `main` (2026-05-22). 6 of 12 step counts validated; the remaining 6 are permanently `xfail` on `main` per ADR-013.

- **PR-M5 (final on main)** [x partial]: reproduces the 12-point timestep sweep from `run_test.csh` against `tests/reference/sweep_no_pcarbon_aging/*.nc`. Validates **`nstep ∈ {60, 120, 180, 360, 900, 1800}`** at `rtol=1e-6` on `num_aer`/`so4_aer`/`soa_aer`/`h2so4_gas`/`soag_gas` for every captured timestep. Worst rel-err **1.98e-8** in that half (50× under ADR-003). The other 6 step counts (`nstep ≤ 30`, `dt ≥ 60s`) are marked `xfail` because Fortran's `mam_soaexch_1subarea` adaptive substepping kicks in there and the `main` JAX port doesn't (per ADR-013 the handwritten port is intentionally skipped; resolution lives on the `diffrax` branch). Plan: `docs/plans/014-convergence-sweep.md`. Plot: `docs/figures/sweep_convergence.png`.
- ~~PR-E2 (handwritten adaptive substepping)~~ — **CANCELLED** per ADR-013. Adaptive substepping is the `diffrax` branch's job, not `main`'s.

**Possible scope expansion (still `main`)**: NetCDF output emission from JAX (so the post-process notebook works against JAX outputs). Defer indefinitely — only useful if a downstream tool needs it.

---

## Milestone 6 — Audit + JAX-idiom optimization (proposed)

**Status:** proposed. Rule #8 phase B. After correctness, perform a sweep for:

- `jax.jit` boundaries.
- `jax.vmap` for column/level dimensions.
- `jax.lax.scan` for the time loop.
- `jax.lax.cond` / `where` for branchy code paths.
- Sharding decisions (single-host CPU first; GPU/TPU later if owner wants).
- Differentiability audit (which processes admit autodiff cleanly).

Each optimization lands as its own PR with a before/after correctness check (still `1e-6`) and a benchmark.

---

## Milestone 7 — Diffrax migration (long-lived `diffrax` branch)

**Status:** owner-introduced 2026-05-21; promoted to dual-branch strategy 2026-05-22 (see ADR-013).

**Branching model.** M7 lives on a separate long-lived `diffrax` branch parallel to `main`, *not* as a sequence of PRs into `main`. Rationale and invariants in ADR-013. Summary:

- `main` keeps handwritten solvers (current state), including the 6 `nstep ≤ 30` `xfail`s on the convergence sweep.
- `diffrax` branch replaces handwritten solvers with diffrax equivalents; dynamic substepping comes from diffrax's standard adaptive controller. The 6 `xfail`ed cases are expected to pass on this branch.
- Both branches stay structurally similar (same module layout, function names, state-dict contract, test fixtures).

**Solvers in scope.** Each is a separate sub-PR within the `diffrax` branch:

- H₂SO₄ analytical solver in `_mam_gasaerexch_1subarea` (the three-branch `tmp_kxt` / Taylor / exp formula at `processes/amicphys.py:603-622`).
- SOA exchange step-1/step-2 semi-implicit solver in `_mam_soaexch_1subarea` — this is where adaptive substepping resolves the `nstep ≤ 30` gap.
- Coag's analytical number-loss and exp-decay mass transfer in `_mam_coag_1subarea` (if a stiffer fixture ever motivates revisiting the closed-form).

**Pros.** JIT/grad/vmap-clean; better numerics on stiff systems (Kvaerno5, KenCarp4); adaptive stepping for free; standard diagnostics/error estimators; resolves the M5 `nstep ≤ 30` gap on the `diffrax` branch without polluting `main`.

**Cons.** Adds runtime dependency (~3 MB on the `diffrax` branch only). Per-step output may differ from Fortran by ~1 ULP because the solver choice and tolerances differ; cross-validation against Fortran at 1e-6 becomes trickier on stiff problems.

**Sequencing.** Each handwritten solver is replaced one at a time on the `diffrax` branch with a per-solver PR (into the `diffrax` branch, not `main`). The test suite from `main` is reused unchanged — `tests/test_sweep.py`'s currently-`xfail`ed cases should flip to expected-pass.

**Out of scope on `main`.** The previously-planned handwritten "PR-E2" (adaptive SOA substepping ported to `main`) is cancelled per ADR-013.

**Out of scope today.** Spec-level decisions (which solver family, which tolerances, how to validate against Fortran) — those happen when this milestone is approved to "in progress".

---

*Whenever a milestone moves from "proposed" to "in progress", flesh out its subtasks here in the same PR.*
