# Plans

The forward-looking roadmap. Each milestone is broken into commit-sized subtasks. Status uses **proposed**, **in progress**, **done**, **deferred**. **Nothing should move from "proposed" to "in progress" without the owner's explicit approval** (rule #3).

When a milestone is in progress, its subtasks become the working task list. As subtasks complete they get a commit/PR link inline.

**GitHub bookkeeping.** Milestones M6 onwards are mirrored on GitHub Milestones; each milestone block below links to its tracker. PRs that belong to a milestone include `--milestone <M_n>` at creation so the milestone page becomes a discoverable index of related work. Standalone deferred items that warrant their own work-tracking are filed as GitHub Issues (cross-linked from `docs/DEFERRED.md`); PRs that close them use the `Closes #N` body convention so the Development linkage closes the issue automatically. M0–M5 predate this convention and aren't backfilled.

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

## Milestone 6 — Audit + JAX-idiom optimization (done)

**Status:** done (2026-05-28). All 5 planned sub-PRs landed on the `diffrax` branch over a single day; PR-J6 (sharding) deferred to its own milestone (now M13). The diffrax branch tip `5ea6330` is tagged `diffrax-v0.1.0`. Merge-back to `main` per ADR-016 is deferred (owner directive 2026-05-28: maintain `diffrax` as parallel canonical for now). GitHub: [milestone 1](https://github.com/reflective-org/MAM4-JAX/milestone/1).

**Why on `diffrax`, not `main`?** M6 will exercise the diffrax-tied codepaths (`solve_ivp`, the new `_h2so4_rhs` / `_soaexch_rhs` RHS functions, etc.) that only exist on `diffrax`. Doing M6 on `diffrax` first means `main` gets the JIT-compiled (fast) version at merge-back. Uncompiled diffrax is ~50× slower than handwritten; JIT-compiled it becomes competitive (PR-D2 observation).

**Acceptance bar per sub-PR.** Each M6 sub-PR must:
- Preserve the 24 h / 3 % bar on `tests/test_sweep.py[1|5]` (ADR-015).
- Preserve the existing per-process tests at their current bars (1e-6 on most; per-test caveats apply).
- Include a before/after **wall-time benchmark** at one representative case (24 h dt=1 s, since that's the slowest and the place JIT helps most).
- Per-mode rel-err breakouts in the PR description if any test's tolerance moves at all (per `project-mam4-per-mode-breakouts`).

**Sub-PRs.** Each landed as its own PR on the `diffrax` branch; per-PR detail in PROGRESS.md.

1. **PR-J1** [x] — `@jax.jit` on `run_step` and per-process subareas. ~70× speedup at dt=1 s after compile. PR [#39](https://github.com/reflective-org/MAM4-JAX/pull/39). Plan: `docs/plans/018-m6-pr-J1-jit.md`.
2. **PR-J2** [x] — `jax.lax.scan` for the driver time loop. Cumulative 113× speedup at dt=1 s; ~80 min → seconds for 24 h trajectories. PR [#40](https://github.com/reflective-org/MAM4-JAX/pull/40). Follow-up PR [#41](https://github.com/reflective-org/MAM4-JAX/pull/41) wrapped `run_timesteps` itself in `@jax.jit` (static `n_steps`) to amortise Python-side dispatch over the 1000-sim benchmark; JAX ended up 3.8× faster than Fortran per simulation.
3. **PR-J3** [x] — `jax.vmap` audit. Codebase already vmap-clean by design (trailing-axis reductions, no leading-axis assumptions); two regression tests added. PR [#42](https://github.com/reflective-org/MAM4-JAX/pull/42). Bundled fix: 4 inherited tests still gated at ADR-003's 1e-6 against Fortran fixtures the diffrax port no longer matches bit-for-bit; relaxed to ADR-015's bar with appropriate `atol` for near-zero tracers.
4. **PR-J4** [x] — `jax.lax.cond` / `where` audit. Zero code changes; documentation only. Grep-verified: 37 control-flow statements all operate on Python-static values; 127 `jnp.where` calls are correct; zero `bool()/.item()/print/assert` inside JIT scope. PR [#43](https://github.com/reflective-org/MAM4-JAX/pull/43).
5. **PR-J5** [x] — reverse-mode autodiff audit. Codebase is autodiff-clean; `jax.grad` through `run_step` and through 60-step `scan` returns finite, deterministic cotangents. Four regression tests added. PR [#44](https://github.com/reflective-org/MAM4-JAX/pull/44). Solver-side adjoint: `solve_ivp` uses the default `diffrax.RecursiveCheckpointAdjoint` (memory-feasibility caveat at 24 h trajectory lengths flagged in PROGRESS.md; switch to `ImplicitAdjoint()` if a future calibration workflow hits a memory ceiling).
6. **PR-J6** — Sharding. **Deferred to its own milestone** (single-host CPU is the current target; GPU/TPU sharding is a distinct effort, not a sub-task of M6).

---

## Milestone 7 — Diffrax migration (long-lived `diffrax` branch)

**Status:** core PRs done (2026-05-26 through 2026-05-28). PR-I1, PR-D1, PR-D2 landed; PR-D3 permanently deferred (see `DEFERRED.md`). Merge-back to `main` per ADR-016 is deferred (owner directive 2026-05-28: maintain `diffrax` as parallel canonical until further notice). GitHub: [milestone 2](https://github.com/reflective-org/MAM4-JAX/milestone/2).

**Branching model.** M7 lives on a long-lived `diffrax` branch parallel to `main`. Rationale and invariants in ADR-013; the merge-back intent and the `main → diffrax` sync convention are in ADR-014. Summary:

- `main` keeps handwritten solvers, including the 6 `nstep ≤ 30` `xfail`s on the convergence sweep. The `v0.1.0` tag (created during PR-I1) anchors this baseline.
- `diffrax` branch replaces handwritten solvers with diffrax equivalents; dynamic substepping comes from diffrax's standard adaptive controller. The 6 `xfail`ed cases are expected to pass.
- Both branches stay structurally similar (same module layout, function names, state-dict contract, test fixtures). Non-solver changes land in `main` first and reach `diffrax` via periodic `main → diffrax` merges (ADR-014).
- Eventually, once the diffrax port is validated end-to-end, `diffrax` merges back into `main` and becomes the canonical implementation (ADR-014).

**Sub-PRs on the `diffrax` branch.** Each landed as its own PR; per-PR detail in PROGRESS.md.

1. **PR-I1** [x] — Infra & tooling. `v0.1.0` tag on `main`; `diffrax` added to `pyproject.toml` on the branch; `mam4_jax/solvers.py` strategy module; ADR-014; `docs/HANDWRITTEN_SOLVER_LIMITATIONS.md`. PR [#31](https://github.com/reflective-org/MAM4-JAX/pull/31). Plan: `docs/plans/015-diffrax-infra.md`.
2. **PR-D1** [x] — Port `_mam_soaexch_1subarea` to diffrax (`Kvaerno5`). Empirical finding: the 6 previously-`xfail`ed M5 cases now pass at ADR-015's 3 %/24 h/dt≤5 s bar (the original 1 %/1 e-6 target was unattainable due to a structural per-step semi-implicit-vs-true-ODE difference that accumulates to ~2.4 % on `soag_gas`, dt-independent — see `project-diffrax-structural-offset`). Test sweep rewritten from M5's 12-point sweep to a 4-dt 24 h test. PR [#34](https://github.com/reflective-org/MAM4-JAX/pull/34). Plan: `docs/plans/016-diffrax-soaexch.md`.
3. **PR-D2** [x] — Port H₂SO₄ analytical solver to diffrax. Validates the `solvers.py` abstraction on a closed-form linear-in-`g` ODE. Confirmed (via H₂SO₄-only ablation) that the 2.4 % `soag_gas` offset is SOA-specific, not a diffrax-vs-handwritten artifact in general — H₂SO₄ + SO4 conserve to ε, only SOA drifts (~0.35 % over 24 h). PR [#36](https://github.com/reflective-org/MAM4-JAX/pull/36). Plan: `docs/plans/017-diffrax-h2so4.md`.
4. **PR-D3** — Coag analytical solvers. **Permanently deferred** (PR [#37](https://github.com/reflective-org/MAM4-JAX/pull/37)); see `docs/DEFERRED.md` entry "Coag analytical → diffrax (PR-D3)" for reasoning (coagulation is algebraic, not an ODE in time; no validation gap, no stiffness motivation, no autodiff motivation on this fixture).

**Pros.** JIT/grad/vmap-clean; better numerics on stiff systems (Kvaerno5, KenCarp4); adaptive stepping for free; standard diagnostics/error estimators; resolves the M5 `nstep ≤ 30` gap without polluting `main` before the merge-back.

**Cons.** Adds runtime dependency (~3 MB on the `diffrax` branch only). Per-step output may differ from Fortran by ~1 ULP because the solver choice and tolerances differ; cross-validation against Fortran at 1e-6 becomes trickier on stiff problems (ADR-013 allows ~1 ULP slack but otherwise enforces ADR-003's 1e-6).

**Validation discipline.** The validation bar against Fortran stays at `rtol=1e-6` (ADR-003). The diffrax controller's *internal* tolerances are much tighter (starting defaults: `rtol=1e-9`, `atol=1e-12`) so that the validation residual is dominated by physics-model differences, not by solver truncation error. Both tolerances are tunable per PR.

**Out of scope on `main`.** The previously-planned handwritten "PR-E2" (adaptive SOA substepping ported to `main`) is cancelled per ADR-013.

---

## Milestone 8 — Cloud chemistry port (proposed)

**Status:** proposed 2026-05-28; owner priority slot 1 of the M8–M13 stack. Runs on the `diffrax` branch. GitHub: [milestone 3](https://github.com/reflective-org/MAM4-JAX/milestone/3).

**Why.** The current JAX port runs `cloud_chem_simple_sub` as a no-op because the box-model fixture has `cldn=0`, which gates off Fortran's `if (cld > 1e-6)` at `driver.F90:1263`. To exercise real cloud-chem behavior the JAX port needs a Fortran reference with `cldn > 1e-6` and the corresponding cloudy amicphys path implemented in JAX.

**Scope sketch (subject to a proper plan).** Port `mam_amicphys_1subarea_cloudy` (the cloudy counterpart to the clear-subarea path) plus any cloud-chem-specific helpers it pulls in. Capture a new Fortran reference fixture with non-zero `cldn` for validation. Stub `cloud_chem_simple_sub` at `driver.py:70-79` becomes a live call instead of a no-op.

**Open questions.**
- Cloudchem-simple only, or the full `modal_aero_cloudchem_intr` (the latter has SO2 → H2O2 → SO4 in-cloud conversion)?
- FEATURES.md currently lists "sulfur chemistry beyond stubs" as out of scope — does cloud-chem porting force revisiting that?
- New fixture namelist: how does `cldn>0` interact with `mdo_gaschem` and the driver-level H2SO4 source term at `driver.F90:1249`?
- Validation bar — ADR-003's 1e-6 against the new fixture, or ADR-015's 3 % if cloud-chem introduces its own ODE that diffrax handles differently than Fortran?

**Sub-PRs.** TBD; sub-PR breakdown lands in a future M8 scoping round per CLAUDE.md rule #3.

---

## Milestone 9 — Calibration / inverse demo (proposed)

**Status:** proposed 2026-05-28; owner priority slot 3 (after M8 + CI). Runs on the `diffrax` branch. GitHub: [milestone 4](https://github.com/reflective-org/MAM4-JAX/milestone/4).

**Why.** M6 PR-J5 audited the diffrax-branch as autodiff-clean end-to-end — `jax.grad` returns finite, deterministic cotangents through 60-step `scan`. M9 converts that audit into a working calibration demo, the first MAM4-JAX deliverable no implementation in this project's lineage has previously produced.

**Scope sketch.** Pick a sensitive physics parameter (candidates: binary nucleation pre-factor, soaexch uptake coefficient, coag pre-factor). Define a synthetic target trajectory from a known parameter value. Gradient-descend (`jax.grad` + `optax`) over a 24 h window to recover the parameter. Plot loss curve + recovered-vs-true parameter trajectory.

**Open questions.**
- Memory feasibility at 24 h trajectory lengths — see PR-J5 memory caveat in `docs/PROGRESS.md`. Likely requires switching `mam4_jax/solvers.py::solve_ivp` to `diffrax.ImplicitAdjoint()` or passing an explicit `RecursiveCheckpointAdjoint(checkpoints=N)` to bound working set. **Sub-question (owned by M9's first sub-PR):** should the adjoint strategy be configurable via `SolverConfig` (which would touch the strategy abstraction added in PR-I1, but lets forward / reverse runs share solver config), or hard-coded to `ImplicitAdjoint()` for calibration runs while keeping forward runs on the default? Probably configurable, but the abstraction churn warrants explicit owner approval before changing `solvers.py`.
- Which physics parameter is the most informative calibration target?
- Loss formulation: trajectory L2, terminal-state L2, or a physics-motivated functional (effective radius, AOD)?

**Sub-PRs.** TBD.

---

## Milestone 10 — NetCDF output emission from JAX (proposed)

**Status:** proposed 2026-05-28; owner priority slot 4. Runs on the `diffrax` branch (or could be ported to `main` if useful before merge-back). GitHub: [milestone 5](https://github.com/reflective-org/MAM4-JAX/milestone/5).

**Why.** The Fortran reference writes `mam_output.nc` (per `driver.F90`), which the postprocess notebook at `mam4-original-src-code/postprocess/postprocess.ipynb` consumes. To make JAX runs first-class citizens in the existing postprocess workflow, the JAX driver needs to emit a comparable NetCDF.

**Scope sketch.** Add a NetCDF writer hook to `run_timesteps` (or to a thin wrapper around it) that produces a file matching the Fortran `mam_output.nc` schema, or close enough that the postprocess notebook works against it.

**Open questions.**
- Match Fortran's NetCDF schema exactly, or define a clean JAX-native schema with an optional converter?
- Write per-step (like Fortran) or end-of-run only (simpler under JIT)?
- Where the write actually happens — inside the JIT'd loop (`jax.experimental.io_callback`) or out-of-loop after `block_until_ready`?

**Sub-PRs.** TBD.

---

## Milestone 11 — Backport ADR-014 + HANDWRITTEN_SOLVER_LIMITATIONS.md to main (proposed)

**Status:** proposed 2026-05-28; owner priority slot 5. Runs against `main`. GitHub: [milestone 6](https://github.com/reflective-org/MAM4-JAX/milestone/6).

**Why.** ADR-014 (dual-branch strategy + merge-back convention) and `docs/HANDWRITTEN_SOLVER_LIMITATIONS.md` landed via PR [#33](https://github.com/reflective-org/MAM4-JAX/pull/33) on the `dev`/`diffrax` lineage but never reached `main`. Per ADR-014's own "baseline changes flow `main → diffrax`" convention they sit on the wrong side of the divide. The merge-back is deferred indefinitely; backporting these docs separately closes the gap.

**Scope sketch.** Small docs-only PR opens against `main`. Cherry-pick or hand-port the relevant content from `dev`/`diffrax`. Add a "superseded on `diffrax-v0.1.0` by M7" pointer at the top of `HANDWRITTEN_SOLVER_LIMITATIONS.md` so a reader landing on the doc via `v0.1.0`'s release notes immediately knows the diffrax branch supersedes the limitations.

**Open questions.**
- Add an ADR addendum to ADR-013/014 noting the merge-back deferral, or leave that out of this PR and write it separately?
- Should ADR-016 (merge-back plan) backport too, given the merge-back is now deferred? Or does that ADR's content become stale on main once it gets there?

**Sub-PRs.** Single PR; no sub-PR breakdown anticipated.

---

## Milestone 12 — Multi-column / multi-level support (proposed)

**Status:** proposed 2026-05-28; owner priority slot 6. Runs on the `diffrax` branch. GitHub: [milestone 7](https://github.com/reflective-org/MAM4-JAX/milestone/7).

**Why.** The current port targets `ncol=1, pver=1` (matching the Fortran box-model build flags `-DPCOLS=1 -DPVER=1`). M6 PR-J3 verified `vmap`-cleanness across leading axes — the JAX side is structurally ready. What's missing is a multi-column Fortran reference for validation.

**Scope sketch.** Capture a multi-column Fortran reference (either by rebuilding with `PCOLS>1` if the box-model supports it, or by synthesizing one from multiple single-column runs). Add multi-column trajectory tests on diffrax. Likely interacts with M9 calibration (multi-column adds a richer parameter-sensitivity surface).

**Open questions.**
- Does the box-model Fortran build accept `PCOLS>1` / `PVER>1` without source changes? CPP flags suggest it might; depends on `box_model_utils/ppgrid.F90` shim's assumptions.
- Validation bar — ADR-003's 1e-6 across columns, or ADR-015's 3 % if diffrax substeps differently per column?

**Sub-PRs.** TBD.

---

## Milestone 13 — GPU/TPU sharding (proposed)

**Status:** proposed 2026-05-28; owner priority slot 7. Was PR-J6 in the original M6 plan, broken out into its own milestone because the work is substantially distinct from M6's JIT/scan/vmap/grad cleanup. Runs on the `diffrax` branch. GitHub: [milestone 8](https://github.com/reflective-org/MAM4-JAX/milestone/8).

**Why.** Single-host CPU has been the validation target throughout (`float64`, easiest Fortran diff). Multi-device execution opens up large-scale calibration (M9), regional / global multi-column runs (M12), and aerosol-data assimilation workflows.

**Scope sketch.** Set up sharding spec for the `(ncol, pver)` leading axes. Pick a target backend (GPU first or TPU first). Re-validate against Fortran under the new execution model.

**Open questions.**
- GPU (CUDA) or TPU first? Consumer NVIDIA GPUs lack dedicated hardware FP64 throughput (FP64 outside the FP64 tensor cores found only on A100 / H100 is emulated via paired-FP32 sequences and is substantially slower than FP32); A100 / H100 / TPU have full-throughput hardware FP64. Does the project allow a `float32` fallback on consumer GPUs, accepting a relaxed bar? Actual slowdown factor varies by op mix and should be benchmarked once the target hardware is picked.
- Sharding strategy: `shard_map`, `jax.device_put` with explicit sharding spec, or `pmap` (deprecated)?
- Memory: at scale, M9's calibration memory caveat reappears with a multi-device twist.

**Sub-PRs.** TBD.

---

## Smaller deferred items (tracked as GitHub Issues)

These don't warrant their own milestones but are tracked as GitHub Issues so a future PR can use the Development linkage to close them. See `docs/DEFERRED.md` for the detailed deferral context.

- **CI / GitHub Actions** — [#46](https://github.com/reflective-org/MAM4-JAX/issues/46). Owner-recommended mid-priority slotting (between M8 and M9).
- **License selection** — [#47](https://github.com/reflective-org/MAM4-JAX/issues/47). Blocks public announcement; admin / legal decision.
- **calcsize_sub stress test for dead branches** — [#48](https://github.com/reflective-org/MAM4-JAX/issues/48). Defensive; slot wherever convenient.
- **Strip `__pycache__/*.pyc` from vendored Fortran subtree** — [#49](https://github.com/reflective-org/MAM4-JAX/issues/49). Fold into next vendored-snapshot refresh.

---

*Whenever a milestone moves from "proposed" to "in progress", flesh out its subtasks here in the same PR.*
