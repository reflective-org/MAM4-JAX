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
5. **Amicphys** — the remaining microphysics. **Scope correction (2026-05-19):** the standalone modules `modal_aero_newnuc.F90`, `modal_aero_coag.F90`, `modal_aero_gasaerexch.F90`, and `modal_aero_rename.F90` are **not invoked** by the box-model driver — `modal_aero_amicphys_intr` (in `e3sm_src_modified/modal_aero_amicphys.F90:310`) contains its own self-contained orchestration and four sub-routines (`mam_gasaerexch_1subarea`, `mam_rename_1subarea`, `mam_newnuc_1subarea`, `mam_coag_1subarea`). See `docs/ARCHITECTURE.md` for the full module map. The M3 amicphys port targets those internal sub-routines, not the standalone files. Multi-PR plan:
   - 5a. [x] **Orchestration shell**: ported in `mam4_jax/processes/amicphys.py` (replaces M1 stub). Implements `_mam_amicphys_1gridcell` → `_mam_amicphys_1subarea_clear` → four sub-process stubs in the Fortran order (gasaerexch → rename → newnuc → coag). Cloudy path not implemented (unreachable for `cldn=0`). Validation: bit-exact passthrough vs `tests/reference/per_process_amicphys_off/amicphys_{before,after}.npz` (captured with all `mdo_*=0`). Capture via `scripts/capture_reference.py --mode instrumented-amicphys-off`.
   - 5b. [x] `mam_rename_1subarea` (~323 LOC) — Aitken → accum mode-transfer. Ported in `mam4_jax/processes/amicphys.py` against the amicphys-local view (`qnum_cur`, `qaer_cur`, `qaer_delsub_grow4rnam`, `qwtr_cur`, `fac_m2v_aer`). Validated against `tests/reference/per_process/rename_{before,after}.npz` captured via the new `scripts/patches/rename_hook.patch` overlay. Max rel-err: qnum 2.5e-9, qaer 7.0e-10 across all 60 timesteps. The orchestration shell's call to rename is deferred to PR-C — wiring requires the state-dict ↔ amicphys-local-view unpacking that lands alongside `_mam_gasaerexch_1subarea`. Plot: `docs/figures/rename_residuals.png`. Plan: `docs/plans/002-rename-port.md`.
   - 5c. [x] **Foundation + wire rename**: state-dict ↔ amicphys-local-view unpacking layer (`_unpack_state_to_amicphys_view`, `_repack_amicphys_view_to_state` in `mam4_jax/processes/amicphys.py`) using a two-stage conversion (driver-side mmr→vmr via `MWDRY/ADV_MASS` + amicphys-internal vmr→local via `FCVT_*`). Wires `_mam_rename_1subarea` into the orchestration shell. Validated via new single-toggle Fortran capture `tests/reference/per_process_rename_only/` and the new test `test_orchestration_rename_only_matches_fortran` (rel-err < 1e-12 across 60 steps). Empirical finding: with gasaerexch off, the Fortran rename's optaa=40 guard trips and rename is a no-op every step, so the orchestration test is a full unpack/repack passthrough check. Plan: `docs/plans/003-foundation-rename-wiring.md`. **Scope expansion (2026-05-20):** reading `mam_gasaerexch_1subarea`'s source revealed it calls `mam_soaexch_1subarea` (~330 LOC) plus `gas_aer_uptkrates_1box1gas` (~148 LOC), so the original 4-PR remainder is now a 5-PR remainder.
   - 5d. [ ] `mam_gasaerexch_1subarea` proper (~306 LOC) — H₂SO₄ analytical solver + uptake helpers. SOA-exchange skipped (separate PR-E).
   - 5e. [ ] `mam_soaexch_1subarea` (~330 LOC) — secondary-organic-aerosol condensation/evaporation (called from gasaerexch's body).
   - 5f. [ ] `mam_newnuc_1subarea` (~415 LOC) — binary H₂SO₄–H₂O nucleation.
   - 5g. [ ] `mam_coag_1subarea` (~437 LOC) — Brownian coagulation kernels.

Each sub-routine port (5d–5g) needs a single-toggle capture (e.g., `mdo_gasaerexch=1, others=0`) so its effect can be isolated from the others. Final validation reuses the existing `tests/reference/per_process/amicphys_{before,after}.npz` (full-bundle, 60-step) once all sub-processes are in place.

Each port lands as its own PR following the validation workflow in `CLAUDE.md` (capture reference, port, diff to `1e-6`, plot residuals, log in `PROGRESS.md`).

---

## Milestone 4 — Operator-splitting time loop (proposed)

**Status:** proposed. Requires Milestones 1–3 complete for at least the processes the loop calls. Initial implementation is a Python `for` loop (rule #8 phase A); `jax.lax.scan` is deferred to Milestone 6.

---

## Milestone 5 — Convergence test reproduction (proposed)

**Status:** proposed. Reproduce the 12-point timestep sweep from `run_test.csh` (`1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800` substeps over 1800 s) and match Fortran outputs to `1e-6` at every step count. Generates a convergence-plot deliverable.

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

*Whenever a milestone moves from "proposed" to "in progress", flesh out its subtasks here in the same PR.*
