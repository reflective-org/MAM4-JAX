# Progress

A running, append-only log of project milestones. Most-recent entry on top. Update in the same PR that lands the work being recorded.

Each entry: date, short title, links to commits / PRs, one-paragraph summary.

---

## 2026-05-19 — Milestone 3.5 (PR-B) — Calcsize Aitken ↔ accumulation transfer

- PR: pending (`m3/calcsize-aitacc-transfer`)
- Completes `modal_aero_calcsize_sub`. Adds the Aitken ↔ accumulation mode-transfer block (Fortran lines 944–1294) to `mam4_jax/processes/calcsize.py`. The function now matches the canonical Fortran box-model call (`do_aitacc_transfer_in=.true.`).
- **Transfer-pair tables** computed at module-import in `mam4_jax/data.py`:
  - `AITKEN_MODE_IDX`, `ACCUM_MODE_IDX` (0-based mode indices).
  - `LSPECFRMA_CSIZXF` / `LSPECTOOA_CSIZXF` (interstitial) and the cloud-borne counterparts — 5 species pairs (1 number + 4 mass: sulfate, s-organic, seasalt, m-organic) matched between Aitken and accum by `lspectype_amode`.
  - `NOXF_ACC2AIT`: mask of accum slots whose species isn't in Aitken (p-organic, black-c, dust).
  - `V2NZZ_AIT_ACC`: geometric-mean v2n threshold (= √(voltonumb_aitken · voltonumb_accum)).
- **New helpers** in `mam4_jax/processes/calcsize.py`:
  - `_xferfrac_pair(num_t, drv_t, v2n_target, v2nzz, direction)`: computes (xferfrac_num, xferfrac_vol, triggered_mask) for one direction (ait→acc or acc→ait), faithfully mirroring the Fortran's full-transfer-vs-fractional and clamp logic.
  - `_apply_aitacc_transfer(...)`: full transfer-block implementation. Vectorized per (col, level); pair-list loop is Python-level (5 iterations).
- **`calcsize` now takes** `do_aitacc_transfer: bool = True` keyword. `False` matches the `per_process_no_aitacc/` reference (PR-A's path); `True` matches the canonical `per_process/` reference (this PR's path).
- **`tests/reference/per_process/` refreshed** from nstep=1 to nstep=60 (matches `per_process_no_aitacc/`). The wateruptake test (uses `[0]` snapshot) still passes unchanged.
- **Validation**:
  - Updated `tests/test_calcsize.py` to call with `do_aitacc_transfer=False` explicitly (matches no-aitacc reference fixture name).
  - New `tests/test_calcsize_transfer.py` (4 tests) validates `do_aitacc_transfer=True` against the full-transfer reference. dgncur_a rel-err 2.12e-16, q rel-err < ADR-003 (with `np.allclose(atol=1e-25, rtol=1e-6)` to absorb a ~1e-26 machine-noise artifact at the exactly-zero m-organic mass index), qqcw bit-exact zero.
  - **Structural test**: `do_aitacc_transfer=True` ≡ `do_aitacc_transfer=False` on the box-model fixture — confirms transfer is a no-op here.
- Full suite: **43/43 green** (was 39).
- **`modal_aero_calcsize_sub` is now fully ported.** The transfer block code is faithful but exercised "in spirit only" by the current test (the transfer never triggers in the canonical reference, see `docs/DEFERRED.md`).

## 2026-05-19 — Milestone 3.5 (PR-A) — Calcsize per-mode adjustment + M2 extension

- PR: pending (`m3/calcsize-per-mode-adjust`)
- Two-PR bottom-up plan for `modal_aero_calcsize_sub`; this PR-A covers the per-mode number-bounds adjustment and the dgncur_a recomputation. PR-B will add the Aitken ↔ accum mode-transfer block.
- **M2 extension** (rule #5 — every change supports its tests):
  - New `scripts/patches/disable_aitacc_transfer.patch` (one-line overlay flipping `do_aitacc_transfer_in=.true.` → `.false.` in driver.F90's calcsize call). Cleanly applies on top of `driver_instrumentation.patch`.
  - `build_reference.sh --no-aitacc-transfer` applies the overlay (requires `--instrumented`).
  - `capture_reference.py --mode instrumented-no-aitacc` writes to `tests/reference/per_process_no_aitacc/` (separate from the default `per_process/` so the two captures coexist). Default nstep=60 because calcsize is essentially trivial at nstep=1.
- **JAX port** in `mam4_jax/processes/calcsize.py` (replaces the M1 stub): vectorized per-mode adjustment with the full 3-step bounds procedure (Fortran lines 812–869) covering all four branches (drv_a/c zero vs positive). Helpers `_gather_per_slot`, `_adjusted_num_*`, `_compute_dgn_v2n`. Skips Aitken-accum transfer (PR-B); equivalent to Fortran `do_aitacc_transfer_in=.false.`.
- New constants in `mam4_jax/data.py`: `DGNUM_AMODE`, `DGNUMLO_AMODE`, `DGNUMHI_AMODE`, derived `ALNSG_AMODE`, `DUMFAC_AMODE`, `VOLTONUMB_AMODE`/`VOLTONUMBLO_AMODE`/`VOLTONUMBHI_AMODE` — all from `rad_constituents.F90:167-170` and `modal_aero_initialize_data.F90:428-435`.
- Validation (`tests/test_calcsize.py`, 4 tests): batched across all 60 timesteps. Max relative error in `dgncur_a` evolution = **2.12e-16** — bit-exact at machine ε across all 240 (60 × 4) data points. Number tracers (which never adjust in the box-model setup) pass through unchanged at machine ε.
- `tests/test_scaffolding.py`: dropped `calcsize` from the `PROCESS_MODULES` stub-raises list.
- Residual figure: `docs/figures/calcsize_residuals.png` (top: dgncur_a evolution per mode JAX vs Fortran; bottom: per-(timestep, mode) rel-err).
- Full suite: **39/39 green** (was 36).
- Documentation: `docs/DEFERRED.md` got a new entry calling out that the bounds-adjust + Aitken-accum-transfer branches are dead in the captured reference; `tests/reference/SCHEMA.md` mirrors the note.

## 2026-05-19 — Milestone 3.4 (PR-C) — Wateruptake driver + completion of M3.4

- PR: pending (`m3/wateruptake-driver`)
- Final piece of the wateruptake bottom-up chain. Replaces the M1 `NotImplementedError` stub at `mam4_jax/processes/wateruptake.py` with the full port of `modal_aero_wateruptake_dr` + `modal_aero_wateruptake_sub` (~250 lines vectorized).
- Added per-species and per-mode property tables to `mam4_jax/data.py`:
  - `SPECDENS_AMODE`, `SPECHYGRO_AMODE` (9 species types, from `rad_constituents.F90:96-103`).
  - `SIGMAG_AMODE`, `RHCRYSTAL_AMODE`, `RHDELIQUES_AMODE` (4 modes).
  - Pre-computed `PER_SLOT_DENSITY` / `PER_SLOT_HYGRO` (4 × 14) lookup tables and a `SLOT_VALID` mask for vectorized per-(mode, slot) gather.
  - `RHOH2O = 1000 kg/m³` added to `mam4_jax/constants.py`.
- `wateruptake(state, params, config)` (ADR-009 signature) takes a state dict with `q`, `dgncur_a`, `t`, `pmid`, `cldn` and returns a new state with `dgncur_awet`, `qaerwat`, `wetdens` updated. Internally: gather per-mode dry mass / volume / hygroscopicity using `INDEX_TABLES`, compute v2ncur_a / naer / dryrad / drymass per mode, compute RH from `qsat_water(t, pmid)` and the clear-sky cloud adjustment, call `modal_aero_kohler` per (column, level, mode), apply the deliquescence/crystallization hysteresis branches.
- Validation (`tests/test_wateruptake.py`, 4 tests): end-to-end against `tests/reference/per_process/wateruptake_{before,after}.npz`. Box-model meteorology (`t=273`, `pmid=1e5`, `cldn=0`) is pinned by the namelist + `driver.F90:591` so the test doesn't need additional instrumentation. Measured relative errors:
  - `dgncur_awet`: max 4.53e-16 (machine ε)
  - `qaerwat`: max 1.86e-7 — *but* at the 10⁻²⁰ absolute scale (primary-carbon mode where rwet ≈ rdry and qaerwat is essentially numerical noise). All other modes match at machine ε.
  - `wetdens`: max 2.07e-16 (machine ε)
- Test cleanup: `wateruptake` removed from the `PROCESS_MODULES` stub-raises tuple in `tests/test_scaffolding.py` — it's a real implementation now.
- Residual figure: `docs/figures/wateruptake_residuals.png` (4-panel: dry vs wet diameters, aerosol water content, wet density, per-(mode, var) rel-err).
- Full suite: **36/36 green** (was 33).

## 2026-05-19 — Milestone 3.4 (PR-B) — Port `modal_aero_kohler`

- PR: pending (`m3/kohler-solver`)
- Second bottom-up step of the wateruptake chain: the Köhler-equilibrium wet-radius solver itself, consuming the `makoh_cubic` / `makoh_quartic` polynomial root finders that landed in PR-A.
- Renamed `scripts/patches/expose_makoh.patch` → `scripts/patches/expose_internals.patch` and extended it to also expose `modal_aero_kohler` (single consolidated patch is cleaner than two competing ones touching the same source region).
- `scripts/reference_drivers/kohler_driver.F90`: sweeps a `(rdry, hygro, s)` grid of 7 × 4 × 6 = 168 points designed to exercise all four branches of the solver — insoluble particle (vol ≤ 1e-12 microns³), small-p approximation, generic quartic, near-saturation interpolation. `build_reference.sh --kohler` and `capture_reference.py --mode kohler` produce `tests/reference/kohler/reference.npz` (~6 KB).
- `mam4_jax/kohler.py`: added `modal_aero_kohler(rdry_in, hygro, s)` plus an internal `_pick_smallest_valid_real_root` helper. Vectorised over the batch axis; both polynomial families are solved unconditionally then masked to the appropriate branch via `jnp.where`. Skips the `verify_wateruptake` bisection branch (macro is off in the reference build).
- Constants embedded as literals (Fortran lines 533-539): `mw=18`, `surften=76`, `ugascon=8.3e7`, `tair=273`, `rhow=1` — these are the in-routine values the Fortran uses (the physically-derived alternatives are commented out at lines 525-531).
- Validation (`tests/test_kohler.py`, 4 tests): max relative error against Fortran is **9.77e-14** across all 168 grid points — 8 orders below ADR-003's tolerance. The worst-case is at small rdry near saturation, where root selection is fiddly.
- Residual figure: `docs/figures/kohler_residuals.png` shows Köhler growth-factor curves per hygroscopicity panel (JAX dashed over Fortran solid) plus a per-point rel-err panel.
- Full suite: **33/33 green** (was 29).

## 2026-05-19 — Milestone 3.4 (PR-A) — Port `makoh_cubic` and `makoh_quartic`

- PR: pending (`m3/makoh-polynomial-solvers`)
- First bottom-up step of the wateruptake port chain: the two analytical polynomial root finders that the Köhler solver consumes.
- `scripts/patches/expose_makoh.patch`: small overlay that adds `public :: makoh_cubic, makoh_quartic` to `modal_aero_wateruptake.F90` (the routines are otherwise private). Applied by `build_reference.sh --makoh` onto the transient build copy; vendored tree pristine.
- `scripts/reference_drivers/makoh_driver.F90`: standalone harness that feeds the makoh routines six representative cubic and six representative quartic test cases (well-conditioned plus the "insoluble particle" edge), writes complex roots to text. `scripts/capture_reference.py --mode makoh` parses to `tests/reference/makoh/reference.npz` (~2 KB).
- `mam4_jax/kohler.py` (new module): `makoh_cubic(p0, p1, p2)` and `makoh_quartic(p0, p1, p2, p3)` returning `complex128` roots. Line-by-line port of `modal_aero_wateruptake.F90:684-793`. NaN propagation faithfully matches Fortran (no `safe_cy` guards) so the algorithm's degenerate cases produce the same NaN they do in the reference. Naming rationale: this module will grow with the kohler solver in PR-B; the process-level entry point (the M1 stub at `mam4_jax/processes/wateruptake.py`) gets filled in by PR-C and will call into this module.
- Documented Fortran quirk preserved: `makoh_cubic` accepts `p2` but ignores it (Cardano's method on the depressed cubic). The JAX port exposes `p2` for signature parity with `del p2` and a docstring note.
- Validation (`tests/test_makoh.py`, 4 tests): max relative error **1.49e-14 (cubic)** and **3.47e-15 (quartic)** across all 6 + 6 test cases. Both ~8 orders below ADR-003's 1e-6 tolerance.
- Residual figure: `docs/figures/makoh_residuals.png` (4 panels — absolute and relative error per case for each root branch of cubic + quartic).
- Full suite: **29/29 green** (was 25).

## 2026-05-19 — Milestone 3.3 — Populate `IndexTables` from instrumented Fortran capture

- PR: pending (`m3/populate-index-tables`)
- Extended `scripts/patches/mam4_dump_state.F90` with a `dump_indices()` subroutine that writes `modal_aero_data`'s integer index tables (`numptr_amode`, `numptrcw_amode`, `lspectype_amode`, `lmassptr_amode`, `lmassptrcw_amode`, `nspec_amode`, `modename_amode`, `specname_amode`) to `mam4_indices.txt` once at init, right before `cambox_do_run`'s `main_time_loop`. The unified-diff patch (`driver_instrumentation.patch`) gains the corresponding `call dump_indices()` line via the existing `_generate_driver_patch.py` regenerator.
- `scripts/capture_reference.py --mode instrumented` now also parses `mam4_indices.txt` and writes `tests/reference/indices/reference.npz` (~4 KB, 11 arrays + 3 scalar dims, all 0-based with `-1` sentinels for unused slots).
- `mam4_jax/data.py`: replaced sentinel-filled `IndexTables` with hard-coded MAM4-MOM constants (`NUMPTR_AMODE`, `LMASSPTR_AMODE`, `LMASSPTRCW_AMODE`, `LSPECTYPE_AMODE` — all 0-based) and a module-level `INDEX_TABLES` instance. Accessors `get_number`, `get_mass`, and new `get_mass_by_species_name` now return actual `pcnst`-axis slices instead of raising. `make_sentinel_tables()` kept for tests of the sentinel-raise path.
- Reference-axis ordering: Python uses `(mode, slot)`. Fortran is `(slot, mode)` (column-major); the parser swaps. Documented in `tests/reference/SCHEMA.md`.
- Tests: scaffolding suite grew from 12 to 18 (+`test_index_tables_populated`, `test_index_tables_match_npz_reference`, `test_get_number_returns_slice`, `test_get_mass_returns_slice`, `test_get_mass_raises_on_unused_slot`, `test_get_mass_by_species_name`). Full suite: **25/25 green**.
- The `.npz` is committed as provenance; the Python constants are the source of truth. `tests/test_scaffolding.py::test_index_tables_match_npz_reference` fails loudly if they ever drift.

## 2026-05-18 — Milestone 3.2 — Ports: `qsat_water` and `qsat_ice` + physical constants

- PR: pending (`m3/qsat-functions`)
- Added `mam4_jax/constants.py` with the canonical physical constants (BOLTZ, AVOGAD, RGAS, MWDAIR, MWWV, LATICE, LATVAP, derived RDAIR/RH2O/EPSQS, plus `wv_saturation`-name aliases HLATV/HLATF/RGASV/EPSQS). Values transcribed verbatim from `mam4-original-src-code/e3sm_src/shr_const_mod.F90:33-61` so the JAX port uses the same numbers the Fortran sets through `gestbl()`.
- Built a reference driver (`scripts/reference_drivers/qsat_driver.F90`) that calls `gestbl` with box-model constants then sweeps `qsat_water` (Goff–Gratch via inline polysvp formula) and `qsat_ice` (Clausius–Clapeyron with combined latent heat of sublimation) over a 301-T × 5-p grid. New `--qsat` flag in `build_reference.sh`, `--mode qsat` in `capture_reference.py`. Output: `tests/reference/qsat/reference.npz` (~48 KB).
- Ported `qsat_water(T, p)` and `qsat_ice(T, p)` to `mam4_jax/saturation.py`, plus a `qs_from_es(es, p)` helper that captures the shared `qs = epsqs · es / (p − (1 − epsqs) · es)` formula and the Fortran's `qs < 0 → qs = 1` clamp. **Preserved the Fortran inconsistency**: `qsat_ice` uses Clausius–Clapeyron, not `polysvp_ice`. Documented in the saturation module docstring; callers wanting consistency can `qs_from_es(polysvp_ice(T), p)`.
- Validation (`tests/test_qsat.py`): max relative error against Fortran is **9.36e-14 (water)** and **7.81e-15 (ice)**. Both ~8+ orders below ADR-003's 1e-6 tolerance. Test suite total: 19/19 green.
- Residual figure: `docs/figures/qsat_residuals.png` (four panels — qs(T) per pressure level for water + ice, with rel-err vs T below).

## 2026-05-18 — Milestone 3.1 — First port: `polysvp` (saturation vapor pressure)

- PR: pending (`m3/polysvp-port`)
- Built a standalone Fortran reference driver (`scripts/reference_drivers/polysvp_driver.F90`) that calls `wv_saturation::polysvp` over a 170 K – 320 K sweep (1501 points, 0.1 K resolution). Linked against the existing baseline build's object files. `scripts/build_reference.sh --polysvp` produces `run/polysvp_driver.exe`; `scripts/capture_reference.py --mode polysvp` runs it and archives `tests/reference/polysvp/reference.npz` (~36 KB, arrays `T`, `esat_water`, `esat_ice`).
- Ported `polysvp` to `mam4_jax/saturation.py` as `polysvp_water(T)` and `polysvp_ice(T)` (plus a Fortran-parity `polysvp(T, type)` dispatcher). Direct line-by-line port of the Goff–Gratch polynomial — each Python line traces 1:1 to the Fortran source.
- Validation (`tests/test_polysvp.py`): max relative error against the Fortran reference is **4.31e-15 (water)** and **4.14e-15 (ice)** across 1501 points — eleven orders of magnitude below ADR-003's 1e-6 tolerance, essentially bit-equivalent in `float64`.
- Residual figure: `docs/figures/polysvp_residuals.png`, generated by `scripts/plot_polysvp_residuals.py`. Top panel overlays JAX and Fortran on log axes; bottom panel shows rel-err vs T with the 1e-6 tolerance line and the float64 ε floor.

## 2026-05-18 — Milestone 2 — Fortran reference output capture

- PR: pending (`m2/reference-capture`)
- Built the vendored MAM4 Fortran box model end-to-end via `scripts/build_reference.sh` (auto-detects `gfortran` + NetCDF via `nf-config`/`nc-config`; adds `-fallow-invalid-boz` for modern gfortran and two `-L` paths for Homebrew's split NetCDF prefixes). Vendored tree stays pristine; build artifacts live in gitignored `mam4-original-src-code/{build,run}/`.
- Captured the canonical 12-point convergence sweep (`1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800` substeps over 1800 s) into `tests/reference/sweep/*.nc` (12 NetCDF files, ~1.7 MB total). Discovered and worked around the upstream `run_test.csh`'s broken sweep loop and hard-coded outpath by reimplementing the sweep in `scripts/capture_reference.py`.
- Added the patch-overlay instrumentation (ADR-012): `scripts/patches/mam4_dump_state.F90` is a small Fortran helper module that writes binary state snapshots; `scripts/patches/driver_instrumentation.patch` inserts six `call dump_snapshot(...)` hooks around `calcsize`, `wateruptake`, and `amicphys` inside `cambox_do_run`. The build script applies both onto a transient copy of `driver.F90` and overrides `OBJ9` so the helper compiles before `driver.o`.
- `scripts/capture_reference.py --mode instrumented` rebuilds with the overlay, runs a single configurable-`nstep` integration, parses the six `mam4_dump_*.bin` files, and writes them as `tests/reference/per_process/*.npz` with a documented array contract.
- Authored `docs/REFERENCE_BUILD.md` (prereqs, build flag rationale, what the scripts do, missing-from-upstream `&size_parameters` namelist group, why the upstream `run_test.csh` is replaced) and `tests/reference/SCHEMA.md` (artifact layout for both sweep and per-process outputs, array shapes/dtypes, VMR-conversion caveat for `amicphys`).
- `git diff mam4-original-src-code/` is empty before, during, and after a build — the vendored tree contract from ADR-001 holds.

## 2026-05-18 — Milestone 1 — JAX package scaffold

- PR: pending (`m1/scaffold-jax-package`)
- Added top-level `mam4_jax/` package: `__init__.py` enables `jax_enable_x64`; `config.py` defines four frozen dataclasses (`TimeConfig`, `ControlConfig`, `MetConfig`, `ChemConfig`) mirroring the Fortran namelist groups plus a `RunConfig` composite and YAML loader; `data.py` transcribes MAM4-MOM compile-time constants (PCNST=35, NTOT_AMODE=4, NTOT_ASPECTYPE=9, NSPEC_AMODE=(7,4,7,3), mode + species names) and exposes a sentinel-filled `IndexTables` with `get_number`/`get_mass` accessors that raise until M2 populates real indices.
- Added `mam4_jax/processes/` with seven `NotImplementedError`-raising stubs (`calcsize`, `wateruptake`, `gasaerexch`, `newnuc`, `coag`, `rename`, `amicphys`) using the ADR-009 pure-functional signature.
- Added `tests/test_scaffolding.py` (12 assertions; all pass against `jax 0.9.2` / `pytest 9.0.2`).
- Recorded ADR-008 (tracer rep), ADR-009 (pure-functional signatures), ADR-010 (dataclass+YAML config), ADR-011 (all-changes-via-PR, supersedes ADR-006). The technical ADRs were pre-approved in `docs/plans/001` under the numbering 007–009; the +1 shift is documented in the archived plan.

## 2026-05-18 — Plans archive convention + first plan archived

- PR: [#1](https://github.com/reflective-org/MAM4-JAX/pull/1) (merged at [`e643c20`](https://github.com/reflective-org/MAM4-JAX/commit/e643c20); content commit [`cce06f6`](https://github.com/reflective-org/MAM4-JAX/commit/cce06f6))
- Established the convention to archive approved plans under `docs/plans/NNN-<slug>.md` (ADR-007).
- Archived the first plan as `docs/plans/001-scaffold-and-reference-capture.md`, which covers Milestones 1 (JAX package scaffold) and 2 (Fortran reference output capture) and recommends `polysvp` as the M3 first-port warm-up.

## 2026-05-18 — Documentation scaffold

- Commit: [`a82e42d`](https://github.com/reflective-org/MAM4-JAX/commit/a82e42d)
- Added `docs/` with `ARCHITECTURE.md`, `PROGRESS.md`, `PLANS.md`, `KEY_DECISIONS.md`, `DEFERRED.md`, `FEATURES.md`.
- Extracted the MAM4 architecture section and embedded design decisions out of `CLAUDE.md` into `docs/ARCHITECTURE.md` and `docs/KEY_DECISIONS.md` (ADR-001 through ADR-006). `CLAUDE.md` now holds rules, guardrails, validation workflow, and pointers into the deeper docs.

## 2026-05-18 — Initial repo setup and Fortran reference vendoring

- Commit: [`22f212d`](https://github.com/reflective-org/MAM4-JAX/commit/22f212d)
- Created the MAM4-JAX repository at `reflective-org/MAM4-JAX`. Vendored the MAM4 Fortran box model as a frozen snapshot under `mam4-original-src-code/`, sourced from `reflective-org/MAM4_box_model@4150e2d` (2025-12-10). Authored initial `README.md`, `CLAUDE.md` (rules, architecture overview, behavioral guardrails). Nested `.git/` in the vendored subtree was removed so files are tracked normally; provenance is recorded in `README.md`. No JAX code yet.

---

*Future entries should follow the same format: date, title, commit/PR link, summary. Keep entries terse — link to the docs they update rather than restating the change.*
