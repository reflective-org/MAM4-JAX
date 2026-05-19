# Progress

A running, append-only log of project milestones. Most-recent entry on top. Update in the same PR that lands the work being recorded.

Each entry: date, short title, links to commits / PRs, one-paragraph summary.

---

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
