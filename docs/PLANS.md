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
   - 3c. [ ] `modal_aero_wateruptake_sub` + `_dr` (`:130-485`) — driver + per-column workhorse, validated end-to-end against `tests/reference/per_process/wateruptake_{before,after}.npz`.
4. [ ] **`modal_aero_calcsize_sub`** (`modal_aero_calcsize.F90`) — size redistribution. Heaviest of the per-process ports (~1500 lines). Reference data: `tests/reference/per_process/calcsize_{before,after}.npz`.
5. [ ] **`modal_aero_newnuc`** — binary H2SO4–H2O nucleation (Vehkamäki).
6. [ ] **`modal_aero_coag`** — Brownian coagulation kernels.
7. [ ] **`modal_aero_gasaerexch`** — condensation onto modes.
8. [ ] **`modal_aero_rename`** — Aitken → accumulation transfer.

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
