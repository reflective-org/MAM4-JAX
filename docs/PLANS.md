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

## Milestone 2 — Reference output capture (proposed)

**Status:** proposed.

- [ ] Get the Fortran reference building locally (verify `NETCDF_LIB` / `NETCDF_INCLUDE` env, fix the hard-coded `outpath` in `run_test.csh`). Document in `docs/REFERENCE_BUILD.md`.
- [ ] Run the existing 12-point timestep sweep; archive NetCDF outputs under `tests/reference/sweep/`.
- [ ] Add patch-overlay instrumentation (new ADR) hooking `driver.F90:1118`, `:1208`, `:1283` to dump per-process I/O without modifying the vendored Fortran tree.
- [ ] `scripts/capture_reference.py` driving the build + instrumented run + `.npz` dump under `tests/reference/per_process/`.
- [ ] `tests/reference/SCHEMA.md` documenting the capture artifact contract.

---

## Milestone 3 — First process ports (proposed)

**Status:** proposed. Recommended ordering (from `docs/plans/001` plan recommendation):

1. **`polysvp`** (within `wv_saturation.F90:699-736`) — pure scalar Goff-Gratch saturation vapor pressure. ~40 lines, no module state, no aerosol coupling. Exercises the whole validation pipeline (capture, JAX port, `float64`, 1e-6 diff, residual plot) at minimum complexity. Reference data: tabulated `polysvp(T, type)` for a sweep of T values, generated in Python (no need to instrument Fortran).
2. **Other `wv_saturation` leaf functions** as needed (e.g., `qsat_water`, `qsat_ice`) — same shape as `polysvp`, may share the saturation-vapor-pressure helpers.
3. **`modal_aero_wateruptake_dr`** (`modal_aero_wateruptake.F90:130-150`) — equilibrium water uptake. First aerosol-state-aware port. Reference data: instrumented dump at `driver.F90:1208`.
4. **`modal_aero_calcsize_sub`** (`modal_aero_calcsize.F90`) — size redistribution. Heaviest of the per-process ports (~1500 lines). Reference data: instrumented dump at `driver.F90:1118`.
5. **`modal_aero_newnuc`** — binary H2SO4–H2O nucleation (Vehkamäki).
6. **`modal_aero_coag`** — Brownian coagulation kernels.
7. **`modal_aero_gasaerexch`** — condensation onto modes.
8. **`modal_aero_rename`** — Aitken → accumulation transfer.

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
