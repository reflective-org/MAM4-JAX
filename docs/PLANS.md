# Plans

The forward-looking roadmap. Each milestone is broken into commit-sized subtasks. Status uses **proposed**, **in progress**, **done**, **deferred**. **Nothing should move from "proposed" to "in progress" without the owner's explicit approval** (rule #3).

When a milestone is in progress, its subtasks become the working task list. As subtasks complete they get a commit/PR link inline.

---

## Milestone 0 — Repo + documentation scaffold

**Status:** in progress.

- [x] Vendor Fortran reference, write `README.md` + initial `CLAUDE.md`. (`22f212d`)
- [x] Extract architecture and decisions out of `CLAUDE.md` into `docs/`; create `PROGRESS.md`, `PLANS.md`, `KEY_DECISIONS.md`, `DEFERRED.md`, `FEATURES.md`. *(current PR)*
- [ ] **Open with owner:** confirm Milestone 1 scope and order below.

---

## Milestone 1 — JAX package scaffolding (proposed, not started)

**Status:** proposed. Awaiting owner approval on the open architectural questions in `ARCHITECTURE.md` ("JAX port: proposed layout").

Proposed subtasks (each ≈ one commit):

- [ ] Decide tracer representation (flat array vs. structured pytree). → `KEY_DECISIONS.md` ADR-007.
- [ ] Decide process signature convention (pure-functional vs. delta-returning). → ADR-008.
- [ ] Decide configuration mechanism (dataclass / dict / YAML). → ADR-009.
- [ ] Create empty `mam4_jax/` package with `config.py` (enables `jax_enable_x64`), `data.py` (mode/species index tables, transcribed from `modal_aero_data.F90`), and empty stubs for each process module.
- [ ] Add `pyproject.toml`, dev dependencies (`jax`, `jaxlib`, `numpy`, `pytest`, `netCDF4`).
- [ ] Add `tests/` skeleton with one trivial sanity test that imports the package and asserts `float64` is enabled.

---

## Milestone 2 — Reference output capture (proposed)

**Status:** proposed.

- [ ] Get the Fortran reference building locally (verify `NETCDF_LIB` / `NETCDF_INCLUDE` env, fix the hard-coded `outpath` in `run_test.csh`).
- [ ] Run the existing 12-point timestep sweep; archive `mam_output.nc` outputs under `tests/reference/`.
- [ ] Instrument selected Fortran subroutines to dump per-process inputs + outputs (likely `modal_aero_calcsize` and `modal_aero_wateruptake` first). → ADR-010 on instrumentation approach.
- [ ] Capture reference data for the first process to be ported.

---

## Milestone 3 — First process port (proposed)

**Status:** proposed. Owner to pick which process to port first. Candidates (in suggested order of increasing complexity):

1. **`wv_saturation`** — pure thermodynamics, no aerosol state. Smallest scope, but exercises `float64` precision and closed-form porting style.
2. **`modal_aero_wateruptake`** — equilibrium, no time integration. Single-mode physics, well-suited for diff-vs-Fortran.
3. **`modal_aero_calcsize`** — size redistribution, more state coupling.
4. **`modal_aero_newnuc`** — nucleation (Vehkamäki).
5. **`modal_aero_coag`** — coagulation kernels.
6. **`modal_aero_gasaerexch`** — condensation.
7. **`modal_aero_rename`** — Aitken→accumulation transfer.

Each first-port follows the validation workflow in `CLAUDE.md` (capture reference, port, diff to `1e-6`, plot residuals, log in `PROGRESS.md`).

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
