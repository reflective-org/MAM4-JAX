# Plan 001 — Scaffold JAX package and capture Fortran reference

> **Status:** approved 2026-05-18.
>
> **Editorial note:** This plan was approved during an interactive planning session and pre-reserved ADR numbers 007–010 for its technical decisions. After approval, the project added one earlier ADR (ADR-007 — "store plans under `docs/plans/`"). The technical ADRs in this plan therefore land in `docs/KEY_DECISIONS.md` as **ADR-008** (tracer representation), **ADR-009** (pure-functional signatures), **ADR-010** (dataclass+YAML config), and **ADR-011** (Fortran instrumentation overlay). The body below preserves the originally-approved numbering — when reading, mentally add +1 to any ADR reference inside the "Recommended ADR resolutions" and "Milestone" sections. Future plans should not need this kind of note; ADRs will be allocated at write-time.

---

## Context

We are porting the MAM4 aerosol-microphysics box model from Fortran 90 to JAX (`reflective-org/MAM4-JAX`). The Fortran reference is vendored at `mam4-original-src-code/` (frozen at `4150e2d`). Project rules and decisions are in `CLAUDE.md` and `docs/KEY_DECISIONS.md`; the high-level roadmap is in `docs/PLANS.md`.

Current state: docs scaffolded; **no JAX code exists**. Three architectural ADRs (007 tracer representation, 008 signature convention, 009 config mechanism) are blocking Milestone 1. The first-port choice is needed to focus the Milestone 2 reference-capture work.

This plan covers **Milestone 1 (JAX package scaffold)** and **Milestone 2 (Fortran reference output capture)** in execution-ready detail, plus a forward-looking recommendation for the first process to port (M3, not detailed here).

**Process convention (new):** every plan we produce, including this one, is copied to `docs/plans/NNN-<slug>.md` so the planning history lives alongside the project docs. This plan's copy will be `docs/plans/001-scaffold-and-reference-capture.md`.

## Recommended ADR resolutions (to land in M1)

### ADR-007 — Tracer representation: flat `pcnst` array, with named (mode, species) accessors

- **Primary state:** a single JAX array `q` of shape `(pcols, pver, pcnst)` mirroring the Fortran `q(:,:,pcnst)`. Cloud-borne tracers `qqcw` mirror this shape. Reason: enables byte-for-byte diff against the Fortran reference with no index translation, which is critical for the 1e-6 validation target (ADR-003).
- **Accessor layer:** `mam4_jax/data.py` exposes index tables `numptr_amode[ntot_amode]`, `lmassptr_amode[maxd_aspectype, ntot_amode]`, `nspec_amode[ntot_amode]`, etc. (transcribed from `mam4-original-src-code/e3sm_src/modal_aero_data.F90:180-185`, populated in `modal_aero_initialize_data.F90:250-309`). Helpers like `get_number(q, mode)` and `get_mass(q, mode, species_slot)` return views; physics modules use these rather than raw indices.
- **Rejected:** a nested per-(mode, species) pytree — would force per-element diffs and obscure the Fortran correspondence during validation.

### ADR-008 — Process signature convention: pure-functional

- Every process function has the shape `process_fn(state, params, config) -> new_state` where `state` is a pytree and `new_state` is a new pytree. No in-place mutation, no pointer-output.
- This is the **opposite** of the Fortran convention (`modal_aero_wateruptake.F90:130-150` uses pointer outputs and side-effects), so JAX code will look structurally different from the Fortran. That is expected and tolerable; correctness via diff is the constraint, not structural fidelity.
- Reason: pure functions are required for `jit`/`vmap`/`scan` in Phase B (ADR-004) and dramatically simplify per-process testing.

### ADR-009 — Configuration: Python `dataclass` with optional YAML loader

- Namelist groups in `driver.F90` (`&time_input`, `&cntl_input`, `&met_input`, `&chem_input`) become four typed `@dataclass(frozen=True)` configs.
- A small loader function reads a YAML file and constructs the dataclasses; the same YAML file ships alongside captured reference data for reproducibility.
- Reason: scalar configs with clear types; type-safety beats free-form dicts; YAML serializes cleanly for archiving alongside NetCDF outputs.

## Recommended first port (forward-looking — executed in M3, not in this plan)

**Port `polysvp` from `wv_saturation.F90:699-736` first** as a warm-up. Rationale:

- Pure scalar thermodynamics: `polysvp(T, type) -> real(r8)` where `type ∈ {0=water, 1=ice}`. No module state inside the function.
- Tiny scope (~40 lines of Goff-Gratch polynomial evaluation) — exercises the entire validation pipeline (capture, JAX port, `float64`, 1e-6 diff, residual plot) without entangling aerosol state.
- After `polysvp`, the proposed M3 order is: `polysvp` → other `wv_saturation` leaf functions if needed → `modal_aero_wateruptake_dr` → `modal_aero_calcsize_sub` → `modal_aero_amicphys_intr` sub-processes.
- M2 below focuses reference-capture instrumentation on the **microphysics call sites** for the later ports; `polysvp` is small enough to validate from in-Python tabulation rather than instrumented Fortran runs.

## Milestone 1 — JAX package scaffold (~6 commits, 1 PR)

Each bullet ≈ one commit; the bundle ships as one PR titled `M1: scaffold JAX package`.

1. **Add `pyproject.toml`** with dependencies pinned: `jax`, `jaxlib`, `numpy`, `netCDF4`, `pyyaml`, `matplotlib` (for residual plots), `pytest` (dev). **Top-level `mam4_jax/` layout** (no `src/`).
2. **Create `mam4_jax/__init__.py` + `mam4_jax/config.py`** that calls `jax.config.update("jax_enable_x64", True)` and exports the dataclass config types from ADR-009.
3. **Create `mam4_jax/data.py`** transcribing index tables from `modal_aero_data.F90:180-185` and `modal_aero_initialize_data.F90`. Hard-code the MAM4-MOM constants (`ntot_amode=4`, `pcnst=35`, `ntot_aspectype`, species names per mode). Provide accessor helpers per ADR-007.
4. **Create empty process module stubs** under `mam4_jax/processes/`: `calcsize.py`, `wateruptake.py`, `gasaerexch.py`, `newnuc.py`, `coag.py`, `rename.py`, `amicphys.py`. Each contains a single `NotImplementedError`-raising function with the ADR-008 signature. No physics yet.
5. **Create `tests/test_scaffolding.py`** with three assertions: (a) `mam4_jax` imports cleanly, (b) `jax.config.read("jax_enable_x64")` is `True`, (c) every stub raises `NotImplementedError` when called. This is the M1 acceptance test.
6. **Add ADR-007/008/009 to `docs/KEY_DECISIONS.md`**, update `docs/PLANS.md` M1 status to "done", append M1 entry to `docs/PROGRESS.md`, update `docs/FEATURES.md` "JAX status" column where M1 changes apply.

## Milestone 2 — Fortran reference output capture (~5 commits, 1 PR)

PR title: `M2: reference output capture harness`.

1. **Build the Fortran reference locally.** Document the gfortran/NetCDF env setup in `docs/REFERENCE_BUILD.md` (new file). Patch `run_test.csh` to remove the hard-coded `outpath` belonging to a previous developer (`/Users/sunj695/...`) or factor it into an env var. **Do not modify any `.F90` files in this commit** — keep the vendored snapshot pristine (ADR-001).
2. **Capture the baseline 12-point convergence sweep.** Run the existing `run_test.csh` sweep `(1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800)` and archive the resulting NetCDF files under `tests/reference/sweep/`. This is the baseline the JAX port must reproduce in Milestone 5.
3. **Add per-process instrumentation** by writing a wrapper script `scripts/capture_reference.py` that runs the Fortran build with namelist-controlled flags, *plus* a small Fortran patch file (in `mam4-original-src-code/.patches/`, not applied to the vendored tree) that adds `write` statements before/after the three microphysics call sites at `test_drivers/driver.F90:1118` (calcsize), `:1208` (wateruptake), `:1283` (amicphys). The patch is applied at build time only — the vendored tree stays unmodified. → ADR-010 to document the "patch overlay" approach.
4. **Define the reference-capture data contract** in `tests/reference/SCHEMA.md`. **Two artifact formats:** (a) the 12-point sweep runs archive as **NetCDF** under `tests/reference/sweep/` (matches Fortran output natively, interoperates with the existing post-process notebook); (b) per-process I/O dumps from the instrumentation hooks archive as **`.npz`** under `tests/reference/per_process/` (lighter, many small files, no NetCDF metadata overhead). Each `.npz` contains the full `q`, `qqcw`, `dgncur_a`, `dgncur_awet`, `qaerwat`, `wetdens`, plus relevant scalar state (T, p, RH, dt). Used by every per-process test going forward.
5. **Add `docs/REFERENCE_BUILD.md` and update docs.** Status updates to `PROGRESS.md`, `PLANS.md` (M2 → done), `FEATURES.md` (validation row → "harness ready").

## Critical files

To create:
- `pyproject.toml`
- `mam4_jax/__init__.py`, `mam4_jax/config.py`, `mam4_jax/data.py`
- `mam4_jax/processes/{calcsize,wateruptake,gasaerexch,newnuc,coag,rename,amicphys}.py`
- `tests/test_scaffolding.py`
- `tests/reference/SCHEMA.md`
- `scripts/capture_reference.py`
- `mam4-original-src-code/.patches/driver_instrumentation.patch` (overlay; vendored tree stays clean)
- `docs/REFERENCE_BUILD.md`
- `docs/plans/001-scaffold-and-reference-capture.md` (this plan, copied)

To modify (docs only — no Fortran or new JAX code modifications outside the above):
- `docs/KEY_DECISIONS.md` (add ADR-007, 008, 009, 010)
- `docs/PROGRESS.md` (append M1 + M2 entries)
- `docs/PLANS.md` (mark M1, M2 done; flesh out M3 subtasks)
- `docs/FEATURES.md` (status column updates)

To consult (read-only) — already explored:
- `mam4-original-src-code/e3sm_src/modal_aero_data.F90:180-185` — tracer index declarations
- `mam4-original-src-code/e3sm_src/modal_aero_initialize_data.F90:250-309` — index population pattern
- `mam4-original-src-code/box_model_utils/wv_saturation.F90:699-736` — `polysvp` (forward-looking M3)
- `mam4-original-src-code/test_drivers/driver.F90:1118, 1208, 1283` — M2 instrumentation hook lines

## Verification

**M1 acceptance:**
- `python -m pytest tests/test_scaffolding.py` passes.
- `python -c "import mam4_jax; assert mam4_jax.config.x64_enabled"` returns successfully.
- Every stub in `mam4_jax/processes/` raises `NotImplementedError` when invoked.
- ADRs 007–009 appear in `docs/KEY_DECISIONS.md` with rationales.

**M2 acceptance:**
- Running `bash mam4-original-src-code/run_test.csh` (with documented env vars set) produces the 12 baseline NetCDF files under `tests/reference/sweep/` without manual path fixups.
- Running `python scripts/capture_reference.py --namelist <path>` applies the instrumentation patch, builds, runs once, and emits per-process I/O dumps conforming to `tests/reference/SCHEMA.md`.
- The vendored tree (`git diff mam4-original-src-code/`) shows zero changes after a full capture run.
- One sample diagnostic plot (e.g., baseline `q` time series at the longest timestep) lands under `docs/figures/` to confirm the capture pipeline produces inspectable output.

## Settled execution choices (approved during planning)

- **Package layout:** top-level `mam4_jax/` — no `src/` layer.
- **Capture artifact formats:** NetCDF for the 12-point sweep (`tests/reference/sweep/`); `.npz` for per-process instrumentation dumps (`tests/reference/per_process/`).
- **Task runner:** plain `python scripts/...` invocations, documented in `docs/REFERENCE_BUILD.md`. No `Makefile` or `justfile` at the repo root.
