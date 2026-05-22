# Plan 012 — M4 PR-A: operator-splitting driver scaffold

> **Status:** approved 2026-05-22. Completed 2026-05-22.

---

## Context

M3.6 is complete (PR #24, merged 2026-05-22). The four amicphys
sub-processes — gasaerexch, rename, newnuc, coag — are all ported,
wired into the orchestration, and validated per single-toggle Fortran
capture. Each process has been tested independently.

M4 builds the operator-splitting *driver* that chains calcsize →
wateruptake → amicphys per timestep over the 1800 s integration
window. After M4 lands, the JAX side can run an actual time evolution
(rather than per-step on captured Fortran before-states), which is
what the owner asked about earlier when requesting a mode-by-mode
size-distribution comparison figure.

**Owner-approved split** (2026-05-22):
- **PR-M4-A (this plan)**: scaffold the driver module, 1-step wiring
  test, new fixture that matches JAX's M3.6 scope. Plus structural
  smoke tests for `run_timesteps`. **No trajectory test, no figure.**
- **PR-M4-B**: 60-step trajectory test + the mode-by-mode
  size-distribution comparison figure (4 modes × {number, dg} time
  series, Fortran solid / JAX dashed, plus a per-(step, tracer)
  rel-err panel). **Completes M4.**

## Scope (PR-M4-A)

**Implementation** (`mam4_jax/driver.py`, ~120 LOC):

- `run_step(state) -> new_state`: one operator-splitting timestep.
  Sequence mirrors `driver.F90:1080-1367`'s `main_time_loop` for the
  MAM4-MOM box-model fixture:
  1. `calcsize(state)` — size redistribution + apply tendency to `q`.
  2. `wateruptake(state)` — wet diameters, aerosol water, wet density.
  3. *gas-chem* — currently absorbed inside `gasaerexch`'s H₂SO₄
     analytical solver (`processes/amicphys.py:594`: `qgas_netprod_h2so4 = 1e-16`).
     **Not lifted to driver layer in PR-A** because doing so requires
     operator-splitting between gas-chem and gasaerexch and reworking
     the validated PR-D analytical solver — out of M4-A scope. M5's
     namelist-sweep work may force the lift; defer until then.
  4. `cloud_chem_simple_sub(state)` — no-op stub. Box-model fixture
     has `cldn=0` so Fortran's `if (cld > 1e-6)` gate at `driver.F90:1263`
     never fires. Stubbed so the operator-splitting sequence reads
     correctly; implement when a future fixture demands it.
  5. `amicphys(state)` — defaults to all `mdo_*=1`. The full
     microphysics including the vmr↔mmr writeback (implicit via
     `_repack_amicphys_view_to_state`).
- `run_timesteps(state, n_steps) -> trajectory`: plain Python `for`
  loop. Returns a dict of stacked snapshots — leading axis = `n_steps`.
  The IC is NOT included (matches Fortran's `do nstep = 1, nstop`
  convention; NetCDF output indexes step `i` as the post-step-`i+1`
  state). `jax.lax.scan` deferred to M6 per ADR-004.

**Validation infrastructure**:

- New `--mode instrumented-full-minus-pcarbon-aging` in
  `scripts/capture_reference.py`. Canonical full-physics namelist
  (all `mdo_*=1`) with `scripts/patches/skip_pcarbon_aging.patch`
  applied at build time. Matches the JAX port's M3.6 scope (pcarbon
  aging deferred).
- Output → `tests/reference/per_process_full_minus_pcarbon_aging/`
  (9 fixtures including the rename hook, 60 steps each).

**Why a new fixture is needed**. The canonical `per_process/` fixture
has pcarbon-aging ON. JAX's M3.6 port omits pcarbon-aging
intentionally. A 1-step driver test against `per_process/` diverges
by ~20% on Aitken/pcarbon tracers — well above ADR-003's 1e-6 budget
and unrelated to the driver's correctness. The new fixture removes
the confound.

**Tests** (`tests/test_driver.py`, 3 new tests):

1. `test_run_step_one_step_matches_fortran`: JAX `run_step` on
   `calcsize_before[0]` reproduces `amicphys_after_writeback[0]` at
   `rtol=1e-6, atol=1e-20` on `q`/`qqcw`; size fields at
   `rtol=1e-3, atol=1e-15` (Fortran mid-substep `update_aerosol_props`
   caveat). **Actual worst rel-err: 2.5e-9 on `q`** — 3 orders below
   ADR-003.
2. `test_run_timesteps_shapes`: trajectory leading-axis size matches
   `n_steps`; step-0 snapshot equals the standalone `run_step` output.
3. `test_run_timesteps_rejects_zero`: `n_steps=0` raises (matches
   Fortran's `do nstep = 1, nstop` requiring `nstop >= 1`).

## Verification

- `python -m pytest tests/test_driver.py -v` → 3 tests pass.
- `python scripts/capture_reference.py --mode instrumented-full-minus-pcarbon-aging --nstep 60`
  regenerates the fixture deterministically.
- Full suite: **60/60 green** (57 + 3 new).
- Vendored tree (`mam4-original-src-code/`) unchanged.

## What this PR does NOT do

- No 60-step trajectory test (PR-M4-B).
- No size-distribution comparison figure (PR-M4-B).
- No `jax.lax.scan` (M6).
- No gas-chem extraction to driver layer (deferred — only required
  when M5 needs per-namelist sweep variants).
- No cloud-chem implementation (stubbed; current fixture doesn't need it).
- No pcarbon-aging port (deferred per existing `docs/DEFERRED.md` entry).

## Open questions

None. PR-M4-B will exercise the trajectory accumulation and produce
the figure the owner asked about.
