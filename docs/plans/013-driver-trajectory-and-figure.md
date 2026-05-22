# Plan 013 — M4 PR-B: 60-step trajectory test + size-distribution figure

> **Status:** approved 2026-05-22. Completed 2026-05-22. **Closes M4.**

---

## Context

PR-M4-A (PR #25, merged 2026-05-22) scaffolded the operator-splitting
driver in `mam4_jax/driver.py` (`run_step` + `run_timesteps`) and
validated the per-step wiring on a single timestep — JAX matched
Fortran at max rel-err 2.5e-9.

PR-M4-B exercises the trajectory accumulation: drives JAX
`run_timesteps(ic, 60)` from `calcsize_before[0]` of the
`per_process_full_minus_pcarbon_aging/` fixture and validates every
per-step snapshot against the Fortran capture. **Closes M4.**

After M4 lands, M5 (12-point convergence sweep + NetCDF reproduction)
is unblocked.

## Scope (PR-M4-B)

**Test** (`tests/test_driver.py`):

- New `test_run_timesteps_60_step_trajectory_matches_fortran`.
- IC: `per_process_full_minus_pcarbon_aging/calcsize_before[step=0]`.
- `traj = run_timesteps(ic, n_steps=60)`.
- Assert `traj["q"]` matches Fortran `amicphys_after_writeback["q"]`
  at `rtol=1e-6, atol=1e-20`; same for `qqcw`.
- Size fields (`dgncur_a` / `dgncur_awet` / `qaerwat` / `wetdens`) at
  `rtol=1e-3, atol=1e-15` — same Fortran mid-substep
  `update_aerosol_props` caveat as the per-process amicphys tests.

Empirically the worst rel-err is ~2e-8 (Aitken-mode number, mid-
trajectory). 50× under ADR-003. Errors flatten by step ~5 — no
runaway accumulation.

**Figure** `docs/figures/driver_60step_trajectory.png` (script:
`scripts/plot_driver_trajectory.py`):

- **Top — 4 mode panels (2×2 grid)**, one per MAM4-MOM mode (accum /
  Aitken / coarse / pcarbon). Each panel has dual y-axes:
  - Left (log scale): number-density `q[..., NUMPTR_AMODE[mode]]`
    in `#/kmol-air`.
  - Right (linear): dry diameter `dgncur_a[..., mode]` in nm.
  - x-axis: timestep index (0..59).
  - Fortran solid (lw 2.0), JAX dashed (lw 0.9).
- **Bottom — rel-err panel**: per-(step, tracer) `|rel-err|` for all
  35 tracers, semilog y, with ADR-003 1e-6 reference line and
  float64 ε reference line. Title quotes worst rel-err with step
  and tracer indices.
- Figure suptitle quotes the overall worst rel-err.

Per `feedback-validation-must-be-driven`: this is a self-driven JAX
trajectory comparison against the Fortran capture, **not** per-step
JAX on captured before-states. This is the criterion from the
owner's earlier "wait for M4" decision.

## Verification

- `python -m pytest tests/test_driver.py -v` → 4 tests pass (3 from
  PR-A + 1 new).
- `python -m pytest tests/` → 61/61 green.
- `python scripts/plot_driver_trajectory.py` regenerates the figure.
- Worst trajectory rel-err: **1.97e-8** at step 29, tracer 17 (Aitken
  number). 50× under ADR-003.

## What this PR does NOT do

- No `jax.lax.scan` (M6).
- No 12-point convergence sweep (M5).
- No NetCDF output emission (M5 or later).
- No gas-chem extraction to driver layer (deferred — see PR-M4-A).
- No pcarbon-aging port (deferred per `docs/DEFERRED.md`).

## Open questions

None. M5 (convergence reproduction) is the next milestone; the M5
section in `PLANS.md` now carries tentative subtasks.
