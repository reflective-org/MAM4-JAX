# Plan 014 — M5: convergence sweep reproduction (partial)

> **Status:** approved 2026-05-22. Completed 2026-05-22 with reduced scope (6 of 12 step counts). Remaining 6 step counts deferred to PR-E2.

---

## Context

M4 closed 2026-05-22 (PR #26). The JAX driver runs end-to-end at
`nstep = 60` (`deltat = 30s`) at max rel-err 1.97e-8.

M5 reproduces the full 12-point convergence sweep from
`mam4-original-src-code/run_test.csh`: `(1, 2, 4, 9, 18, 30, 60, 120,
180, 360, 900, 1800)` substeps over a fixed 1800 s integration window.

## Scope decision (mid-plan, 2026-05-22)

**Empirical finding** during M5 implementation: a sharp threshold at
`nstep = 60` (`deltat = 30s`) divides the sweep into two regimes:

| nstep | deltat | worst rel-err |
| ----- | ------ | ------------- |
| 1     | 1800s  | 1.3e-1        |
| 2     | 900s   | 9.4e-2        |
| 4     | 450s   | 5.7e-2        |
| 9     | 200s   | 2.5e-2        |
| 18    | 100s   | 9.7e-3        |
| 30    | 60s    | 3.4e-3        |
| **60**| **30s**| **1.97e-8**   |
| 120   | 15s    | 1.98e-8       |
| 180   | 10s    | 1.98e-8       |
| 360   | 5s     | 1.98e-8       |
| 900   | 2s     | 1.98e-8       |
| 1800  | 1s     | 1.98e-8       |

**Diagnosis**: at `dt ≥ 60s`, Fortran's
`mam_soaexch_1subarea`'s adaptive substepping
(`modal_aero_amicphys.F90:3835-3843`, `dtcur = alpha_astem/tmpa`)
breaks one amicphys call into multiple smaller integration steps. The
JAX port's `_mam_soaexch_1subarea` assumes single-substep
(`dtcur = dtfull`); the M3.6 PR-E plan flagged this as deferred to
"PR-E2 if a fixture ever triggers it" (per `docs/DEFERRED.md`).

**Owner-approved decision (2026-05-22)**: validate the `nstep ≥ 60`
half in this PR. Open PR-E2 separately for adaptive SOA substepping.
After PR-E2 lands, re-run M5 to close the remaining 6 step counts.

## Scope (this PR)

**Capture infrastructure**:

- New `--mode sweep-no-pcarbon-aging` in `scripts/capture_reference.py`:
  12 NetCDF runs with `skip_pcarbon_aging.patch` applied at build time.
  Matches the JAX port's M3.6 scope. Output →
  `tests/reference/sweep_no_pcarbon_aging/mam_dt<DT>_ndt<N>.nc`.
- `scripts/build_reference.sh` constraint relaxed: `--skip-pcarbon-aging`
  no longer requires `--instrumented`. The patch is independent of
  the instrumentation overlay; the prior coupling was a leftover.

**Tests** (`tests/test_sweep.py`, parametrized):

- `test_sweep_matches_fortran[nstep]` for `nstep ∈ {60, 120, 180, 360,
  900, 1800}`: validate that JAX `run_timesteps(ic, nstep)`
  reproduces every per-step entry of Fortran's NetCDF output
  (`num_aer`, `so4_aer`, `soa_aer`, `h2so4_gas`, `soag_gas`) at
  `rtol=1e-6, atol=1e-20`. `dgn_a` at `rtol=1e-3, atol=1e-15` (size-
  field caveat from prior milestones).
- `test_sweep_xfail_without_adaptive_soa_substep[nstep]` for `nstep ∈
  {1, 2, 4, 9, 18, 30}`: explicitly `xfail`ed with the PR-E2 deferral
  reason. The worst num_aer rel-err is quoted in the xfail message so
  the size of the gap stays visible. When PR-E2 lands, the assertions
  flip to "expect passing" and these step counts move into `NSTEP_OK`.

**IC**: reuse
`tests/reference/per_process_full_minus_pcarbon_aging/calcsize_before[0]`
from M4 PR-A. The IC depends only on the namelist, not on `nstep`, so
the same snapshot serves every sweep point.

**Plot** `docs/figures/sweep_convergence.png`:
- Top-left: per-mode final-step number-density vs `nstep` (log x),
  Fortran solid / JAX dashed.
- Top-right: final-step H₂SO₄ gas vs `nstep`.
- Bottom: worst rel-err per `nstep` (semilog y) with the ADR-003 1e-6
  reference line and a shaded "PR-E2 deferred" region for `nstep ≤ 30`.

## Verification

- `python -m pytest tests/test_sweep.py -v` → 6 passed, 6 xfailed.
- `python -m pytest tests/` → 67 passed, 6 xfailed.
- `python scripts/capture_reference.py --mode sweep-no-pcarbon-aging`
  regenerates the fixture deterministically (12 NetCDF files).
- `python scripts/plot_sweep_convergence.py` regenerates the figure.
- Worst rel-err in the validated half: **1.98e-8** (50× under ADR-003).

## What this PR does NOT do

- No adaptive SOA substepping (PR-E2 follow-up; ~100 LOC change to
  validated code).
- No NetCDF output emission from JAX (deferred — only needed if the
  post-process notebook is wanted against JAX outputs).
- No M6 / M7 work (audit + JAX-idiom optimization / diffrax).
- No pcarbon-aging port (deferred per `docs/DEFERRED.md`).

## Open questions

None blocking. PR-E2 is the next concrete step; the rest of the
roadmap (M6 audit + JAX-idiom optimization, M7 diffrax migration)
opens up once PR-E2 lands.
