# Handwritten-solver limitations of the `v0.1.0` baseline

> **Scope:** This document describes the MAM4-JAX implementation as
> it stands on `main` at the `v0.1.0` tag (and equivalently, on the
> `diffrax` branch up to PR-I1, before any solver port). It is
> useful for anyone checking out `v0.1.0` directly, or auditing
> what the eventual `diffrax → main` merge-back replaces.

## What `v0.1.0` covers

- **M0** — repo + documentation scaffold.
- **M1** — JAX package scaffold; `float64` enforced at import; eight
  process module stubs.
- **M2** — Fortran reference capture (12-point convergence sweep +
  per-process instrumentation overlay).
- **M3.1–3.5** — `polysvp`, `qsat_water`/`qsat_ice`, `IndexTables`,
  the full water-uptake chain (`makoh_*` → `modal_aero_kohler` →
  `modal_aero_wateruptake_sub`), and `modal_aero_calcsize_sub`.
- **M3.6** — `amicphys` orchestration plus all four sub-processes
  (`rename`, `gasaerexch` incl. H₂SO₄ analytical solver,
  `soaexch`, `newnuc`, `coag`) ported with handwritten solvers.
- **M4** — operator-splitting driver (`run_step`, `run_timesteps`)
  validated over 60 steps at max rel-err 1.97e-8.
- **M5 (partial)** — convergence-sweep reproduction: 6 of 12 step
  counts validated at `rtol=1e-6`; remaining 6 marked `xfail` (see
  below).

See `docs/PROGRESS.md` for the full milestone log with PR links.

## The three handwritten solver call sites

All three live in `mam4_jax/processes/amicphys.py`:

1. **`_mam_soaexch_1subarea`** — semi-implicit step-1/step-2 solver
   for SOA gas/aerosol exchange. Assumes one substep per call
   (`dtcur = dtfull`). Fortran's equivalent triggers adaptive
   substepping inside `mam_soaexch_1subarea` when
   `dtfull * tmpa > alpha_astem`
   (`mam4-original-src-code/e3sm_src_modified/modal_aero_amicphys.F90`,
   subroutine `mam_soaexch_1subarea`); the JAX port does not.
2. **H₂SO₄ analytical uptake** inside `_mam_gasaerexch_1subarea` —
   three-branch closed form on `tmp_kxt`: `tmp_kxt > 0.001` uses
   `exp(-tmp_kxt)`, `tmp_kxt ≤ 0.001` uses a Taylor expansion,
   `tmp_kxt < 1e-20` skips the `qaer` update. No adaptive control;
   the branch boundaries are hand-picked.
3. **Coag's analytical number-loss + exp-decay mass transfer** in
   `_mam_coag_1subarea` — closed-form `(1 - exp(...))` decay for
   number, two-branch number-loss guard at `tmpa < 1e-5`. No
   coupled-ODE solver; the time integration is hand-derived per
   pair.

## Known accuracy gap: `nstep ≤ 30` on the convergence sweep

At Fortran-driver substep sizes `dt ≥ 60 s` (i.e. `nstep ≤ 30`
over the 1800 s window), the handwritten `_mam_soaexch_1subarea`'s
single-substep assumption diverges from Fortran's adaptive
substepping. From `tests/test_sweep.py` (xfailed) — worst per-step
relative error vs Fortran on `num_aer`/`so4_aer`/`soa_aer`/
`h2so4_gas`/`soag_gas`:

| nstep | dt    | worst rel-err |
| ----- | ----- | ------------- |
| 1     | 1800s | 1.3e-1        |
| 2     | 900s  | 9.4e-2        |
| 4     | 450s  | 5.7e-2        |
| 9     | 200s  | 2.5e-2        |
| 18    | 100s  | 9.7e-3        |
| 30    | 60s   | 3.4e-3        |
| **60**| 30s   | 1.97e-8       |
| ...   | ...   | ~1.98e-8      |

The `nstep ≥ 60` half of the sweep is validated at `rtol=1e-6`
(50× under ADR-003's threshold). The `nstep ≤ 30` half is marked
`xfail` on `main` indefinitely; resolution lives on the `diffrax`
branch.

## Other intentional gaps

- **No NetCDF output emission from JAX.** The post-process
  notebook (`mam4-original-src-code/postprocess/postprocess.ipynb`)
  is currently driven by Fortran NetCDFs only. Adding JAX-side
  NetCDF emission is deferred (see `docs/DEFERRED.md`).
- **No primary-carbon aging.** All M3.6+ reference captures apply
  `scripts/patches/skip_pcarbon_aging.patch` to the Fortran build.
  The pcarbon-aging port is deferred (see `docs/DEFERRED.md`).
- **No `jit` / `vmap` / `scan`.** Phase A (correctness) only;
  optimization is Milestone 6.
- **No marine-organics modes.** The Fortran box-model
  configuration uses MAM4-MOM (4 modes + MOM), but the JAX coag
  port dropped marine-organics blocks where MAM4 vs MAM4-MOM diverge
  (the modes are absent in the validation fixture).

## What the `diffrax` branch resolves

Per ADR-013 / ADR-014, the parallel `diffrax` branch replaces the
three handwritten solver call sites with diffrax-based equivalents
(default solver `Kvaerno5`, adaptive PI controller). The expected
outcome:

- The 6 `xfail`ed `nstep ≤ 30` cases flip to expected-pass at
  `rtol=1e-6` (with the ADR-013 ~1 ULP slack rule available).
- JIT/grad/vmap cleanliness improves (handwritten branches with
  `jnp.where` on solver state are replaced by diffrax's controller
  logic).
- Adaptive-controller diagnostics (step counts, rejection ratios)
  become first-class via `solvers.SolverResult.stats`.

When the diffrax branch is validated end-to-end, ADR-014 plans a
`diffrax → main` merge-back; at that point this document and the
six `xfail` markers in `tests/test_sweep.py` are removed.

## Pointers

- ADR-013 (`docs/KEY_DECISIONS.md`) — dual-branch rationale.
- ADR-014 (`docs/KEY_DECISIONS.md`) — merge-back intent, sync via
  merge.
- `docs/PLANS.md` Milestone 7 — sub-PR breakdown (PR-I1 / PR-D1 /
  PR-D2 / PR-D3).
- `docs/plans/015-diffrax-infra.md` — full PR-I1 spec.
- `docs/plans/014-convergence-sweep.md` — M5 partial reproduction
  plan, where the `nstep ≤ 30` gap was characterized.
- `docs/DEFERRED.md` — other punted items.
