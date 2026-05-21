# Plan 005 — M3.6 PR-E: port `mam_soaexch_1subarea` (non-adaptive)

> **Status:** approved 2026-05-21.

---

## Context

PR-D (PR #17, merged) ported the H₂SO₄ analytical-solver path of `mam_gasaerexch_1subarea`, leaving SOA exchange out of scope by skipping the `call mam_soaexch_1subarea(...)` line via `scripts/patches/gasaerexch_skip_soaexch.patch`. This PR replaces that skip with a faithful JAX port of the Fortran soaexch (~330 LOC, `modal_aero_amicphys.F90:3589-3918`), bringing the JAX gasaerexch in line with the unmodified Fortran for both SOA and H₂SO₄ in one sub-process call.

## Scope decisions (owner-approved adjustments, 2026-05-21)

The original plan was a single PR with adaptive sub-stepping via Python `while`. Owner adjustments:

1. **Drop the `do_soaexch` flag** — JAX `_mam_gasaerexch_1subarea` calls soaexch unconditionally (matches Fortran's API). The PR-D `test_orchestration_gasaerexch_only_matches_fortran` test and its `per_process_gasaerexch_only/` fixture are deleted. Replaced with a single new orchestration test against a new fixture (no soaexch-skip patch).
2. **Provisional split into PR-E (non-adaptive) + PR-E2 (adaptive)** — PR-E assumes `dtcur = dtfull` and runs the step-1/step-2 solver once. Empirical math says this holds on the box-model fixture: with H₂SO₄ `sum(uptkaer) ≈ 1e-3 /s` and SOA scaled at 0.81×, the adaptive-step formula gives `dtcur = dtmax = dtfull`. A runtime assertion in the JAX port trips loudly if the assumption ever breaks. PR-E2 (when needed) adds `jax.lax.while_loop`-based adaptive stepping.
3. **`jax.lax.while_loop` for PR-E2** when triggered — JIT-ready from the start (not Python `while`).

## Forward-looking: switch to diffrax (Milestone 7, proposed)

The handwritten solvers we've been writing (H₂SO₄ exponential branches in PR-D; soaexch's step-1/step-2 semi-implicit; eventually coag's coupled ODEs in PR-G) are all candidates for replacement by `diffrax` — the JAX-native ODE/SDE library. Trade-offs captured in this plan:

- **Pro:** JIT/grad/vmap-clean out of the box, better numerics on stiff systems (Kvaerno5, KenCarp4), adaptive stepping for free.
- **Con:** Adds runtime dependency (~3 MB), changes per-step output by ~1 ULP from Fortran's specific solver choices, makes 1e-6 cross-validation against Fortran trickier on stiff problems.
- **When:** After all four sub-processes are ported faithfully (M3.6 done) so we have a stable correctness baseline. Land as Milestone 7 or fold into M6 (Phase-B optimization per ADR-004). Captured in `docs/PLANS.md` + `docs/DEFERRED.md` during this PR's subtask 7 (docs).

## Subtasks

Each ≈ one commit; single PR titled `M3.6 (PR-E): port mam_soaexch_1subarea (non-adaptive)`.

1. **Extend amicphys init dump** with SOA-specific constants: `npoa`, `nsoa`, `iaer_pom`, `iaer_soa`, `npca`, `nufi`, `mode_aging_optaa(ntot_amode)`, `lptr2_soa_a_amode(ntot_amode, nsoa)`. Add `AMICPHYS_NPOA / NSOA / IAER_POM / IAER_SOA / NPCA / NUFI`, `MODE_AGING_OPTAA`, `LPTR2_SOA_A_AMODE_PRESENT` to `data.py` (0-based; Fortran's sentinel `-999888777` → `-1`). Parity test in `tests/test_scaffolding.py`. **Done in subtask-1 commit.**

2. **Port `_mam_soaexch_1subarea`** in `mam4_jax/processes/amicphys.py`. ~200 LOC of JAX. Single-substep assumption: `dtcur = dtfull`. Runtime assertion `dtmax * tmpa <= alpha_astem` to detect any fixture where adaptive stepping would be needed. Accepts batched inputs (uses ellipsis indexing patterns from PR-B/PR-D). Returns updated `(qgas_cur, qgas_avg, qaer_cur)`. The Fortran's `qnum_cur`, `qwtr_cur` are declared inout but never modified — preserved unchanged.

3. **Wire into `_mam_gasaerexch_1subarea`** at the position matching Fortran line 3430. Unconditional (no `do_soaexch` flag).

4. **Remove the soaexch-skip patch overlay** from the build script invocation for the new mode. New capture mode `instrumented-gasaerexch-with-soaexch-only`: `mdo_gasaerexch=1, others=0`, NO `gasaerexch_skip_soaexch.patch`, YES `skip_pcarbon_aging.patch`. Output → `tests/reference/per_process_gasaerexch/`. Delete `tests/reference/per_process_gasaerexch_only/` (the soaexch-skipped fixture is no longer needed once gasaerexch is fully ported).

5. **Tests** (`tests/test_amicphys.py`):
   - Drop `test_orchestration_gasaerexch_only_matches_fortran` (no longer accurate — JAX now runs soaexch unconditionally).
   - New `test_orchestration_gasaerexch_matches_fortran` against `tests/reference/per_process_gasaerexch/amicphys_after_writeback.npz`. Rel-err < 1e-6 on `q` and `qqcw`. Size fields use 1e-3 tolerance (Fortran's `update_aerosol_props` still re-runs wateruptake mid-step).

6. **Residual plot** → `docs/figures/soaexch_residuals.png`. Time series of SOA gas + SOA aerosol mass per active mode + per-(timestep, tracer) rel-err. **Flag in chat when generating.**

7. **Docs** (rule #5): PROGRESS (M3.6 PR-E entry), PLANS (mark 5e done + new Milestone 7 diffrax migration entry), SCHEMA (per_process_gasaerexch fixture, remove `per_process_gasaerexch_only/`), REFERENCE_BUILD (replace gasaerexch-only row with the new mode), FEATURES (gasaerexch row → "full SOA path ported"). DEFERRED: diffrax-migration entry with "revisit after M3.6 done" condition.

## Critical files

To **create**:
- `tests/reference/per_process_gasaerexch/*.npz` (new fixture)
- `scripts/plot_soaexch_residuals.py`
- `docs/figures/soaexch_residuals.png`
- `docs/plans/005-soaexch-port.md` (this file)

To **modify**:
- `scripts/patches/amicphys_init_dump.patch` (extend with SOA constants — subtask 1)
- `scripts/build_reference.sh` (new mode flag)
- `scripts/capture_reference.py` (parser + new mode)
- `mam4_jax/data.py` (new constants — subtask 1)
- `mam4_jax/processes/amicphys.py` (port + wiring)
- `tests/test_amicphys.py` (new test, drop old)
- `tests/test_scaffolding.py` (parity test — subtask 1)
- Docs per rule #5.

To **delete**:
- `tests/reference/per_process_gasaerexch_only/*.npz` (replaced by the new fixture)

## Verification

- `python -m pytest -q` → 49/49 green (49 = previous 49 + 0 net new; PR-D test dropped, PR-E test added).
- `python scripts/capture_reference.py --mode instrumented-gasaerexch-with-soaexch-only --nstep 60` regenerates the new fixture.
- `python scripts/plot_soaexch_residuals.py` renders the figure (rel-err on SOA-modified tracers < 1e-6).
- Runtime assertion `dtmax * tmpa <= alpha_astem` does not trip on the box-model fixture across 60 steps.

## Out of scope

- Adaptive sub-stepping (PR-E2 if needed).
- `mam_gasaerexch_RK4_1subarea` (still off by default).
- Cloud-borne path.
- MOSAIC alternative (`mosaic_gasaerexch_1subarea_intr`).
- `mam_pcarbon_aging_1subarea` (separate sub-process; intentionally skipped via overlay).
- `qnum_cur`, `qwtr_cur` modification — Fortran declares them inout but never writes.
- diffrax migration — captured as Milestone 7.
