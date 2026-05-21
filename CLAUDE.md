# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mission

Port the **MAM4 (Modal Aerosol Module, 4-mode)** aerosol-microphysics box model from Fortran 90 to **JAX**, preserving scientific behavior to within a configurable relative tolerance (start at `1e-6`).

The original Fortran reference implementation lives in `mam4-original-src-code/` (an upstream snapshot of the kaizhangpnl/MAM_box_model repository). The JAX port has not been started yet — when scaffolding the Python package, treat the Fortran tree as **read-only reference**.

## Non-negotiable working rules (from the project owner)

These override default Claude Code behavior. Re-read them before starting any task.

1. **Plan first, always.** Every task begins with a written plan: logical tasks broken into commit-sized subtasks. No code changes without an agreed plan.
2. **Each meaningful unit of progress becomes its own PR** so the owner can review before the next step.
3. **No silent assumptions.** Before making any modeling, numerical, or API choice (even small ones — tolerances, dtype, function signatures, file layout), ask. When in doubt, ask.
4. **Do not mention Claude, AI, or auto-generation in commit messages or PR descriptions.** Write them as if a human engineer authored them.
5. **Maintain living docs.** `README.md` at the root; `ARCHITECTURE.md`, `PROGRESS.md`, `PLANS.md`, `KEY_DECISIONS.md`, `DEFERRED.md`, `FEATURES.md` under `docs/`. Update them as part of the same PR that introduces the change they describe.
6. **Scientific integrity comes first.** Every ported component must be validated against the Fortran reference. Add plots whenever a visualization clarifies a discrepancy, convergence behavior, or process result — diagnostic figures are first-class deliverables, not decoration.
7. **Target relative error: `1e-6`** (rationale: `docs/KEY_DECISIONS.md` ADR-003). This threshold may be relaxed *only* with explicit owner approval, documented as a new ADR.
8. **Port properly, in two phases** (rationale: `docs/KEY_DECISIONS.md` ADR-004).
   - Phase A — **scaffold** the JAX package structure end-to-end (modules, signatures, data flow, tests) before filling in physics.
   - Phase B — fill in physics, validate, then perform a **code audit + JAX-idiom optimization pass** (vectorization, `jit`, `vmap`, `scan`, sharding decisions) after correctness is established. Do **not** prematurely optimize during initial porting.
9. **Default precision is `float64`** everywhere (rationale: `docs/KEY_DECISIONS.md` ADR-002). Aerosol microphysics is stiff and quantity ratios span many orders of magnitude; do not silently downcast. Call `jax.config.update("jax_enable_x64", True)` at package import.
10. **Ask, ask, ask.** Even for small tweaks. Better to ask one extra question than to commit a wrong assumption.
11. **Project values:** scientific integrity, modern language idioms, thorough documentation, collaborative review, reproducibility, transparency. Trade off against these explicitly when forced to.

## Behavioral guardrails

These bias toward caution over speed. For trivial tasks (typo fixes, one-line doc edits), use judgment. For everything else, treat them as binding.

### Think before coding — don't assume, don't hide confusion, surface tradeoffs

Before implementing:
- **State assumptions explicitly.** If uncertain, ask.
- **If multiple interpretations exist, present them** — don't pick silently.
- **If a simpler approach exists, say so.** Push back when warranted.
- **If something is unclear, stop.** Name what's confusing. Ask.

### Simplicity first — minimum code that solves the problem, nothing speculative

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you wrote 200 lines and it could be 50, rewrite it.

Senior-engineer test: would they call this overcomplicated? If yes, simplify.

### Surgical changes — touch only what you must; clean up only your own mess

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, **mention it — don't delete it**.

Orphans from your edits:
- Remove imports / variables / functions that **your** changes made unused.
- Do **not** remove pre-existing dead code unless asked.

Acceptance test: every changed line traces directly to the request.

### Goal-driven execution — define success criteria, then loop until verified

Translate vague tasks into verifiable goals before writing code:
- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after."
- "Port subroutine Y to JAX" → "Diff JAX output against captured Fortran reference; max rel-err < 1e-6."

For multi-step work, state the plan up front in the form:

```
1. <step>  → verify: <check>
2. <step>  → verify: <check>
3. <step>  → verify: <check>
```

Strong, checkable criteria let you iterate without re-asking. Weak ones ("make it work") leak ambiguity into the diff.

**These guardrails are working when:** diffs contain fewer drive-by changes, fewer rewrites are needed due to overcomplication, and clarifying questions arrive **before** implementation rather than after mistakes ship.

## Documentation map

Deeper detail lives under `docs/`. This file holds binding rules, guardrails, and the validation workflow; the docs hold reference material and project state.

| Doc | What's in it |
| --- | --- |
| `README.md` | Project blurb + Fortran-reference provenance. |
| `docs/ARCHITECTURE.md` | MAM4 model architecture; proposed JAX package layout; open architectural questions. |
| `docs/PROGRESS.md` | Append-only log of milestones and PRs. |
| `docs/PLANS.md` | Forward-looking roadmap, milestones, subtasks. Nothing moves from "proposed" to "in progress" without owner approval. |
| `docs/KEY_DECISIONS.md` | ADRs (Architecture Decision Records) — the *why* behind load-bearing choices. |
| `docs/DEFERRED.md` | Things explicitly punted, with the condition that brings them back. |
| `docs/FEATURES.md` | Catalog of Fortran-reference features vs. JAX-port status. |
| `docs/UPSTREAM_FORTRAN_BUGS.md` | Bugs / lint / porter-surprising patterns found in the vendored Fortran reference, to be communicated upstream. |
| `docs/plans/NNN-<slug>.md` | Archived approved plans (append-only). See ADR-007 in `KEY_DECISIONS.md`. |

Update the relevant doc in the **same PR** as the change it describes (rule #5).

## Repository layout

```
mam4-jax/                              # repo root (this directory)
  CLAUDE.md                            # this file — rules, guardrails, validation workflow
  README.md                            # project blurb + Fortran-reference provenance
  docs/                                # ARCHITECTURE / PROGRESS / PLANS / KEY_DECISIONS / DEFERRED / FEATURES
  mam4-original-src-code/              # READ-ONLY Fortran reference (vendored snapshot)
    e3sm_src/                          # MAM4 modules identical to E3SMv1
    e3sm_src_modified/                 # E3SMv1 MAM4 modules with box-model edits
    box_model_utils/                   # Box-model shims for E3SM infrastructure (ppgrid, physconst, wv_saturation, ...)
    test_drivers/                      # driver.F90 + cambox_config.*.in build config
    postprocess/                       # postprocess.ipynb (NetCDF -> figures)
    Makefile, run_test.csh             # Fortran build + convergence test harness
```

The JAX package and test harness do not exist yet — they will live under `mam4_jax/` and `tests/` when `docs/PLANS.md` Milestone 1 is approved.

## How to build & run the Fortran reference (for validation)

The Fortran reference produces the "ground truth" the JAX port must reproduce. To regenerate reference outputs:

```bash
cd mam4-original-src-code/
./run_test.csh        # csh script: builds in build/, runs in run/, writes mam_output.nc
```

Notes before running:
- Requires `gfortran` (default) or `ifort`, plus a NetCDF Fortran library. Set `NETCDF_LIB` and `NETCDF_INCLUDE` env vars (see `test_drivers/cambox_config.make.in`).
- `run_test.csh` currently hard-codes an `outpath` belonging to a previous developer (`/Users/sunj695/...`). Either edit it or remove the post-run `mv` before running.
- Default build flags (`test_drivers/cambox_config.cpp.in`) configure **MAM4 with marine organics** (`-DMODAL_AERO_4MODE_MOM`), `PCNST=35`, `PCOLS=1`, `PVER=1` — single-column, single-level box. Do not change these flags without owner approval; they define the reference configuration the port targets.
- The script sweeps timestep counts `(1 2 4 9 18 30 60 120 180 360 900 1800)` over a fixed 1800 s window — this is a **convergence test** suite. Reproducing this sweep is a key validation milestone for the JAX port.

Namelist (built inline by `run_test.csh`) controls process toggles (`mdo_gaschem`, `mdo_gasaerexch`, `mdo_rename`, `mdo_newnuc`, `mdo_coag`), meteorology (`temp`, `press`, `RH_CLEA`), and initial aerosol/gas mixing ratios. Use the same namelist values when generating reference data for any given JAX test.

## MAM4 architecture

The model architecture (modes, operator splitting, tracer layout, module roles) is documented in `docs/ARCHITECTURE.md`. **Read it before touching any porting code** — the tracer index bookkeeping in `modal_aero_data` is the single largest source of porting bugs.

## Validation workflow (apply to every port PR)

1. Identify the smallest Fortran subroutine being ported.
2. Generate reference inputs/outputs by instrumenting the Fortran code or capturing from a `run_test.csh` run.
3. Port to JAX, `float64`, no `jit` yet.
4. Diff against reference; require `max relative error < 1e-6` element-wise.
5. Produce a plot (when meaningful: time series, scatter of JAX-vs-Fortran, residual histogram). Save under `docs/figures/`.
6. Record the result in `docs/PROGRESS.md` with the PR number.
7. Only after correctness lands: open a follow-up PR for `jit`/`vmap` and update `docs/PROGRESS.md`.

## Git, commits, PRs

- `mam4-original-src-code/` is a **vendored** frozen snapshot — no nested `.git/`, no submodule. Provenance (upstream URL, commit SHA, date) is recorded in `README.md`. Do not modify files under that directory; to refresh the snapshot, replace it wholesale and update the provenance table in the same PR.
- Branch per task; PR per logical task as defined in `docs/PLANS.md`.
- Commit messages: imperative mood, scope-prefixed (`scaffold:`, `port:`, `docs:`, `test:`, `fix:`). No Claude/AI attribution. No emojis unless the owner requests them.
- Open PRs against `main` and tag with the matching `docs/PLANS.md` task ID.
