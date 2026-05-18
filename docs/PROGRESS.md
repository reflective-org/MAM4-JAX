# Progress

A running, append-only log of project milestones. Most-recent entry on top. Update in the same PR that lands the work being recorded.

Each entry: date, short title, links to commits / PRs, one-paragraph summary.

---

## 2026-05-18 — Milestone 1 — JAX package scaffold

- PR: pending (`m1/scaffold-jax-package`)
- Added top-level `mam4_jax/` package: `__init__.py` enables `jax_enable_x64`; `config.py` defines four frozen dataclasses (`TimeConfig`, `ControlConfig`, `MetConfig`, `ChemConfig`) mirroring the Fortran namelist groups plus a `RunConfig` composite and YAML loader; `data.py` transcribes MAM4-MOM compile-time constants (PCNST=35, NTOT_AMODE=4, NTOT_ASPECTYPE=9, NSPEC_AMODE=(7,4,7,3), mode + species names) and exposes a sentinel-filled `IndexTables` with `get_number`/`get_mass` accessors that raise until M2 populates real indices.
- Added `mam4_jax/processes/` with seven `NotImplementedError`-raising stubs (`calcsize`, `wateruptake`, `gasaerexch`, `newnuc`, `coag`, `rename`, `amicphys`) using the ADR-009 pure-functional signature.
- Added `tests/test_scaffolding.py` (12 assertions; all pass against `jax 0.9.2` / `pytest 9.0.2`).
- Recorded ADR-008 (tracer rep), ADR-009 (pure-functional signatures), ADR-010 (dataclass+YAML config), ADR-011 (all-changes-via-PR, supersedes ADR-006). The technical ADRs were pre-approved in `docs/plans/001` under the numbering 007–009; the +1 shift is documented in the archived plan.

## 2026-05-18 — Plans archive convention + first plan archived

- PR: [#1](https://github.com/reflective-org/MAM4-JAX/pull/1) (merged at [`e643c20`](https://github.com/reflective-org/MAM4-JAX/commit/e643c20); content commit [`cce06f6`](https://github.com/reflective-org/MAM4-JAX/commit/cce06f6))
- Established the convention to archive approved plans under `docs/plans/NNN-<slug>.md` (ADR-007).
- Archived the first plan as `docs/plans/001-scaffold-and-reference-capture.md`, which covers Milestones 1 (JAX package scaffold) and 2 (Fortran reference output capture) and recommends `polysvp` as the M3 first-port warm-up.

## 2026-05-18 — Documentation scaffold

- Commit: [`a82e42d`](https://github.com/reflective-org/MAM4-JAX/commit/a82e42d)
- Added `docs/` with `ARCHITECTURE.md`, `PROGRESS.md`, `PLANS.md`, `KEY_DECISIONS.md`, `DEFERRED.md`, `FEATURES.md`.
- Extracted the MAM4 architecture section and embedded design decisions out of `CLAUDE.md` into `docs/ARCHITECTURE.md` and `docs/KEY_DECISIONS.md` (ADR-001 through ADR-006). `CLAUDE.md` now holds rules, guardrails, validation workflow, and pointers into the deeper docs.

## 2026-05-18 — Initial repo setup and Fortran reference vendoring

- Commit: [`22f212d`](https://github.com/reflective-org/MAM4-JAX/commit/22f212d)
- Created the MAM4-JAX repository at `reflective-org/MAM4-JAX`. Vendored the MAM4 Fortran box model as a frozen snapshot under `mam4-original-src-code/`, sourced from `reflective-org/MAM4_box_model@4150e2d` (2025-12-10). Authored initial `README.md`, `CLAUDE.md` (rules, architecture overview, behavioral guardrails). Nested `.git/` in the vendored subtree was removed so files are tracked normally; provenance is recorded in `README.md`. No JAX code yet.

---

*Future entries should follow the same format: date, title, commit/PR link, summary. Keep entries terse — link to the docs they update rather than restating the change.*
