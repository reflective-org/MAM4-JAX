# Plan 009 — M3.6 PR-G1: port `getcoags` leaf

> **Status:** approved 2026-05-21. Completed 2026-05-21.

---

## Context

PR-F3 (PR #22, merged) finished `mam_newnuc_1subarea` and closed out
M3.6 PR-F (newnuc). Only PR-G (`mam_coag_1subarea`, ~437 LOC of
amicphys orchestration glue) remains in M3.6. Reading the call chain
from `mam_coag_1subarea` exposes a much larger dependency tree:

- `mam_coag_1subarea` (~437 LOC, amicphys-internal)
- → `getcoags_wrapper_f` (~130 LOC, in `modal_aero_coag.F90`) — prep
  math + post-processing of the 8 raw coefficients into the 8
  `betaij*` / `betaii*` / `betajj*` consumed by the orchestration
- → `getcoags` (~1685 LOC, in `modal_aero_coag.F90`) — closed-form
  Whitby coagulation coefficients. **~1200 of those LOC are
  correction-factor lookup tables** (`bm0`, `bm0ij`, `bm3i`, `bm2ii`,
  `bm2iitt`, `bm2ij`, `bm2ji`); the actual physics is ~250 LOC of
  exponentials and closed-form arithmetic.

The owner-approved split (mirroring the PR-F newnuc pattern):
- **PR-G1 (this plan)**: port the deepest leaf `getcoags` as a pure JAX
  function, validate via a standalone Fortran driver.
- **PR-G2**: port `getcoags_wrapper_f` (prep + post-processing), reuse
  the PR-G1 fixture (driver captures wrapper outputs too).
- **PR-G3**: port `mam_coag_1subarea`, wire into `_mam_amicphys_1subarea_clear`,
  add the single-toggle Fortran capture + end-to-end orchestration test.
  Completes M3.6.

## Scope (PR-G1)

**Port** in new module `mam4_jax/coag.py`:
- `getcoags(lamda, kfmatac, kfmat, kfmac, knc, dgatk, dgacc, sgatk, sgacc, xxlsgat, xxlsgac)` → 8-tuple `(qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12)`.
- Batch-friendly via standard JAX broadcasting.
- Direct translation of the Fortran closed-form math, preserving
  operator order where 1-ULP-relevant (same pattern we've used for
  gasaerexch / mer07_veh02 / etc.).

**Lookup tables** extracted once by `scripts/extract_coag_tables.py`
into `mam4_jax/_coag_tables.npz` and loaded at module import time. The
extractor is committed so future Fortran-source bumps can re-run it.

**Index lookup** uses `jnp.clip(jnp.round(...), 1, 10) - 1` to mirror
Fortran's `max(1, min(10, nint(...)))`. JAX's `round` is banker's
rounding while `nint` rounds half-away-from-zero, but for the
`4*(sigmag - 0.75)` and `1 + log/log(√2)` arguments those branches
never land on half-integers in MAM4 use, so the difference is moot.
Documented in the source.

## Validation strategy

**New standalone Fortran driver** `scripts/reference_drivers/coag_coefficients_driver.F90`:

Sweeps (4 T × 2 P × 5 dgnumA × 6 dgnumB = **240 records**) for fixed
MAM4-MOM defaults (`sg_atk=1.6`, `sg_acc=1.8`, `pdens_atk=pdens_acc=1770 kg/m³`).
For each grid point the driver:
1. Computes the intermediates (`lamda`, `knc`, `kfmat`, `kfmac`,
   `kfmatac`) using the prep code from `getcoags_wrapper_f`.
2. Calls `getcoags` with those intermediates → 8 raw outputs.
3. Calls `getcoags_wrapper_f` with the same physical inputs → 8
   post-processed coefficients (PR-G2 validation surface).

Output `.npz` carries both sets of outputs so the same fixture serves
both PR-G1 and PR-G2 — no extra Fortran capture needed for the wrapper.

**Patches**: `expose_internals.patch` extended with a third hunk
making `getcoags` `public` in `modal_aero_coag` alongside the existing
`getcoags_wrapper_f` exposure.

**Output ranges** justify the wide-format `1pe27.16e3` writer:
`qv12 ~1e-38 to 5e-35`, `qn11 ~1e-15 to 1e-12`, `qs11 ~1e-32 to 1e-30`.

## Subtasks

1. Extend `scripts/patches/expose_internals.patch` to expose `getcoags`.
2. Write `scripts/reference_drivers/coag_coefficients_driver.F90`.
3. Add `--coag-coefficients` build flag (build_reference.sh) +
   `--mode coag-coefficients` capture mode (capture_reference.py
   with `_read_coag_coefficients` parser).
4. **JAX port** in new `mam4_jax/coag.py`:
   `getcoags(...)` returning the 8-tuple. Load lookup tables from
   `_coag_tables.npz` (extracted by `scripts/extract_coag_tables.py`,
   also new in this PR).
5. `tests/test_coag.py::test_getcoags_matches_fortran` — `rtol=1e-6`
   on all 8 outputs across 240 records.
6. `docs/figures/getcoags_residuals.png` — 4×2 grid of JAX-vs-Fortran
   log-log scatter, one panel per output, colored by Whitby table
   index `n1`. (User confirmed layout choice in chat before generation.)
7. Docs: PROGRESS PR-G1 entry; PLANS split 5g into G1/G2/G3 and mark
   G1 done; SCHEMA `coag_coefficients/` section; REFERENCE_BUILD
   `coag-coefficients` row + standalone-driver paragraph update;
   FEATURES update; plan archived here.

## Verification

- `python -m pytest tests/test_coag.py -v` passes.
- Worst rel-err across 8 outputs × 240 records: **6.5e-9** (three orders
  below ADR-003 budget).
- Full suite: 55/55 green (54 + 1 new).
- `mam4_jax/_coag_tables.npz` regenerable by `python scripts/extract_coag_tables.py`.

## What this PR does NOT do

- No `getcoags_wrapper_f` (PR-G2).
- No `_mam_coag_1subarea` orchestration; no amicphys wiring (PR-G3).
- No single-toggle Fortran capture; no end-to-end orchestration test
  (PR-G3).
- Stub `mam4_jax/processes/coag.py` and the no-op `_mam_coag_1subarea`
  inside `amicphys.py` are untouched.

## Risks / open questions

- **Index rounding**: documented above. If a future fixture hits a
  half-integer argument, swap `jnp.round` for `jnp.floor(x + 0.5)`.
- **No batching test**: the standalone-driver fixture passes one
  scalar tuple per record; we exercise broadcasting in the JAX test by
  passing 240-element 1-D arrays. Higher-rank batching (the shape
  pattern PR-G3 will need) is not validated here — PR-G3's end-to-end
  test will cover that.
