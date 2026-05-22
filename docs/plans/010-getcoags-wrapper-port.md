# Plan 010 — M3.6 PR-G2: port `getcoags_wrapper_f`

> **Status:** approved 2026-05-21. Completed 2026-05-21. Folded into PR #23 alongside PR-G1 per owner direction.

---

## Context

PR-G1 (this branch, prior commit) ported `getcoags` (the closed-form
Whitby coagulation-coefficient leaf) and captured a 240-record fixture
that includes **both** `getcoags` outputs and `getcoags_wrapper_f`
outputs in the same `.npz`. PR-G2 is the wrapper:
`modal_aero_coag.F90:999-1129` (~130 LOC) — input prep, a call to
`getcoags`, and the "CMAQ → MIRAGE2" post-processing that turns the
8 raw coagulation rates into the 8 `betaij*`/`betaii*`/`betajj*`
coefficients consumed by `mam_coag_1subarea`.

## Scope

**Port** in `mam4_jax/coag.py` (~70 new LOC, ending the file):

`getcoags_wrapper_f(airtemp, airprs, dgatk, dgacc, sgatk, sgacc, xxlsgat, xxlsgac, pdensat, pdensac)` → 8-tuple
`(betaij0, betaij2i, betaij2j, betaij3, betaii0, betaii2, betajj0, betajj2)`.

Steps:
1. `t0 = TMELT + 15`, `sqrt_temp = sqrt(airtemp)`.
2. `lamda = 6.6328e-8 · PSTD · airtemp / (t0 · airprs)` — mean free path (U.S. Standard Atmosphere 1962 table I.2.8).
3. `amu = 1.458e-6 · airtemp · sqrt_temp / (airtemp + 110.4)` — dynamic viscosity (U.S. Std Atm 1962 page 14).
4. `knc / kfmat / kfmac / kfmatac` from the Binkowski-Shankar 1995 formulas with `BOLTZ`.
5. Call `getcoags(lamda, kfmatac, kfmat, kfmac, knc, dgatk, dgacc, sgatk, sgacc, xxlsgat, xxlsgac)` → 8 raw outputs.
6. Compute `dumacc2 = dgacc² · exp(2 log²σac)`, `dumatk2 = dgatk² · exp(2 log²σat)`, `dumatk3 = dgatk³ · exp(4.5 log²σat)`.
7. Build the 8 betas with `max(0, ...)` clamps and divisions by `dumatk2 / dumacc2 / dumatk3` per the Fortran's exact assignment order.

**Constants** added to `mam4_jax/constants.py`:
- `PSTD = 101325.0` Pa (`shr_const_pstd`).
- `TMELT = 273.15` K (`shr_const_tkfrz`; the box-model never overrides the namelist-settable `tmelt`).

These are the first JAX consumers of either constant — they belong in
`constants.py` rather than inlined in `coag.py` so future modules can
reuse them.

## Validation strategy

The PR-G1 fixture (`tests/reference/coag_coefficients/reference.npz`)
already carries `betaij0`/`betaij2i`/`betaij2j`/`betaij3`/`betaii0`/
`betaii2`/`betajj0`/`betajj2` — they were captured for this exact
purpose. No new Fortran capture, no new driver build.

New test in `tests/test_coag.py`: `test_getcoags_wrapper_f_matches_fortran`.
`rtol = 1e-6` on all 8 outputs across 240 records.

**Expected behavior**: 7 of 8 outputs at machine ε (the post-processing
is a clamp + division, both numerically stable); `betaij2j` inherits
PR-G1's 6.5e-9 worst rel-err because it is `qs21 / dumatk2` and
`qs21` is the limiting `getcoags` output.

## Figures

Extend `scripts/plot_getcoags_residuals.py` to render two figures in
one run:
- `docs/figures/getcoags_residuals.png` (existing, PR-G1).
- `docs/figures/getcoags_wrapper_residuals.png` (new, this PR) — same
  4×2 layout for the 8 betas. Visual confirmation that the
  post-processing does not amplify the leaf-port error.

## Subtasks

1. Add `PSTD` and `TMELT` to `mam4_jax/constants.py`.
2. Add `getcoags_wrapper_f` to `mam4_jax/coag.py`; update module docstring.
3. Add `test_getcoags_wrapper_f_matches_fortran` to `tests/test_coag.py`.
4. Extend `scripts/plot_getcoags_residuals.py` to render the wrapper figure (flag layout in chat first).
5. Docs: PROGRESS PR-G2 entry; PLANS mark 5g.PR-G2 done; FEATURES coag row + new wrapper row; plan archived here.

## Verification

- `python -m pytest tests/test_coag.py -v` → 2 tests pass.
- `python -m pytest tests/` → 56/56 green (55 + 1 new).
- `python scripts/plot_getcoags_residuals.py` regenerates both figures.

## What this PR does NOT do

- No `_mam_coag_1subarea` orchestration; no amicphys wiring (PR-G3).
- No new Fortran capture (intentional — PR-G1's fixture already covers this).
- `mam4_jax/processes/coag.py` stub is untouched.

## Risks / open questions

None — the wrapper is a thin post-processing layer over PR-G1's
already-validated `getcoags`. The expected worst rel-err equals PR-G1's
worst rel-err, since the `betaij2j` path passes the dominant residual
through unchanged.
