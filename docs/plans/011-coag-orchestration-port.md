# Plan 011 — M3.6 PR-G3: port `mam_coag_1subarea` + wire into amicphys

> **Status:** approved 2026-05-22. Completed 2026-05-22. **Closes M3.6.**

---

## Context

PR-G1 (PR #23, merged 2026-05-21) ported `getcoags` (closed-form
Whitby coagulation coefficients) and PR-G2 (folded into the same PR)
ported `getcoags_wrapper_f` (input prep + CMAQ→MIRAGE2
post-processing). PR-G3 is the amicphys orchestration glue
(`modal_aero_amicphys.F90:4670-5106`, ~437 LOC) that composes them
into a per-substep number/mass-transfer step.

After PR-G3 lands, **M3.6 is complete**. The next milestone is M4 —
the operator-splitting time loop that drives calcsize → wateruptake →
amicphys for the full 1800 s integration window.

## Scope decisions

**MAM4-MOM-specific simplifications** (relative to the full 5-mode Fortran):

- No marine-organics modes (`nmait < 0`, `nmacc < 0`) → all `if (nmait > 0)`
  and `if (nmacc > 0)` blocks are dead code. Drop them (~50 LOC saved).
- Active coag-pair count is exactly 3 (Fortran's `ip = 1, 2, 3`):
  1. Aitken → accum
  2. pcarbon → accum
  3. Aitken → pcarbon (aging path; eventually coarsens to accum)
- Coarse mode never enters coag (Brownian rates negligible at super-µm
  diameters — correct).
- `qaer_del_coag_in` output (feeds `mam_pcarbon_aging_1subarea`) is not
  accumulated — the matching capture applies `skip_pcarbon_aging.patch`
  so pcarbon aging is a no-op there too. Same pattern as PR-D/E/F3.
- Diagnostic-output blocks (`CAMBOX_ACTIVATE_THIS` guards) omitted.

**Branch reformulations for JAX:**

- Fortran's two-branch number-loss formula
  (`if (tmpa < 1e-5) qnum = ... else qnum = tmpn*exp(-tmpa)/(1+(tmpb*tmpn/tmpa)*(1-exp(-tmpa)))`)
  → `jnp.where` with `safe_tmpa = jnp.where(small, 1, tmpa)` so the
  dead branch never NaNs from 0-division.
- Fortran's `if (tmpc > epsilonx2)` mass-transfer guard
  → multiply by `jnp.where(have_coag, 1 - exp(-tmpc), 0)`; the dead
  branch contributes zero to all `qaer` updates naturally.

**Data tables added** (`mam4_jax/data.py`):

- `PCARBON_MODE_IDX = 3` (was missing; `AITKEN_MODE_IDX` and
  `ACCUM_MODE_IDX` were already defined).
- `N_COAGPAIR = 3`, `MODEFRM_COAGPAIR = (1, 3, 1)`, `MODETOO_COAGPAIR = (0, 0, 3)`
  (0-based MAM4-MOM mode indices). Derived deterministically from
  Fortran's init loop given the MAM4-MOM mode constants.

## Validation strategy

**New capture mode** `instrumented-coag-only`: namelist `mdo_coag=1,
mdo_gasaerexch=mdo_rename=mdo_newnuc=0` plus `skip_pcarbon_aging.patch`.
No new Fortran patch beyond reusing existing infrastructure.

Output → `tests/reference/per_process_coag/` (7 fixtures, one per
hook tag, 60 records each).

**New test** in `tests/test_amicphys.py`:
`test_orchestration_coag_only_matches_fortran`. Validates `q` and
`qqcw` at `rtol=1e-6, atol=1e-20` for **aerosol-tracer slots only**
(33 of 35); size fields at `rtol=1e-3, atol=1e-15` (same caveat as
prior PRs — Fortran's mid-step `update_aerosol_props` re-uptake is
out of M3.6 scope).

**Why gas-tracer slots are excluded.** `driver.F90:1249` applies a
`vmr += 1e-16·deltat` gas-chem stub to H₂SO₄ **outside** the amicphys
call. Fortran's writeback dump captures this driver-side increment;
the JAX orchestration (no driver layer at PR-G3) doesn't apply it.
The `gasaerexch` test absorbs the same term inside the H₂SO₄
analytical solver's `qgas_netprod_otrproc`; coag-only has no such
mechanism because gasaerexch is off. Since coag itself never touches
gases, the gas-tracer slots are not part of coag's validation surface
— excluding them is the right scoping, not a workaround for a JAX
bug.

When M4's time loop lands, the driver-level gas-chem step will move
into JAX, and gas tracers will be validated end-to-end against the
Fortran NetCDF sweep.

## Subtasks (all completed)

1. Add `PCARBON_MODE_IDX`, `N_COAGPAIR`, `MODEFRM_COAGPAIR`, `MODETOO_COAGPAIR` to `mam4_jax/data.py`.
2. Add `instrumented-coag-only` capture mode (`scripts/capture_reference.py`).
3. Run capture → produce `tests/reference/per_process_coag/*.npz` (7 fixtures, 60 steps each).
4. Implement `_mam_coag_1subarea` in `mam4_jax/processes/amicphys.py`; replace the no-op stub; wire into `_amicphys_1subarea_clear` call site.
5. Add `test_orchestration_coag_only_matches_fortran` to `tests/test_amicphys.py`.
6. Generate `docs/figures/coag_orchestration_residuals.png` (per-mode number trajectories on top, per-record rel-err for 33 aerosol slots on bottom).
7. Docs sweep: PROGRESS PR-G3 entry; PLANS 5g.PR-G3 done + M3.6 complete; FEATURES coag fully ported; SCHEMA `per_process_coag/` section; REFERENCE_BUILD `instrumented-coag-only` row; plan archived here.

## Verification

- `python -m pytest tests/test_amicphys.py::test_orchestration_coag_only_matches_fortran -v` → passes.
- Worst rel-err across 33 aerosol slots × 60 timesteps: **4.1e-13** (7 orders below ADR-003's 1e-6 budget).
- Full suite: **57/57 green** (56 + 1 new).
- `python scripts/plot_coag_orchestration_residuals.py` regenerates the figure.

## What this PR does NOT do

- No driver-level gas-chem stub in JAX (M4 work — moves into the time loop).
- No `qaer_del_coag_in` accumulation for pcarbon aging (pcarbon aging itself is out of M3.6 scope; capture skips it).
- No M4 work (operator-splitting loop, JAX driver, NetCDF output, convergence sweep).
