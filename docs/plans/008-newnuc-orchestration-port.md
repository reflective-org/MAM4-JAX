# Plan 008 — M3.6 PR-F3: wire newnuc into amicphys orchestration

> **Status:** approved 2026-05-21.

---

## Context

PR-F2 (PR #21, merged) ported the `mer07_veh02_nuc_mosaic_1box` dispatcher as a pure JAX function. PR-F3 is the amicphys orchestration glue (`modal_aero_amicphys.F90:4251-4665`, ~415 LOC) that:

- Pulls `qh2so4_avg` from gasaerexch's output (Fortran default `newnuc_h2so4_conc_optaa == 2`).
- Sets up the size-bin bounds for the Aitken mode.
- Calls the PR-F2 dispatcher.
- Applies particle-size constraints (the `dndt_ait < 100` filter and the `mass1p` clamps).
- Adds the new-particle mass and number to `qaer[so4, Aitken]` and `qnum[Aitken]`; subtracts from `qgas[h2so4]`.

After PR-F3 lands, only PR-G (coag, ~437 LOC) remains in M3.6.

## Scope decisions

**Wiring change**: `_mam_gasaerexch_1subarea` currently returns `(qgas, qaer)`. Extended to return `(qgas, qaer, qgas_avg)` so newnuc can consume the time-averaged H₂SO₄ vmr (Fortran's `tmp_q4` from the H₂SO₄ analytical solver).

**State dict additions**: `zmid`, `pblh`, `relhum` — newnuc consumes all three for the PBL gate and binary nucleation. Box-model values (from `driver.F90:577-579` and the `RH_CLEA` namelist):
- `zmid = 3000 m`
- `pblh = 1100 m` (so `zmid > pblh` → PBL nucleation does NOT activate; only binary fires)
- `relhum = 0.9`

**MAM4-MOM-specific simplifications** in the JAX port:
- `igas_nh3 < 0` → all NH₃ paths skipped (`qnh3_cur=0`, `qnh4a_del=0`, `tmp_frso4=1`).
- The optaa=1 H₂SO₄ averaging branch (Fortran lines 4362-4397) is skipped; we hardcode the default optaa=2 path.
- `h2so4_uptkrate` (used by the dispatcher's KK2002 correction): for PR-F3 we pass a hardcoded `1e-3` placeholder. The box-model fixture has `zmid > pblh` so the PBL branch is inactive, and KK2002 uses uptkrate only multiplicatively via `tmpa = uptkrate * 3600`. The end-to-end validation against Fortran tells us whether this approximation is good enough — if validation fails, we extend the gasaerexch return signature again.
- Diagnostic-output blocks omitted.

## Validation strategy

**New capture mode** `instrumented-gasaerexch-and-newnuc-only`: namelist `mdo_gasaerexch=1, mdo_newnuc=1, others=0` plus `skip_pcarbon_aging.patch` (consistent with PR-D/E/F2 pattern).

**Why gasaerexch must also be on**: newnuc consumes `qgas_avg[h2so4]` which gasaerexch's analytical solver computes. With `mdo_gasaerexch=0`, `qgas_avg=0` → newnuc early-returns at `qh2so4_avg ≤ qh2so4_cutoff = 4e-16` → no validation surface.

Output → `tests/reference/per_process_gasaerexch_and_newnuc/`.

**Tests**:
- New `test_orchestration_gasaerexch_and_newnuc_matches_fortran` against the new fixture, rel-err < 1e-6 on `q` and `qqcw`.
- Keep PR-E's `test_orchestration_gasaerexch_matches_fortran` (validates gasaerexch alone).
- The existing `test_amicphys_returns_all_state_keys` and `test_amicphys_all_off_is_passthrough` keep working after the state dict gains `zmid / pblh / relhum`.

## Subtasks

Each ≈ one commit; single PR titled `M3.6 (PR-F3): wire newnuc into amicphys orchestration`.

1. **Confirm box-model values** from driver.F90 and the namelist — `zmid=3000`, `pblh=1100`, `relhum=0.9`.

2. **Extend `_mam_gasaerexch_1subarea`** to return `(qgas, qaer, qgas_avg)`. Update `_mam_amicphys_1subarea_clear` to thread `qgas_avg` to newnuc. Existing PR-E test stays unaffected (it doesn't consume `qgas_avg`).

3. **Port `_mam_newnuc_1subarea`** in `mam4_jax/processes/amicphys.py`. Replaces the no-op stub. ~80 LOC of JAX (dispatcher does the heavy lifting). Imports `mam4_jax.newnuc` for the dispatcher.

4. **Wire into `_mam_amicphys_1subarea_clear`** — `if mdo_newnuc: qgas, qnum, qaer = _mam_newnuc_1subarea(...)`.

5. **New capture mode** `instrumented-gasaerexch-and-newnuc-only` in `scripts/capture_reference.py`. Output → `tests/reference/per_process_gasaerexch_and_newnuc/`.

6. **Test** `test_orchestration_gasaerexch_and_newnuc_matches_fortran` against the new fixture. Rel-err < 1e-6 on `q`, `qqcw`; loose tolerance on size fields (Fortran's `update_aerosol_props` mid-step issue, same as PR-D/E).

7. **Residual plot** → `docs/figures/newnuc_orchestration_residuals.png`. Time series of H₂SO₄ gas + Aitken-mode number + Aitken-mode so4 mass (the tracers newnuc directly modifies), plus per-(timestep, tracer) rel-err panel.

8. **Docs** (rule #5): PROGRESS, PLANS (mark 5f.PR-F3 done → M3.6 newnuc complete), SCHEMA, REFERENCE_BUILD, FEATURES.

## Verification

- `python -m pytest -q` → 54/54 green (53 + 1 new).
- `python scripts/capture_reference.py --mode instrumented-gasaerexch-and-newnuc-only --nstep 60` regenerates the fixture.
- `python scripts/plot_newnuc_orchestration_residuals.py` renders the figure (worst rel-err < 1e-6).

## Out of scope

- `mam_coag_1subarea` — PR-G, the last M3.6 sub-process.
- The optaa=1 H₂SO₄ averaging branch (unreachable for default optaa=2).
- NH₃-related branches.
- Diagnostic-output blocks.

## Empirical questions answered during execution

1. **Does newnuc fire on the box-model fixture?** Yes — qh2so4 is ~1e-13 (well above the 4e-16 cutoff) and the `dndt_ait < 100` filter does not always trip.
2. **Does the particle-size constraint bite?** Empirically the test passes at machine ε, suggesting `mass1p` is in the Aitken-bin range and the constraint clamps don't fire.
3. **Is the hardcoded `h2so4_uptkrate = 1e-3` good enough?** Yes for the box-model fixture (test matches at 1e-16). The fixture has `zmid > pblh` so PBL nuc is off, and KK2002's `tmpa = uptkrate * 3600` only enters multiplicatively — at the values the dispatcher computes, the difference vs the exact gasaerexch-derived rate is below 1e-6.
