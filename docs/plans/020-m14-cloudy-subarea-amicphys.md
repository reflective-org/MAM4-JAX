# Plan 020 — M14: Cloudy-subarea amicphys orchestration

**Status:** in progress (2026-06-05). PR-A landed the subarea split mechanics; PR-B onwards ports the cloudy-subarea physics.
**Branch:** `diffrax-cloud` (sub-PRs land here).
**GitHub milestone:** [M14 #9](https://github.com/reflective-org/MAM4-JAX/milestone/9).
**Antecedent:** Plan 019 (M8) §7 sketches M14 as the structural follow-up; this doc supersedes that sketch with the concrete sub-PR breakdown identified by PR-K3b's substep diagnostic.

---

## 1. Scope

Port Fortran's `mam_amicphys_1subarea_cloudy` (`modal_aero_amicphys.F90:1504-2059`, ~556 LOC) and the gridcell ↔ subarea split that drives it. This closes the M8 cloudchem trajectory bar at `cldn > 0`: PR-K3 wired cloudchem into the driver and verified the per-process port at machine ε, but the cumulative trajectory at `cldn = 0.5` diverged ~18× from Fortran because JAX's `_mam_amicphys_1gridcell` fed gridcell-aggregate state to `_mam_amicphys_1subarea_clear` instead of clear-subarea-concentrated state, and the cloudy-subarea path didn't exist at all.

### Decision log

| Decision | Value | Why |
| --- | --- | --- |
| M14 is a multi-PR milestone (not a single PR) | PR-A scaffolding + PR-B physics + cleanup | Per CLAUDE.md rule #2 (commit-sized PRs). A 556-LOC physics port + 130-LOC structural change in one PR is unreviewable. |
| Cloudy stub in PR-A returns input unchanged | Owner-flagged as "intermediate scaffolding, not the final algorithm" | The user explicitly rejected "passthrough as final"; the stub is documented as temporary. Final algorithm = PR-B's real port. |
| Cloudy sub-process subset | gasaerexch + rename; **skip** gasaerexch_RK4 and pcarbon_aging | Project-wide already-deferred (DEFERRED.md). Cloudchem fixture has these patched out anyway. |
| Validation order | per-process (cloudy-only fixture) → trajectory (existing cloudchem fixture) | Same M3.6 pattern: capture single-toggle fixture first, validate at machine ε, then enable full physics. |
| Validation bar | ADR-015 3 % / 24 h / dt ≤ 5 s (target) | Same as ADR-015 for the rest of diffrax-branch physics. Per PR-K3's owner direction, ≤ 1e-4 acceptable if measurable. |

---

## 2. Why M14 is needed (summary of the PR-K3b finding)

PR-K3b's substep diagnostic at step 39 of the cloudchem fixture:

```
Fortran trajectory:
  amicphys_before  q[17]: 6.27e+04   ← gridcell mmr (post-cloudchem)
  amicphys_after   q[17]: 6.27e+04   ← gridcell mmr (unchanged; q not modified inside amicphys)
  amicphys_after_writeback: 1.34e+06 ← gridcell mmr (post vmr→q + subarea aggregation)

Fortran rename_before qnum_cur[accum]: 3.61e+06   ← clear-subarea-concentrated value
JAX rename_before qnum[accum] (PR-K3): 1.80e+06   ← gridcell value (no subarea split)

Ratio: Fortran is 2× JAX, exactly 1/(1-cldn) for cldn=0.5.
```

The diagnostic localized the bug to: JAX's `_mam_amicphys_1gridcell` was a no-op wrapper that called `_mam_amicphys_1subarea_clear` directly on the gridcell state, assuming `cldn = 0`. For `cldn > 0`, the clear-subarea-internal sub-process rates (coag ~ n², rename trigger, etc.) needed concentrated qnum to match Fortran.

---

## 3. Sub-PR breakdown

### PR-M14-A — Subarea split mechanics + cloudy stub (✅ landed in PR #57)

**Status:** ✅ Merged 2026-06-05.

**What landed**:
- `_PCNST_INTERSTITIAL_MASK` (module-level): True for slots in `LMAP_NUM ∪ valid LMAP_AER` (interstitial aerosol number + mass), False elsewhere.
- `_mam_amicphys_1subarea_cloudy_stub`: documented placeholder returning input unchanged. Correctness contract documented (depends on caller pre-zeroing interstitial in cloudy state).
- `_mam_amicphys_1gridcell` rewrites: builds clear-subarea state (interstitial / (1-cldn), qqcw=0, qaerwat / (1-cldn)) and cloudy-subarea state (interstitial=0, qqcw / cldn, qaerwat=0). Calls `_mam_amicphys_1subarea_clear` on clear; calls cloudy stub on cloudy. Aggregates: `gridcell = (1-cldn)·clear + cldn·cloudy` for q, qqcw, qaerwat.
- Two new tests in `tests/test_amicphys.py`:
  - `test_amicphys_subarea_split_cldn_zero_is_identity`: at cldn=0 the split path must equal direct `_1subarea_clear` byte-for-byte.
  - `test_amicphys_subarea_split_cldn_nonzero_conservation`: at cldn=0.5 with all mdo_*=0, gridcell roundtrip preserves input bit-exactly.

**Empirical result**: cldn=0 bit-exact (77 + 2 new tests = 79 pass byte-identical). cldn=0.5 per-step rel-err on accum-number `q[17]` improved 0.96 → 0.95 from correctly-scaled coag rate; trajectory bar still ~18× — closure waits on PR-B.

### PR-M14-B — Port `mam_amicphys_1subarea_cloudy` physics

**Scope**:
- New `_mam_amicphys_1subarea_cloudy(state, ...)` function in `mam4_jax/processes/amicphys.py`. Replaces the stub.
- Sub-process composition (per Fortran source survey of `modal_aero_amicphys.F90:1504-2059`):
  - **gasaerexch**: yes (reuse existing `_mam_gasaerexch_1subarea`). Cloud-borne aerosols are active participants — qaer/qqcw both feed in. Existing port was developed assuming `qqcw = 0`; verify it produces correct output when qqcw is non-zero or extend.
  - **rename**: yes (reuse existing `_mam_rename_1subarea`). Same caveat.
  - **gasaerexch_RK4**: **skip** (`nonsoa_rk4 = .false.` in box-model build per `cambox_config.cpp.in`; consistent with M3.6's RK4-out-of-scope decision).
  - **pcarbon_aging**: **skip** (deferred project-wide per `DEFERRED.md`; cloudchem fixture is captured with `skip_pcarbon_aging.patch`).
  - **newnuc**, **coag**: **not called** in cloudy (per Fortran source).
- Sub-area state-dict assembly needs `qqcw` to actually flow through — current JAX `_unpack_state_to_amicphys_view` only reads from `state["q"]` + `state["qaerwat"]`. Extend to also read from `state["qqcw"]` and produce a `qaercw` / `qnumcw` view, then route through the cloudy path's sub-process calls.

**Capture + validation**:
- New Fortran fixture: `tests/reference/per_process_cloudy_only/` — captured at `mdo_cloudchem = 0` (drop cloudchem from the chain) + `cldn = 0.5` + the existing `instrumented-cloudchem-only` build flags otherwise. This isolates cloudy-subarea amicphys behavior from cloudchem itself.
  - Need a new patch / namelist mode in `scripts/capture_reference.py`: `instrumented-cloudy-only` (cldn = 0.5 but mdo_cloudchem = 0).
- New per-process test in `tests/test_amicphys.py`: validate `_mam_amicphys_1subarea_cloudy` against the cloudy-only fixture at ADR-003's `1e-6` (machine ε expected for a structural port that doesn't introduce new ODE solvers).

**Expected effort**: ~3-5 days. Bulk of M14's work.

### PR-M14-C — Re-validate cloudchem trajectory + close M8

**Scope**:
- Trajectory test `test_run_timesteps_with_cloudchem_trajectory_diagnostic` becomes asserted (not diagnostic-only). Bar: ADR-015 3 %; tighter if measured.
- FEATURES.md flip: "Sulfur chemistry beyond stubs" → "cloudchem_simple ported; explicit-kinetics aqueous chemistry remains out of scope."
- PLANS.md M8 → done. PLANS.md M14 → done.
- PROGRESS.md entries for M8 and M14 closure.
- M8 milestone closed on GitHub; M14 milestone closed.

**Expected effort**: ~1 day (mostly bookkeeping; physics validation should hold from PR-M14-B).

---

## 4. Risks / known unknowns (PR-B specifically)

1. **`qqcw` propagation through the unpack/repack layer.** The existing `_unpack_state_to_amicphys_view` doesn't surface `qqcw` to amicphys-internal arrays — it ignores it. PR-M14-B needs to extend the unpack to produce a `qaercw` / `qnumcw` view from `state["qqcw"]`, route it through the cloudy sub-process calls, and repack any modifications back into `state["qqcw"]`.

2. **Hidden dependencies on `qqcw = 0` inside existing sub-process ports.** The clear-subarea-tested `_mam_gasaerexch_1subarea` and `_mam_rename_1subarea` were developed against fixtures with `qqcw = 0` everywhere. They might (a) gracefully no-op the cloud-borne paths because operation-on-zero is harmless, or (b) silently produce wrong output because a non-zero `qqcw` exposes a code path that hadn't been exercised. PR-M14-B's per-process validation (capturing single-toggle cloudy-only fixture) is the primary tool for finding any such bugs.

3. **PR-K3b's leftover concern.** Even with the split (PR-M14-A) and a real cloudy port (PR-M14-B), there's a chance that JAX's `_mam_amicphys_1subarea_clear` itself has a bug that only manifests on concentrated-qnum input. PR-M14-A's measurement showed the per-step `q[17]` rel-err dropped from 0.96 to 0.95 (~19 % improvement) but the trajectory rel-err was essentially unchanged. If PR-M14-B's cloudy port doesn't close the remaining gap, there's a deeper issue to investigate — possibly a clear-vs-cloudy code-path divergence inside Fortran's amicphys that doesn't fall out of "same sub-routines, different state."

4. **New fixture wall-time.** Per-process cloudy-only capture is fast (60 steps × dt=30s). End-to-end 24 h sweep extensions are the long ones (~hours per dt × 4 dt). May defer the 24 h sweep to a PR-M14-D rather than bundling into M14-C.

---

## 5. Open questions for PR-B planning

1. **Cloudy unpack/repack design**: extend `_unpack_state_to_amicphys_view` to surface `qaercw`/`qnumcw` (modify signature, affecting all clear-subarea callers), or add a parallel `_unpack_state_to_amicphys_view_cloudy` function? **Recommendation**: parallel function — minimizes blast radius and matches the "subarea-specific" decomposition.
2. **`fac_m2v_aer` and other amicphys-init constants**: do they differ between clear and cloudy contexts? **Recommendation**: assume no — they're set at amicphys init from `modal_aero_data` and are subarea-agnostic. Verify in PR-B against the new fixture.
3. **Validation bar**: per-process at `1e-6` (ADR-003) or `3 %` (ADR-015)? **Recommendation**: per-process at `1e-6` — cloudy is algebraic + same sub-routines as clear, so machine ε is expected (just like PR-K2's cloudchem). The 3 % bar applies to the cumulative trajectory in PR-M14-C.

---

## 6. Pointers

- `docs/plans/019-m8-cloudchem.md` — antecedent M8 plan, §7 sketches M14.
- `docs/KEY_DECISIONS.md` ADR-013 (dual-branch), ADR-015 (3 % bar), ADR-016 (merge-back).
- `mam4-original-src-code/e3sm_src_modified/modal_aero_amicphys.F90:1504-2059` — Fortran source for PR-B.
- GitHub: [M14 milestone #9](https://github.com/reflective-org/MAM4-JAX/milestone/9).
- PR-A: [#57](https://github.com/reflective-org/MAM4-JAX/pull/57).
