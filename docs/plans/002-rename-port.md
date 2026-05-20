# Plan 002 — M3.6 PR-B: port `_mam_rename_1subarea`

> **Status:** approved 2026-05-20.

---

## Context

M3.6 PR-A (PR #13, merged at `dff389d`) landed the JAX orchestration shell for `modal_aero_amicphys_intr` — the entry point, `_mam_amicphys_1gridcell`, `_mam_amicphys_1subarea_clear`, and four no-op sub-process stubs (`_mam_gasaerexch_1subarea`, `_mam_rename_1subarea`, `_mam_newnuc_1subarea`, `_mam_coag_1subarea`). PR-A's validation was "all-mdo-off passthrough", proven bit-exact against `tests/reference/per_process_amicphys_off/`.

This plan covers **PR-B**, the first of four sub-process ports. Per `docs/PLANS.md` Milestone 3 item 5, the order is rename (PR-B) → gasaerexch (PR-C) → newnuc (PR-D) → coag (PR-E). Rename comes first because it's the smallest (~323 LOC), uses only `erfc` plus algebra (no chemistry), and its only physics input from upstream sub-processes is the simple delta `qaer_delsub_grow4rnam`.

The active Fortran subroutine is `mam_rename_1subarea` (`e3sm_src_modified/modal_aero_amicphys.F90:3923–4246`). The standalone `modal_aero_rename.F90` is dead code in the box-model build (PR #12 finding).

## Key technical observation that shapes validation

Rename consumes `qaer_delsub_grow4rnam` — the change made by `mam_gasaerexch_1subarea` in the same sub-area, computed at Fortran line 2433 (`qaer_delsub_grow4rnam = qaer_cur - qaer_sv1`). In our current JAX orchestration `_mam_gasaerexch_1subarea` is still a no-op stub, so when called through the shell with default `mdo_*=1` toggles the delta passed to rename is exactly zero. Inspecting the Fortran (lines 4106–4110):

```fortran
if (dryvol_t_new .le. dryvol_smallest(mfrm)) cycle mainloop1_ipair
...
if (rename_method_optaa .ne. 40) then
   if (dryvol_t_del .le. 1.0e-6*dryvol_t_oldbnd) cycle mainloop1_ipair
end if
```

Zero `qaer_delsub_grow4rnam` → zero `dryvol_t_del` → both guards trigger `cycle` → rename is a structural no-op. **No matter how faithfully we port rename in PR-B, it is provably inactive within the PR-B JAX orchestration.** That's not a useful validation surface.

The validation strategy therefore bypasses the JAX orchestration: we capture rename's inputs and outputs from the **full-physics Fortran run** (the existing `--mode instrumented` build, where `mdo_*=1` and gasaerexch actually generates growth), call `_mam_rename_1subarea` on the captured "before" snapshot, and diff against the captured "after" snapshot. This isolates rename's contribution from the rest of amicphys for the first time on the canonical fixture and answers an open question: does rename even fire in this configuration?

PR-C will revisit rename's integration once gasaerexch is real — the end-to-end orchestration test then becomes a true cross-check against the existing `amicphys_{before,after}.npz` captures.

## Subtasks

Each ≈ one commit; the bundle ships as one PR titled `M3.6 (PR-B): port _mam_rename_1subarea`.

1. **Extend the instrumentation overlay.** Add two new dump tags (`rename_before`, `rename_after`) to `scripts/patches/mam4_dump_state.F90` and extend `scripts/patches/driver_instrumentation.patch` (or a new sibling patch) with `call dump_snapshot(...)` calls wrapping the rename call at `modal_aero_amicphys.F90:2467`. Each snapshot dumps `mtoo_renamexf` (int array, shape `max_mode`), `qnum_cur` (shape `max_mode`), `qaer_cur` (shape `max_aer, max_mode`), `qaer_delsub_grow4rnam` (same shape), `qwtr_cur` (shape `max_mode`). → **verify:** `python scripts/capture_reference.py --mode instrumented --nstep 60` produces the two new `.bin` files alongside the existing six.

2. **Extend `scripts/capture_reference.py`** to parse the new bins into `tests/reference/per_process/rename_{before,after}.npz`. → **verify:** the `.npz` arrays have the expected dtypes/shapes; new files appear in `git status`.

3. **Port `_mam_rename_1subarea`** in `mam4_jax/processes/amicphys.py`. Direct ~323-LOC translation. Vectorized per (col, level); the pair-list `n = 1..ntot_amode` loop stays Python-level (only `npair = 1` ever fires for the box model because `mtoo_renamexf(nait) = nacc` is the only non-zero entry). Update `_mam_amicphys_1subarea_clear` to thread `qaer_delsub_grow4rnam` between the gasaerexch and rename calls (save `state["q"]_aer_view` before gasaerexch, diff after, pass as a keyword arg into `_mam_rename_1subarea`). Cloud-borne path is **not implemented** in this PR (see "Out of scope"). → **verify:** the existing 45 tests still pass — `test_amicphys_all_off_is_passthrough` and `test_amicphys_all_on_with_stubs_is_passthrough` both still hold because the delta is zero whenever gasaerexch is off or a no-op.

4. **Add `tests/test_rename.py`** with 3 tests:
   - `test_rename_matches_fortran_full_physics`: load `rename_{before,after}.npz`; call `_mam_rename_1subarea` on the "before" snapshot; assert max rel-err < 1e-6 against the "after" snapshot for `qnum_cur` and `qaer_cur` (with `np.allclose(rtol=1e-6, atol=1e-25)` to absorb machine-noise artifacts at exactly-zero slots, consistent with `tests/test_calcsize_transfer.py`).
   - `test_rename_is_noop_with_zero_delta`: structural test confirming that `_mam_rename_1subarea(state, qaer_delsub_grow4rnam=zeros)` returns state unchanged. This validates that the orchestration shell's wiring won't trip rename when gasaerexch is off.
   - `test_rename_activity_on_canonical_fixture`: log-only test reporting how many of the 60 captured timesteps had a non-zero transfer; xfail / skip-style, doesn't enforce a value but produces a diagnostic record we can read in CI output. (Stripped if Pytest's `record_property` proves noisy; keep only the first two as binding.)
   → **verify:** `pytest -q` → 47/47 (was 45 + 2 new, dropping the diagnostic third).

5. **Plot residuals** → `docs/figures/rename_residuals.png`. Two-panel: top, Aitken-mode number transfer over time (JAX vs Fortran), bottom, per-(timestep, mode, slot) rel-err histogram. Reuse the styling from `docs/figures/calcsize_residuals.png`.

6. **Doc updates** in the same PR:
   - `tests/reference/SCHEMA.md`: add `rename_before.npz` / `rename_after.npz` rows under the per-process section.
   - `docs/REFERENCE_BUILD.md`: extend the ADR-012 overlay description with the two new hook points.
   - `docs/PROGRESS.md`: M3.6 PR-B entry on top, summarizing port + the activity finding from subtask 4.
   - `docs/PLANS.md`: mark item 5b done with PR link.
   - `docs/FEATURES.md`: rewrite the rename row — drop "stub … scheduled for M3.6 PR-B", note "ported (validated) inside `amicphys.py` at max rel-err X.YeZ; activity status: <active|no-op-on-this-fixture>".
   - If rename is a structural no-op on this fixture, add an entry to `docs/DEFERRED.md` mirroring the calcsize-transfer entry.

## Critical files

To **create:**
- `tests/test_rename.py`
- `tests/reference/per_process/rename_before.npz`, `rename_after.npz` (generated)
- `docs/figures/rename_residuals.png`
- `docs/plans/002-rename-port.md` (this file)

To **modify:**
- `scripts/patches/driver_instrumentation.patch` (or add `scripts/patches/rename_hook.patch`)
- `scripts/patches/mam4_dump_state.F90`
- `scripts/capture_reference.py`
- `mam4_jax/processes/amicphys.py`
- `tests/reference/SCHEMA.md`
- `docs/REFERENCE_BUILD.md`, `docs/PROGRESS.md`, `docs/PLANS.md`, `docs/FEATURES.md`
- (possibly) `docs/DEFERRED.md`

To **consult** (read-only):
- `modal_aero_amicphys.F90:3923–4246` (the subroutine itself)
- `modal_aero_amicphys.F90:2433` (delta construction site)
- `modal_aero_amicphys.F90:2461–2476` (the call site to be wrapped)
- `modal_aero_amicphys.F90:1940–1941` (the `mtoo_renamexf(nait) = nacc` setup that determines `npair = 1`)
- `box_model_utils/rad_constituents.F90:167-170` (dgnum / sigmag values feeding `factoraa`, `factoryy`, `v2nlorlx`, `v2nhirlx`)

## Verification

**PR acceptance:**
- `python -m pytest -q` → 47/47 green (was 45 + 2 new).
- `python scripts/capture_reference.py --mode instrumented --nstep 60` regenerates `rename_{before,after}.npz` bit-for-bit (modulo compiler version).
- Max rel-err in `qnum_cur`, `qaer_cur` after rename < 1e-6 across all 60 captured timesteps.
- `docs/figures/rename_residuals.png` renders and is committed.
- Living docs updated in the same PR (rule #5).

## Out of scope

- **Cloud-borne path:** the Fortran rename signature has optional args `qnumcw_cur`, `qaercw_cur`, `qaercw_del_grow4rnam` used only when `iscldy_subarea=.true.`. For `cldn=0` (box-model default, `driver.F90:591`) the clear sub-area is the only one ever evaluated, so these args are unused. The JAX port omits them. Adding them would be a follow-up if/when the cloudy sub-area path is ever required.
- **Other amicphys sub-processes:** gasaerexch (PR-C), newnuc (PR-D), coag (PR-E) remain as no-op stubs.
- **`jit`/`vmap` of rename:** Phase-B optimization per ADR-004. Keep PR-B as plain JAX numerics.
