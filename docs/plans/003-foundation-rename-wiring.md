# Plan 003 — M3.6 PR-C: foundation + wire rename into orchestration

> **Status:** approved 2026-05-20.

---

## Context

After PR-B (PR #15), `_mam_rename_1subarea` is a pure local-view function (operates on `qnum_cur`, `qaer_cur`, `qaer_delsub_grow4rnam`, `qwtr_cur`, `fac_m2v_aer`) but is **not wired** into the orchestration shell. The shell's `_mam_amicphys_1subarea_clear` still skips the rename call because we don't yet have the state-dict ↔ amicphys-local-view unpacking layer.

This PR adds that layer and wires rename through. Validates the wiring via a new single-toggle Fortran capture (`mdo_rename=1, others=0`). No new physics ported — PR-D through PR-G use this same scaffold.

**Scope change vs. original `PLANS.md`.** The original M3 plan listed 4 remaining sub-routine ports (rename, gasaerexch, newnuc, coag). Reading the source for PR-C planning revealed that `mam_gasaerexch_1subarea` (~306 LOC) calls `mam_soaexch_1subarea` (~330 LOC) plus `gas_aer_uptkrates_1box1gas` (~148 LOC) — collectively ~780 LOC, too large for one reviewable PR. Recommended split: **5 PRs to finish M3.6** (this foundation PR + gasaerexch + soaexch + newnuc + coag) instead of 4. Owner approved 2026-05-20.

## Why the unpacking is non-trivial

Fortran's `mam_amicphys_1gridcell` (modal_aero_amicphys.F90:1331-1369) unpacks `q[pcnst] → (qgas, qaer, qnum, qwtr)` via two indirections:

- **Amicphys-internal species ordering** (`lmap_aer`, `lmap_num`) — different from `modal_aero_data`'s `lmassptr_amode`. Amicphys orders species by its own `name_aerpfx` list, set up in `modal_aero_amicphys_init` (line 5599 onwards).
- **Per-species unit-conversion factors** `fcvt_aer(iaer)`, `fcvt_gas(igas)`, `fcvt_num`, `fcvt_wtr` — kg/kg ↔ mol/mol or kg/kg ↔ #/kmol. Also set in init.

These are module-private to `modal_aero_amicphys`, so the dump has to live inside that module (not in `mam4_dump_state`).

Additionally, `lmap_*` values are **gas_pcnst-relative**, not pcnst-relative. The conversion is `pcnst_idx = lmap_value + loffset` where `loffset` is the chemistry module's offset (5 for the box-model build, confirmed by the dump).

## Subtasks

Each ≈ one commit; bundle ships as one PR titled `M3.6 (PR-C): wire rename into the orchestration shell`.

1. **Capture amicphys init tables.** New `scripts/patches/amicphys_init_dump.patch` injects a `block` near the end of `modal_aero_amicphys_init` (just before `call m_a_amicphys_init_history`, line 6101) that writes `mam4_amicphys_init.txt` with: `loffset`, `ngas`, `naer`, `max_gas`, `max_aer`, `lmap_gas`, `lmap_num`, `lmap_numcw`, `lmap_aer`, `lmap_aercw`, `fcvt_gas`, `fcvt_aer`, `fcvt_num`, `fcvt_wtr`. `scripts/build_reference.sh` applies the patch alongside the other instrumentation overlays. `scripts/capture_reference.py::_read_amicphys_init` parses the new file and merges into `tests/reference/indices/reference.npz` (single .npz, multiple keys). Also dumps `pcnst_lmap_*` variants (loffset-adjusted, 0-based, -1 sentinel) for direct consumer use. → **verify:** `pcnst_lmap_num` matches the existing `numptr_amode` field (cross-check).

2. **Add tables to `mam4_jax/data.py`** as new 0-based constants: `AMICPHYS_NGAS`, `AMICPHYS_NAER`, `LMAP_GAS`, `LMAP_NUM`, `LMAP_NUMCW`, `LMAP_AER`, `LMAP_AERCW`, `FCVT_GAS`, `FCVT_AER`, `FCVT_NUM`, `FCVT_WTR`. Parity test in `tests/test_scaffolding.py` against the `.npz`. → **verify:** new parity test passes; existing 47 tests stay green.

3. **State-dict ↔ amicphys-local-view unpacking in `mam4_jax/processes/amicphys.py`.** Two pure helpers:
   - `_unpack_state_to_amicphys_view(state) → (qgas, qaer, qnum, qwtr)` via `LMAP_*` indexing + `FCVT_*` scaling. JAX-pure (gather via constant indices).
   - `_repack_amicphys_view_to_state(state, qgas, qaer, qnum, qwtr) → state'` inverse.
   - Round-trip test in `tests/test_amicphys.py`: unpacking a captured `amicphys_before.q` → amicphys view → repacking back to `q` is bit-exact.

4. **Wire rename into `_mam_amicphys_1subarea_clear`.** The handler becomes:
   ```python
   qgas, qaer, qnum, qwtr = unpack(state)
   qaer_sv1 = qaer
   if mdo_gasaerexch: ...   # still a no-op stub; PR-D will fill in
   qaer_delsub_grow4rnam = qaer - qaer_sv1   # zero while gasaerexch is a stub
   if mdo_rename:
       qnum, qaer, qwtr = _mam_rename_1subarea(
           qnum, qaer, qaer_delsub_grow4rnam, qwtr, FAC_M2V_AER)
   if mdo_newnuc / mdo_coag: ...   # stubs
   return repack(state, qgas, qaer, qnum, qwtr)
   ```

5. **Single-toggle capture.** New `--mode instrumented-rename-only` in `scripts/capture_reference.py`. Namelist with `mdo_gasaerexch=mdo_newnuc=mdo_coag=0`, `mdo_rename=1`. Output to `tests/reference/per_process_rename_only/{calcsize,wateruptake,amicphys,rename}_{before,after}.npz`. → **verify:** in the rename-only capture, `amicphys_after.q` differs from `amicphys_before.q` only on Aitken/accum number + mass tracers (calcsize and wateruptake snapshots unchanged across configurations).

6. **Tests** (`tests/test_amicphys.py`):
   - New `test_orchestration_rename_only_matches_fortran`: JAX `amicphys(state, mdo_rename=1, others=0)` reproduces the new rename-only capture's `amicphys_after.q` at 1e-6 across 60 steps.
   - Update `test_amicphys_all_on_with_stubs_is_passthrough` → rename to `test_orchestration_with_stubs_matches_rename_only_fortran`. With all four `mdo_*=1` but three sub-processes still stubs, the only active process is rename, so the JAX result matches the rename-only Fortran capture (not the full-physics `amicphys_after.npz`).
   - `test_amicphys_all_off_is_passthrough` unchanged.
   - `test_rename_*` (PR-B local-view tests) unchanged.

7. **Doc updates** (rule #5).
   - `docs/PROGRESS.md`: M3.6 PR-C entry on top.
   - `docs/PLANS.md`: amicphys plan restructured to 5 sub-PRs (foundation + gasaerexch + soaexch + newnuc + coag). 5b stays "done"; new items 5c (foundation, this PR), 5d (gasaerexch), 5e (soaexch), 5f (newnuc), 5g (coag).
   - `tests/reference/SCHEMA.md`: new `per_process_rename_only/` section; extended `indices/reference.npz` schema with amicphys init keys.
   - `docs/REFERENCE_BUILD.md`: new `instrumented-rename-only` row in the Capture modes table.
   - `docs/FEATURES.md`: rename row updated to "wired into orchestration via the new unpacking layer".

## Critical files

To **create**:
- `scripts/patches/amicphys_init_dump.patch`
- `tests/reference/per_process_rename_only/{calcsize,wateruptake,amicphys,rename}_{before,after}.npz`
- `docs/plans/003-foundation-rename-wiring.md` (this file)

To **modify**:
- `scripts/build_reference.sh` (apply the new patch with `--instrumented`)
- `scripts/capture_reference.py` (parser + new `--mode`)
- `mam4_jax/data.py` (new constants)
- `mam4_jax/processes/amicphys.py` (unpacking + wiring)
- `tests/test_scaffolding.py` (parity test)
- `tests/test_amicphys.py` (new + updated tests)
- `tests/reference/indices/reference.npz` (regenerated with amicphys init keys)
- Docs per rule #5.

## Verification

- All existing 47 tests still pass (rename's PR-B local-view function is unchanged).
- New parity test for amicphys init constants passes.
- New `test_orchestration_rename_only_matches_fortran` passes at < 1e-6 across 60 steps.
- The renamed `test_orchestration_with_stubs_matches_rename_only_fortran` passes.
- Living docs updated in the same PR.

## Out of scope

- Cloudy sub-area path (still unreachable at `cldn=0`).
- Gas-phase chemistry mapping (gases aren't touched by rename; will be wired in PR-D alongside gasaerexch).
- Tendency accumulation (`qsub_tendaa`) — only needed when M4 wraps amicphys in a time loop.
- `mam_pcarbon_aging_1subarea` (separate sub-process called outside the four `mdo_*`-gated ones).
- New physics. PR-D onwards.
