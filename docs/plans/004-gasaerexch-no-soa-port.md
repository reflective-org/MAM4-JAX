# Plan 004 — M3.6 PR-D: port `mam_gasaerexch_1subarea` (H₂SO₄ solver, no SOA)

> **Status:** approved 2026-05-20.

---

## Context

M3.6 PR-C (PR #16, merged) added the state-dict ↔ amicphys-local-view unpacking and wired `_mam_rename_1subarea` through the orchestration. PR-D is the first sub-process port that injects real physics through that scaffold: the H₂SO₄ analytical solver path of `mam_gasaerexch_1subarea` (`modal_aero_amicphys.F90:3279-3584`).

**Scope deliberately narrow.** The original `PLANS.md` listed `mam_gasaerexch_1subarea` at "~305 LOC" but didn't account for `mam_soaexch_1subarea` (~330 LOC, called from inside it) or `gas_aer_uptkrates_1box1gas` (~148 LOC, leaf helper). Owner-approved split (2026-05-20): 5 sub-PRs in M3.6 total. PR-D covers gasaerexch's analytical solver only; PR-E will port `mam_soaexch_1subarea` separately.

## Key technical decisions

1. **Skip SOA in the Fortran fixture, not just JAX.** If JAX skips `mam_soaexch_1subarea` while Fortran's gasaerexch still calls it, the SOA gas tracer + SOA aerosol mass column diverge from the start. Cleanest fix: add `scripts/patches/gasaerexch_skip_soaexch.patch` replacing the SOA call with a no-op, applied when capturing the `instrumented-gasaerexch-only` fixture. JAX and Fortran then agree 1:1.
2. **Skip pcarbon-aging in the fixture too.** `mam_pcarbon_aging_1subarea` (called unconditionally inside `mam_amicphys_1subarea_clear` at line 2555) transfers gasaerexch-deposited so4 mass from pcarbon to accum. Without skipping it, the accum.so4 + pcarbon.so4 tracers diverge between Fortran and JAX. Patch: `scripts/patches/skip_pcarbon_aging.patch`.
3. **Add a post-writeback dump.** The existing `amicphys_after` dump records `q` BEFORE the driver's vmr→mmr writeback at `driver.F90:1325`, so it always equals `amicphys_before.q` for any sub-process operating in vmr space. New patch `amicphys_after_writeback.patch` adds a sibling dump AFTER the writeback — that's where gasaerexch's q-space changes actually appear.

## Subtasks

Each ≈ one commit; single PR titled `M3.6 (PR-D): port mam_gasaerexch_1subarea (H₂SO₄ solver, no SOA)`.

1. **Leaf helpers + constants** (`mam4_jax/processes/amicphys.py`, `mam4_jax/data.py`):
   - Port `_mean_molecular_speed(T, MW)` and `_gas_diffusivity(T, p_atm, MW, vm)`.
   - Extend `scripts/patches/amicphys_init_dump.patch` to also capture `vmdry`, `mw_gas`, `vol_molar_gas`, `accom_coef_gas`, plus `mwdry` and `adv_mass` (for the driver-side mmr↔vmr conversion).
   - Add `VMDRY`, `MW_GAS`, `VOL_MOLAR_GAS`, `ACCOM_COEF_GAS`, `MWDRY`, `ADV_MASS`, `MMR_TO_VMR`, `VMR_TO_MMR` to `data.py`. Parity test in `tests/test_scaffolding.py`.

2. **`gas_aer_uptkrates_1box1gas`** (~150 LOC). Two-point Gauss-Hermite quadrature on the Fuchs-Sutugin uptake kernel over the log-normal mode size distribution. Hard-code `XGHQ2` / `WGHQ2` (Fortran defaults from `physconst.F90:237-238`). Batch-friendly via standard JAX broadcasting.

3. **Fortran-side overlays for a 1:1 validation surface**:
   - `scripts/patches/gasaerexch_skip_soaexch.patch` — replace the `call mam_soaexch_1subarea(...)` at line 3430 with `qgas_avg(1:nsoa) = qgas_cur(1:nsoa)` (preserves the qgas_avg assignment that soaexch did).
   - `scripts/patches/skip_pcarbon_aging.patch` — remove the `call mam_pcarbon_aging_1subarea(...)` block at line 2555 in `mam_amicphys_1subarea_clear`.
   - `scripts/patches/amicphys_after_writeback.patch` — new dump tag `amicphys_after_writeback` after the driver writeback at `driver.F90:1325`.
   - `scripts/build_reference.sh` gains a `--skip-soaexch` flag (applies both gasaerexch and pcarbon-aging skip patches; they're paired since both must be off for the PR-D validation).
   - `scripts/capture_reference.py`: extend `DUMP_TAGS` with `amicphys_after_writeback`; new `--mode instrumented-gasaerexch-only`.

4. **JAX gasaerexch body port** (~150 LOC). Replace the no-op `_mam_gasaerexch_1subarea` stub with:
   - Stage A: compute per-gas diffusivity, mean speed, free path, uptake rates per mode (Gauss-Hermite via the new helper). SOA uptake = 0.81 × H2SO4 uptake (cam5.1.00 convention, Fortran line 3407).
   - Stage B: analytical solver for H₂SO4 (lines 3511-3565). Three numerical branches (`tmp_kxt > 0.001` → `exp(-kxt)`; `<= 0.001` → Taylor; `< 1e-20` → no aer update). `qgas_netprod_otrproc[h2so4] = 1e-16 mol/mol/s` hard-coded (matches the driver's stub at `driver.F90:1248` for `mdo_gaschem=0`).
   - Stage C: pack back into qgas / qaer. Skip NH4 limit (`igas_nh3 < 0` in MAM4-MOM). Skip RK4 (default `nonsoa_rk4 = false`). Skip soaexch (PR-E).
   - Wire into `_mam_amicphys_1subarea_clear` via the existing `mdo_gasaerexch` toggle.

5. **New capture mode** `instrumented-gasaerexch-only` writes to `tests/reference/per_process_gasaerexch_only/`. Re-capture ALL existing instrumented fixtures since the new `amicphys_after_writeback` tag is added to `DUMP_TAGS`.

6. **Tests** (`tests/test_amicphys.py`):
   - New `test_orchestration_gasaerexch_only_matches_fortran`. Validates `q` / `qqcw` at 1e-6 rel-err against `amicphys_after_writeback.npz`. Size fields use 1e-3 tolerance (Fortran's `update_aerosol_props` re-runs wateruptake inside the cond sub-stepping loop — Phase A doesn't implement that).
   - Drop the now-obsolete PR-C tripwire `test_orchestration_with_stubs_matches_rename_only_fortran` (no longer accurate post-wiring).

7. **Residual plot** → `docs/figures/gasaerexch_residuals.png`. Two-panel matplotlib figure: H₂SO₄ gas + so4 mass per mode time series (top), per-(timestep, tracer) rel-err vs. ADR-003 1e-6 (bottom). **Flag in chat when generating** per owner's request.

8. **Docs** (rule #5): `PROGRESS.md`, `PLANS.md` (mark 5d done), `SCHEMA.md` (new fixture directory), `REFERENCE_BUILD.md` (new capture mode row), `FEATURES.md` (gasaerexch row).

## Critical files

To **create**:
- `scripts/patches/gasaerexch_skip_soaexch.patch`
- `scripts/patches/skip_pcarbon_aging.patch`
- `scripts/patches/amicphys_after_writeback.patch`
- `scripts/plot_gasaerexch_residuals.py`
- `tests/reference/per_process_gasaerexch_only/*.npz`
- `tests/reference/per_process*/amicphys_after_writeback.npz` (re-captured)
- `docs/figures/gasaerexch_residuals.png`
- `docs/plans/004-gasaerexch-no-soa-port.md` (this file)

To **modify**:
- `scripts/patches/amicphys_init_dump.patch` (extend with mw_gas / vol_molar_gas / accom_coef_gas / vmdry / mwdry / adv_mass)
- `scripts/build_reference.sh` (new `--skip-soaexch` flag, apply `amicphys_after_writeback.patch` always with `--instrumented`)
- `scripts/capture_reference.py` (new mode + parser updates)
- `mam4_jax/data.py` (new constants)
- `mam4_jax/processes/amicphys.py` (helpers + gasaerexch body)
- `tests/test_amicphys.py` (new test + drop obsolete tripwire)
- `tests/test_scaffolding.py` (parity test extensions)
- Docs per rule #5.

## Verification

- `python -m pytest -q` → 49/49 green.
- `python scripts/capture_reference.py --mode instrumented-gasaerexch-only --nstep 60` regenerates the fixture.
- `python scripts/plot_gasaerexch_residuals.py` renders the figure (max rel-err on modified tracers < 1e-6).
- Living docs updated in the same PR.

## Out of scope

- `mam_soaexch_1subarea` (PR-E).
- `mam_gasaerexch_RK4_1subarea` (Fortran's `nonsoa_rk4` is false by default).
- `mam_pcarbon_aging_1subarea` — fundamentally a separate sub-process. Not in M3.6 scope.
- Fortran's `update_aerosol_props` (intra-substep wateruptake re-run) — Phase A; affects size fields only.
- NH4 limit (no NH3 in MAM4-MOM).
- Cloud-borne path (still unreachable at `cldn=0`).
