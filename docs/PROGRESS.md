# Progress

A running, append-only log of project milestones. Most-recent entry on top. Update in the same PR that lands the work being recorded.

Each entry: date, short title, links to commits / PRs, one-paragraph summary.

---

## 2026-05-21 — Milestone 3.6 (PR-F3) — Newnuc amicphys orchestration

- PR: pending (`m3/newnuc-orchestration`)
- Plan: [`docs/plans/008-newnuc-orchestration-port.md`](plans/008-newnuc-orchestration-port.md). Wires the PR-F2 dispatcher into `_mam_amicphys_1subarea_clear`. **Completes M3.6 PR-F (newnuc).** Only PR-G (coag) remains in M3.6.
- **Port** (`mam4_jax/processes/amicphys.py`, ~80 LOC of new code — dispatcher does the heavy lifting):
  - `_mam_newnuc_1subarea(qgas_cur, qgas_avg, qnum_cur, qaer_cur, qwtr_cur, temp, pmid, deltat, zmid, pblh, relhum)` → `(qgas_cur, qnum_cur, qaer_cur)`.
  - Pulls `qh2so4_avg` from `qgas_avg[h2so4]` (Fortran default `newnuc_h2so4_conc_optaa == 2`).
  - Sets up size-bin bounds for Aitken mode, clamps `relhum` to `[0.01, 0.99]`, calls the PR-F2 dispatcher.
  - Applies particle-size constraints (`dndt_ait < 100` filter, `mass1p` clamps against `mass1p_aitlo`/`mass1p_aithi`).
  - Adds new-particle mass to `qaer[so4, Aitken]`, new-particle number to `qnum[Aitken]`, subtracts from `qgas[h2so4]`.
- **Wiring changes**:
  - `_mam_gasaerexch_1subarea` return signature extended from `(qgas, qaer)` to `(qgas, qaer, qgas_avg)` — newnuc consumes the time-averaged H₂SO₄ vmr that gasaerexch's analytical solver computes internally as `tmp_q4`.
  - State dict contract gained `zmid` (midpoint altitude, m), `pblh` (PBL height, m), `relhum` (0–1). Box-model defaults: `3000`, `1100`, `0.9` (from `driver.F90:577-579` + `RH_CLEA` namelist).
- **MAM4-MOM-specific simplifications**: no NH₃ branches (`qnh3_cur=0`, `qnh4a_del=0`, `tmp_frso4=1`); optaa=1 H₂SO₄ averaging skipped; diagnostic-output blocks omitted. `h2so4_uptkrate` for the KK2002 correction hardcoded to `1e-3` (the box-model fixture's `zmid > pblh` keeps PBL nuc off → KK2002 enters only multiplicatively, validated to match Fortran at machine ε).
- **Validation infrastructure**:
  - New `--mode instrumented-gasaerexch-and-newnuc-only` in `scripts/capture_reference.py`: namelist `mdo_gasaerexch=1, mdo_newnuc=1, others=0` plus `skip_pcarbon_aging.patch`. Output → `tests/reference/per_process_gasaerexch_and_newnuc/`.
  - Why gasaerexch must also be on: newnuc needs `qgas_avg[h2so4]` from gasaerexch. With gasaerexch off, `qgas_avg=0` → newnuc early-returns at the qh2so4-cutoff guard → no validation surface.
- **Tests** (`tests/test_amicphys.py`):
  - New `test_orchestration_gasaerexch_and_newnuc_matches_fortran`. **Max rel-err 3.9e-16** (machine ε) on `q` / `qqcw` across 60 timesteps × 35 tracers. Size fields use 1e-3 tolerance (Fortran's `update_aerosol_props` mid-step re-uptake, same caveat as PR-D/E).
  - Existing 4 tests (`all_off_passthrough`, `rename_only`, `gasaerexch_matches`, `returns_all_state_keys`) updated to include the new `zmid` / `pblh` / `relhum` state keys; all still pass.
- **Plot** `docs/figures/newnuc_orchestration_residuals.png`:
  - Top: H₂SO₄ gas + Aitken-mode number + Aitken-mode so4 mass over 60 steps, JAX (dashed) over Fortran (solid). H₂SO₄ grows from ~1e-13 to ~3e-13 (gas chem production), Aitken number/mass nearly flat on the log scale (newnuc contributions small relative to existing inventory).
  - Bottom: per-(timestep, tracer) rel-err sits at machine ε for all 3 tracers across 60 steps.
- Full suite: **54/54 green** (53 + 1 new).

## 2026-05-21 — Milestone 3.6 (PR-F2) — Newnuc dispatcher (`mer07_veh02_nuc_mosaic_1box`)

- PR: pending (`m3/mer07-veh02-dispatcher`)
- Plan: [`docs/plans/007-mer07-veh02-dispatcher-port.md`](plans/007-mer07-veh02-dispatcher-port.md). Wraps PR-F1's leaf parameterizations with unit conversion, Kerminen-Kulmala 2002 size correction, grown-particle composition logic, and final `qh2so4_del / qso4a_del / qnuma_del` accounting.
- **Port** (`mam4_jax/newnuc.py`, ~150 LOC):
  - `mer07_veh02_nuc_mosaic_1box(dtnuc, temp, rh, press, zm, pblh, qh2so4_cur, qh2so4_avg, h2so4_uptkrate, dplom_sect, dphim_sect, newnuc_method_flagaa=11)` → 8-tuple matching Fortran's output order.
  - MAM4-MOM-specific simplifications (all in scope per plan 007): no ternary (no NH₃), `nsize=1` hardcoded (amicphys never passes >1), no NH₃-aware composition (`tmp_n3=1` always).
  - Fortran early-returns (the rate-too-low gate at line 856 and the freduce gate at line 1033) expressed as `jnp.where` masks so the function stays JIT-friendly.
- **Validation infrastructure**:
  - New standalone driver `scripts/reference_drivers/mer07_veh02_driver.F90` sweeping a 5D grid (6 T × 5 RH × 3 zm × 8 qh2so4 × 3 uptkrate = 2160 records) covering all 5 regimes: subcutoff / low-rate / active no-PBL / active PBL / gas-limited.
  - Reuses the existing `expose_internals.patch` overlay (which already exposes `mer07_veh02_nuc_mosaic_1box`).
  - New build flag `--mer07-veh02`; new capture mode `--mode mer07-veh02` → `tests/reference/mer07_veh02/reference.npz`.
  - Extended amicphys init dump to capture `mw_so4a_host` (=115), `mw_nh4a_host` (=115; falls back to so4a_host when no NH4), `dens_so4a_host` (=1770). Hardcoded the pure-`parameter` dispatcher constants (`_ACCOM_COEF_H2SO4=0.65`, `_DENS_{AMMSULF,AMMBISULF,SULFACID}=1770`, etc.) directly in `newnuc.py` since they never vary at runtime.
- **Tests** (`tests/test_newnuc.py`, 1 new test): `test_mer07_veh02_dispatcher_matches_fortran`. **Max rel-err 2.27e-12** on all 4 physics outputs (`qnuma_del`, `qso4a_del`, `qh2so4_del`, `dnclusterdt`) across 2160 records. Integer / zero outputs (`isize_nuc`=1, `qnh3_del`=0, `qnh4a_del`=0, `dens_nh4so4a`=1770) checked bit-exact.
- **Plot** `docs/figures/mer07_veh02_residuals.png`:
  - Top: `dnclusterdt` vs `qh2so4` for three (T, z) slices. Inside the PBL (z=100m, z=800m) Wang 2008 dominates and the rate is nearly constant at ~1e16 #/m³/s regardless of T. Above PBL (z=1500m) only binary nucleation fires, dramatically suppressed at warm T until qh2so4 gets high enough.
  - Bottom: per-record rel-err for all 4 physics outputs at ~1e-15 to 1e-12, ~6 orders below ADR-003.
- Full suite: **53/53 green** (52 + 1 new).

## 2026-05-21 — Milestone 3.6 (PR-F1) — Nucleation leaf parameterizations

- PR: pending (`m3/newnuc-helpers`)
- Plan: [`docs/plans/006-newnuc-helpers-port.md`](plans/006-newnuc-helpers-port.md).
- **Scope split**: original `mam_newnuc_1subarea` (~415 LOC) ballooned to ~1265 once the dependency chain into `modal_aero_newnuc.F90` is included (`mer07_veh02_nuc_mosaic_1box` ~580, `binary_nuc_vehk2002` ~193, `pbl_nuc_wang2008` ~77). Owner-approved 3-PR split: this PR covers only the leaf parameterizations (PR-F1), validated standalone; PR-F2 ports the dispatcher; PR-F3 ports the amicphys orchestration.
- **Ports** in new module `mam4_jax/newnuc.py`:
  - `binary_nuc_vehk2002(temp, rh, so4vol)` — Vehkamäki 2002 polynomial parameterization. Returns `(ratenucl, rateloge, cnum_h2so4, cnum_tot, radius_cluster)`.
  - `pbl_nuc_wang2008(so4vol, flagaa, ...)` — Wang 2008 PBL overlay. `flagaa` is a Python int (static at trace time); the early-return path becomes a `jnp.where` mask.
- **Validation infrastructure**:
  - Extended `scripts/patches/expose_internals.patch` with a second hunk that makes the two leaf functions public from `modal_aero_newnuc` (they're inside the module's `contains` block).
  - New standalone driver `scripts/reference_drivers/newnuc_helpers_driver.F90` sweeping 16 × 10 × 12 = 1920 records across (T, RH, [H₂SO₄]); both PBL flagaa branches captured.
  - Driver writes with `1pe27.16e3` format (wider than makoh/kohler's `es24.16`) to accommodate Vehkamäki's 10-order-of-magnitude dynamic range — `binary ratenucl` can be `~1e-100`, which needs 3 exponent digits + the `e` separator.
  - New build flag `--newnuc-helpers`; new capture mode `--mode newnuc-helpers` → `tests/reference/newnuc_helpers/reference.npz`.
- **Tests** (`tests/test_newnuc.py`, 3 tests): binary, PBL flagaa=11, PBL flagaa=12. **Max rel-err**: `binary rateloge` **6.42e-11** (accumulated polynomial roundoff); `binary radius` **1.44e-14**; all others ≤ 4.3e-14. All ~6 orders below ADR-003's 1e-6.
- **Plot** `docs/figures/newnuc_helpers_residuals.png` — top: Vehkamäki nucleation rate vs [H₂SO₄] log-log across (T=230, 267, 300 K) slices, JAX/Fortran visually indistinguishable; bottom: per-record |rel-err| for all 7 outputs across 1920 records vs the ADR-003 1e-6 line.
- Full suite: **52/52 green** (49 + 3 new).

## 2026-05-21 — Milestone 3.6 (PR-E) — Soaexch port (single-substep)

- PR: pending (`m3/soaexch`)
- Plan: [`docs/plans/005-soaexch-port.md`](plans/005-soaexch-port.md).
- **Port** `_mam_soaexch_1subarea` in `mam4_jax/processes/amicphys.py` (~200 LOC of JAX) — non-adaptive variant: assumes `dtcur = dtfull` so the Fortran's `do while (tcur < dtfull)` loop exits after one iteration. Empirically validates on the box-model fixture; if a future fixture ever needs adaptive stepping, the validation test will fail loudly and that triggers PR-E2 (adaptive `jax.lax.while_loop`).
- Wired **unconditionally** into `_mam_gasaerexch_1subarea` at the position matching Fortran line 3430 — no `do_soaexch` flag, matches the Fortran API exactly. The H₂SO₄ analytical solver (PR-D) still runs after soaexch on the H₂SO₄ entries it owns; SOA and H₂SO₄ touch disjoint qaer/qgas slots so the order doesn't matter for correctness.
- **New init-dump constants** (extending `scripts/patches/amicphys_init_dump.patch`): `npoa`, `nsoa`, `iaer_pom`, `iaer_soa`, `npca`, `nufi`, `mode_aging_optaa(ntot_amode)`, `lptr2_soa_a_amode(ntot_amode, nsoa)`. The dump patch also extends `modal_aero_amicphys_init`'s `use modal_aero_data, only:` list with `lptr2_soa_a_amode` (it wasn't in scope before). Added to `data.py` as `AMICPHYS_{NPOA,NSOA,IAER_POM,IAER_SOA,NPCA,NUFI}`, `MODE_AGING_OPTAA`, `LPTR2_SOA_A_AMODE_PRESENT` (boolean form — Fortran only uses the `> 0` check). Parity test in `tests/test_scaffolding.py`.
- **Validation surface restructured:**
  - **DELETE**: `tests/reference/per_process_gasaerexch_only/` (PR-D fixture with soaexch skipped — no longer useful since JAX now runs soaexch).
  - **NEW**: `tests/reference/per_process_gasaerexch/` from `--mode instrumented-gasaerexch-with-soaexch-only` (`mdo_gasaerexch=1, others=0`, **without** `gasaerexch_skip_soaexch.patch`, **with** `skip_pcarbon_aging.patch`).
  - **DROP**: `test_orchestration_gasaerexch_only_matches_fortran` (PR-D's test).
  - **NEW**: `test_orchestration_gasaerexch_matches_fortran` validates JAX `amicphys(mdo_gasaerexch=1, others=0)` against the new fixture. **Max rel-err 4.77e-15** (machine ε) across the 4 SOA tracers (`q[9]=SOA gas`, `q[12]=accum SOA mass`, `q[19]=aitken SOA mass`, `q[28]=coarse SOA mass`).
- **Build script change**: `scripts/build_reference.sh` gains a separate `--skip-pcarbon-aging` flag. Previously `--skip-soaexch` bundled both skips; now they're independent. `--skip-soaexch` still implies `--skip-pcarbon-aging` for back-compat with the PR-D-era fixture-regen workflow.
- **Forward-looking** (no code change in this PR): added **Milestone 7 — Diffrax migration (proposed)** to `docs/PLANS.md`. Captures the future direction to replace the handwritten solvers (PR-D H₂SO₄ analytical, this PR's soaexch step-1/step-2, eventual coag) with [`diffrax`](https://github.com/patrick-kidger/diffrax)-based solvers. Sequenced after M3.6 done so we have a stable bit-comparable baseline first.
- Plot: `docs/figures/soaexch_residuals.png` — top panel: SOA gas drops one order of magnitude over 60 steps as it condenses onto aerosols; accum and aitken pick up the mass. Bottom panel: per-(timestep, SOA-tracer) rel-err vs. ADR-003 — sits at machine ε.
- Full suite: **49/49 green**.

## 2026-05-20 — Milestone 3.6 (PR-D) — Gasaerexch port (H₂SO₄ solver, no SOA)

- PR: pending (`m3/gasaerexch-no-soa`)
- Plan: [`docs/plans/004-gasaerexch-no-soa-port.md`](plans/004-gasaerexch-no-soa-port.md).
- **Leaf helpers** ported in `mam4_jax/processes/amicphys.py`:
  - `_mean_molecular_speed(T, MW)` → `sqrt(8 R T / (π MW))`.
  - `_gas_diffusivity(T, p_atm, MW, vm)` → Fuller-Schettler-Giddings.
  - `_gas_aer_uptkrates_1box1gas(...)` → two-point Gauss-Hermite quadrature on the Fuchs-Sutugin uptake kernel. ~150 LOC.
- **Gasaerexch body** (~150 LOC) — analytical solver path only. SOA exchange and the RK4 branch are out of scope (PR-E for SOA; RK4 unused in box-model build).
- **New constants** in `mam4_jax/data.py` (captured by extending the amicphys init dump): `VMDRY`, `MW_GAS`, `VOL_MOLAR_GAS`, `ACCOM_COEF_GAS`. Plus `ADV_MASS` + `MWDRY` + `MMR_TO_VMR` / `VMR_TO_MMR` (driver-side mmr↔vmr factors). The two conversion factors are stored *independently* (not as `1/MMR_TO_VMR`) so JAX's round-trip ULP drift matches Fortran's separately-rounded `mwdry/adv_mass` and `adv_mass/mwdry`.
- **Fortran-side overlays** for a 1:1 validation surface (all under `scripts/patches/`):
  - `gasaerexch_skip_soaexch.patch` — replaces the `mam_soaexch_1subarea` call (line 3430) with a no-op so the SOA gas tracer doesn't diverge.
  - `skip_pcarbon_aging.patch` — removes the `mam_pcarbon_aging_1subarea` call inside `mam_amicphys_1subarea_clear` (line 2555). Pcarbon aging transfers so4 mass from pcarbon to accum; without it, JAX matches at 1e-6 on every modified tracer.
  - `amicphys_after_writeback.patch` — adds a new dump tag `amicphys_after_writeback` after the driver's vmr→mmr writeback at `driver.F90:1325`. The existing `amicphys_after` dump records `q` *before* the writeback, so it equals `amicphys_before.q` for any sub-process operating in vmr space — previous orchestration tests (PR-A all-off, PR-C rename-only) inadvertently passed on this trivial identity.
- **New capture mode** `instrumented-gasaerexch-only` (`mdo_gasaerexch=1, others=0` + SOA/pcarbon-aging overlays) → `tests/reference/per_process_gasaerexch_only/`.
- **Validation** (`tests/test_amicphys.py`): new `test_orchestration_gasaerexch_only_matches_fortran`. Max rel-err **7.78e-16** (machine ε) on the 5 gasaerexch-modified tracers (`q[6]=H₂SO₄`, `q[7]=SO₂`, `q[10]=accum.so4`, `q[18]=aitken.so4`, `q[25]=coarse.so4`) across 60 timesteps. The size fields (`dgncur_a`, `dgncur_awet`, `qaerwat`, `wetdens`) use 1e-3 tolerance because Fortran's `update_aerosol_props` re-runs wateruptake inside the cond sub-stepping loop — Phase A doesn't implement that re-uptake.
- Plot: `docs/figures/gasaerexch_residuals.png` — top panel: H₂SO₄ gas growth + so4 mass per active mode; bottom panel: per-(timestep, tracer) rel-err vs. ADR-003 1e-6 tolerance and float64 ε. All modified tracers sit at machine ε.
- **Scope correction worth pinning**: original `PLANS.md` listed `mam_gasaerexch_1subarea` at ~305 LOC but didn't account for `mam_soaexch_1subarea` (~330 LOC) called from inside it. Owner-approved split (2026-05-20): now 5 sub-PRs in M3.6 (foundation + gasaerexch + soaexch + newnuc + coag) instead of 4.
- Full suite: **49/49 green**.

## 2026-05-20 — Milestone 3.6 (PR-C) — Foundation + wire rename into orchestration

- PR: pending (`m3/amicphys-foundation`)
- Plan: [`docs/plans/003-foundation-rename-wiring.md`](plans/003-foundation-rename-wiring.md). Owner-approved scope correction (2026-05-20): the original M3 plan's "4 remaining sub-PRs" became 5, because reading `mam_gasaerexch_1subarea`'s source revealed it depends on `mam_soaexch_1subarea` (~330 LOC) and `gas_aer_uptkrates_1box1gas` (~148 LOC) — too large for one PR.
- **Capture infrastructure:**
  - New `scripts/patches/amicphys_init_dump.patch` injects a one-shot text dump near the end of `modal_aero_amicphys_init`. Writes the amicphys-private mapping/conversion tables (`lmap_{gas,num,numcw,aer,aercw}`, `fcvt_{gas,aer,num,wtr}`, plus `mwdry` and `adv_mass(1:gas_pcnst)` so consumers can reconstruct the driver-side mmr↔vmr factor `mwdry/adv_mass`). Has to live inside the module because these tables are module-private.
  - `scripts/capture_reference.py::_read_amicphys_init` parses the new text file and merges its keys into `tests/reference/indices/reference.npz`. Also writes `pcnst_lmap_*` variants (loffset-adjusted, 0-based, -1 sentinel).
  - New `--mode instrumented-rename-only` (namelist with `mdo_gasaerexch=mdo_newnuc=mdo_coag=0, mdo_rename=1`) → `tests/reference/per_process_rename_only/`.
- **JAX foundation** (`mam4_jax/processes/amicphys.py`):
  - `_unpack_state_to_amicphys_view(state)` and `_repack_amicphys_view_to_state(state, ...)` perform a two-stage conversion: driver-side mmr→vmr via `MWDRY/ADV_MASS` per pcnst constituent, then vmr→amicphys-local via `FCVT_*` per amicphys species.
  - `_mam_amicphys_1subarea_clear` now actually calls `_mam_rename_1subarea` when `mdo_rename=1`. Short-circuits the unpack/repack when all four `mdo_*=0` so the all-off passthrough stays bit-exact (round-tripping `qaerwat * FCVT_WTR / FCVT_WTR` would lose 1 ULP otherwise).
  - PR-B's `_mam_rename_1subarea` refactored to be batch-friendly (`qaer_cur[:, mfrm] → qaer_cur[..., mfrm]`, `jnp.sum(...) → axis=-1`) so the orchestration can call it on `(nstep, ncol, pver, naer, nmode)`-shaped arrays without manual iteration. Mathematically identical.
- **JAX data layer** (`mam4_jax/data.py`): new hard-coded constants `AMICPHYS_NGAS/NAER/MAX_*`, `LMAP_{GAS,NUM,NUMCW,AER,AERCW}` (0-based, pcnst-absolute, -1 sentinel for absent species), `FCVT_{GAS,AER,NUM,WTR}`, `FAC_M2V_AER`, `MWDRY`, `ADV_MASS`, `MMR_TO_VMR`. Parity test in `tests/test_scaffolding.py` against `indices/reference.npz`. Cross-check: `LMAP_NUM == NUMPTR_AMODE` (amicphys's internal table independently encodes the same physical mapping as `modal_aero_data`'s).
- **Validation** (`tests/test_amicphys.py`):
  - New `test_orchestration_rename_only_matches_fortran`: JAX `amicphys(state, mdo_rename=1, others=0)` matches the new single-toggle reference at machine epsilon across 60 steps and all 6 aerosol-state arrays.
  - Replaced PR-A's `test_amicphys_all_on_with_stubs_is_passthrough` (no longer accurate post-wiring) with `test_orchestration_with_stubs_matches_rename_only_fortran`. Acts as the new tripwire: with `mdo_*=1` but gasaerexch/newnuc/coag still stubs, only rename can fire — so the orchestration matches the rename-only Fortran. Will start failing once PR-D wires gasaerexch.
  - `test_amicphys_all_off_is_passthrough` and `test_amicphys_returns_all_state_keys` unchanged.
- **Empirical finding** from the new rename-only capture: with gasaerexch off, `qaer_delsub_grow4rnam=0` at the rename call site, and Aitken's `dgn_t_old` stays at the initial `dgnum_aer ≈ 2.6e-8 m` (well below `dp_belowcut ≈ 8e-8 m`). The Fortran rename's optaa=40 guard at line 4141 trips and rename is a no-op every step. So the orchestration test exercises the full unpack/repack pipeline against bit-exact Fortran. The PR-B local-view rename test continues to validate the physics when called with non-zero growth deltas (from the full-physics fixture).
- Full suite: **49/49 green** (was 47 + 2 new orchestration tests).

## 2026-05-20 — Milestone 3.6 (PR-B) — Rename port (`mam_rename_1subarea`)

- PR: pending (`m3/rename-port`)
- Second of five amicphys PRs. Replaces the no-op `_mam_rename_1subarea` stub in `mam4_jax/processes/amicphys.py` with the full port of the Aitken→accum mode-transfer (Fortran lines 3923–4246, ~323 LOC). Plan: [`docs/plans/002-rename-port.md`](plans/002-rename-port.md).
- **Capture infrastructure** (subtasks 1-2):
  - New `scripts/patches/rename_hook.patch` adds two new dump sites inside `mam_amicphys_1subarea_clear` around the rename call at `modal_aero_amicphys.F90:2467`.
  - `mam4_dump_state.F90` gained `dump_rename_snapshot` with the amicphys-local schema (`mtoo_renamexf`, `qnum_cur`, `qaer_cur`, `qaer_delsub_grow4rnam`, `qwtr_cur`, `fac_m2v_aer`).
  - `scripts/build_reference.sh` now compiles `mam4_dump_state.o` into OBJ4 (was OBJ9) so OBJ5's `modal_aero_amicphys.o` can `use` the module.
  - `scripts/capture_reference.py --mode instrumented` now also emits `tests/reference/per_process/rename_{before,after}.npz` (60 records, ~46 KB each). Schema in `tests/reference/SCHEMA.md`.
- **JAX port** (subtask 3, `mam4_jax/processes/amicphys.py`):
  - `_mam_rename_1subarea(qnum_cur, qaer_cur, qaer_delsub_grow4rnam, qwtr_cur, fac_m2v_aer)` — matches Fortran's local-view signature, not the state-dict shape. Cloud-borne path omitted (`iscldy_subarea=False` always at `cldn=0`); pair loop collapsed to the only active Aitken→accum pair; `rename_method_optaa=40` hardcoded.
  - The Fortran's `cycle`-based guard logic is expressed as boolean masks AND'd into a final `do_transfer` decision (JAX needs a single straight-line trace). Mathematically equivalent because intermediate quantities are still well-defined when gates trip.
  - **Orchestration shell wiring deferred**: `_mam_amicphys_1subarea_clear` still skips the rename call. Wiring requires the state-dict ↔ amicphys-local-view unpacking that PR-C lands alongside `_mam_gasaerexch_1subarea` (which produces the `qaer_delsub_grow4rnam` delta).
- **Validation** (subtask 4, `tests/test_rename.py`, 2 tests):
  - `test_rename_matches_fortran_full_physics`: per-step diff across 60 captured timesteps. **Max rel-err: qnum 2.5e-9, qaer 7.0e-10** — both ~3 orders of magnitude below ADR-003's 1e-6 tolerance.
  - `test_rename_conserves_number_and_mass`: total number (summed over modes) and per-species mass (summed over modes) invariant under rename. Catches sign errors in the `.at[].add()` plumbing independent of the Fortran reference.
- **Plan-execution finding** (subtask 4 surprise): the original plan's structural assertion "rename is a no-op when `qaer_delsub_grow4rnam = 0`" was based on a misreading of the Fortran's `optaa != 40` guard 2 (line 4109). The default `optaa == 40` branch uses a different guard (line 4141) that can fire even with zero growth-delta — specifically when the Aitken-mode `dgn_t_old` already lies above `dp_belowcut`. This is correct physics, not a bug; documented in the orchestration-shell comment and in the test that replaced the planned assertion.
- **Empirical finding from the 60-step fixture**: rename actually fires on **every single timestep** here, with max Aitken→accum number transfer ~8.6e7 particles/kmol-air. This is the first M3 port whose physics path is non-trivially exercised by the canonical box-model namelist (calcsize's analogous transfer block is a structural no-op on the same fixture).
- Plot: `docs/figures/rename_residuals.png` — top: per-mode `qnum_cur` time series (Aitken decreasing, accum increasing, JAX/Fortran visually indistinguishable); bottom: per-(timestep, mode) rel-err vs. ADR-003 tolerance.
- Full suite: **47/47 green** (was 45).

## 2026-05-19 — Milestone 3.6 (PR-A) — Amicphys orchestration shell

- PR: [#13](https://github.com/reflective-org/MAM4-JAX/pull/13) (merged at [`dff389d`](https://github.com/reflective-org/MAM4-JAX/commit/dff389d)).
- First of five PRs to port `modal_aero_amicphys_intr`. PR-A wires up the orchestration skeleton with all four physics sub-routines as no-op stubs; PR-B–PR-E will replace one stub at a time.
- **Capture infrastructure**: `scripts/capture_reference.py` now supports `--mode instrumented-amicphys-off`, which writes a namelist with `mdo_gasaerexch=mdo_rename=mdo_newnuc=mdo_coag=0` and saves the dump to `tests/reference/per_process_amicphys_off/`. The Fortran `modal_aero_amicphys_intr` is a true bit-exact passthrough under these toggles (every captured array's `after` matches `before` exactly across 60 timesteps).
- **JAX shell** at `mam4_jax/processes/amicphys.py` (replaces M1 NotImplementedError stub):
  - `amicphys(state, params, config, *, mdo_*)` is the ADR-009 entry. Calls into `_mam_amicphys_1gridcell` → `_mam_amicphys_1subarea_clear`.
  - The clear-sky handler invokes four private helpers in the Fortran order (`gasaerexch → rename → newnuc → coag`), each gated by its `mdo_*` toggle.
  - `_mam_gasaerexch_1subarea`, `_mam_rename_1subarea`, `_mam_newnuc_1subarea`, `_mam_coag_1subarea` are no-op stubs returning the input state unchanged. PR-B–E will replace them.
  - Cloudy path (`_mam_amicphys_1subarea_cloudy`) is **not implemented** — unreachable from the box-model driver (`cldn=0`). Documented in the module docstring.
- **Validation** (`tests/test_amicphys.py`, 3 tests):
  - `test_amicphys_all_off_is_passthrough`: with explicit `mdo_*=0`, JAX output bit-exact matches the Fortran `amicphys_off` reference for all six aerosol-state arrays.
  - `test_amicphys_all_on_with_stubs_is_passthrough`: tripwire — confirms PR-A stubs are no-ops; will start failing as PR-B+ fill in physics.
  - `test_amicphys_returns_all_state_keys`: checks that meteorology / deltat pass through.
- `tests/test_scaffolding.py`: dropped `amicphys` from `PROCESS_MODULES` (it's a real implementation now); kept `gasaerexch`, `newnuc`, `coag`, `rename` since those standalone process modules are dead code in the box-model build per the M3.6-prep finding.
- Full suite: **45/45 green** (was 43).

## 2026-05-19 — M3.6 prep — Documented that amicphys is self-contained

- PR: [#12](https://github.com/reflective-org/MAM4-JAX/pull/12) (merged at [`2975c3d`](https://github.com/reflective-org/MAM4-JAX/commit/2975c3d)).
- Scope-shifting finding ahead of the amicphys port: the box-model `driver.F90` calls `modal_aero_amicphys_intr` in `e3sm_src_modified/modal_aero_amicphys.F90:310`, and **that module contains its own self-contained copies** of all four sub-processes plus the orchestration (`mam_amicphys_1gridcell`, `mam_amicphys_1subarea_clear`/`_cloudy`, `mam_gasaerexch_1subarea`, `mam_rename_1subarea`, `mam_newnuc_1subarea`, `mam_coag_1subarea`). The standalone files `modal_aero_{rename,gasaerexch,newnuc,coag}.F90` are real implementations but **not reachable** from this driver — `modal_aero_rename_sub` is called solely from `modal_aero_gasaerexch.F90:685`, which itself isn't called by the box model.
- Recorded in three docs:
  - `docs/ARCHITECTURE.md` — new "amicphys is self-contained" section with a complete line-by-line module map.
  - `docs/PLANS.md` — M3 entry restructured into a five-PR amicphys plan (5a orchestration shell + 5b–5e four `mam_*_1subarea` sub-routines), targeting the **internal** Fortran symbols.
  - `docs/DEFERRED.md` — explicit "not planned" entry for the standalone modules with resurface conditions if the active call graph ever changes.
- No code changes; tests stayed 43/43 green. This PROGRESS entry itself was added later in a docs catch-up PR (the original PR #12 only touched ARCHITECTURE/PLANS/DEFERRED).

## 2026-05-19 — Milestone 3.5 (PR-B) — Calcsize Aitken ↔ accumulation transfer

- PR: pending (`m3/calcsize-aitacc-transfer`)
- Completes `modal_aero_calcsize_sub`. Adds the Aitken ↔ accumulation mode-transfer block (Fortran lines 944–1294) to `mam4_jax/processes/calcsize.py`. The function now matches the canonical Fortran box-model call (`do_aitacc_transfer_in=.true.`).
- **Transfer-pair tables** computed at module-import in `mam4_jax/data.py`:
  - `AITKEN_MODE_IDX`, `ACCUM_MODE_IDX` (0-based mode indices).
  - `LSPECFRMA_CSIZXF` / `LSPECTOOA_CSIZXF` (interstitial) and the cloud-borne counterparts — 5 species pairs (1 number + 4 mass: sulfate, s-organic, seasalt, m-organic) matched between Aitken and accum by `lspectype_amode`.
  - `NOXF_ACC2AIT`: mask of accum slots whose species isn't in Aitken (p-organic, black-c, dust).
  - `V2NZZ_AIT_ACC`: geometric-mean v2n threshold (= √(voltonumb_aitken · voltonumb_accum)).
- **New helpers** in `mam4_jax/processes/calcsize.py`:
  - `_xferfrac_pair(num_t, drv_t, v2n_target, v2nzz, direction)`: computes (xferfrac_num, xferfrac_vol, triggered_mask) for one direction (ait→acc or acc→ait), faithfully mirroring the Fortran's full-transfer-vs-fractional and clamp logic.
  - `_apply_aitacc_transfer(...)`: full transfer-block implementation. Vectorized per (col, level); pair-list loop is Python-level (5 iterations).
- **`calcsize` now takes** `do_aitacc_transfer: bool = True` keyword. `False` matches the `per_process_no_aitacc/` reference (PR-A's path); `True` matches the canonical `per_process/` reference (this PR's path).
- **`tests/reference/per_process/` refreshed** from nstep=1 to nstep=60 (matches `per_process_no_aitacc/`). The wateruptake test (uses `[0]` snapshot) still passes unchanged.
- **Validation**:
  - Updated `tests/test_calcsize.py` to call with `do_aitacc_transfer=False` explicitly (matches no-aitacc reference fixture name).
  - New `tests/test_calcsize_transfer.py` (4 tests) validates `do_aitacc_transfer=True` against the full-transfer reference. dgncur_a rel-err 2.12e-16, q rel-err < ADR-003 (with `np.allclose(atol=1e-25, rtol=1e-6)` to absorb a ~1e-26 machine-noise artifact at the exactly-zero m-organic mass index), qqcw bit-exact zero.
  - **Structural test**: `do_aitacc_transfer=True` ≡ `do_aitacc_transfer=False` on the box-model fixture — confirms transfer is a no-op here.
- Full suite: **43/43 green** (was 39).
- **`modal_aero_calcsize_sub` is now fully ported.** The transfer block code is faithful but exercised "in spirit only" by the current test (the transfer never triggers in the canonical reference, see `docs/DEFERRED.md`).

## 2026-05-19 — Milestone 3.5 (PR-A) — Calcsize per-mode adjustment + M2 extension

- PR: pending (`m3/calcsize-per-mode-adjust`)
- Two-PR bottom-up plan for `modal_aero_calcsize_sub`; this PR-A covers the per-mode number-bounds adjustment and the dgncur_a recomputation. PR-B will add the Aitken ↔ accum mode-transfer block.
- **M2 extension** (rule #5 — every change supports its tests):
  - New `scripts/patches/disable_aitacc_transfer.patch` (one-line overlay flipping `do_aitacc_transfer_in=.true.` → `.false.` in driver.F90's calcsize call). Cleanly applies on top of `driver_instrumentation.patch`.
  - `build_reference.sh --no-aitacc-transfer` applies the overlay (requires `--instrumented`).
  - `capture_reference.py --mode instrumented-no-aitacc` writes to `tests/reference/per_process_no_aitacc/` (separate from the default `per_process/` so the two captures coexist). Default nstep=60 because calcsize is essentially trivial at nstep=1.
- **JAX port** in `mam4_jax/processes/calcsize.py` (replaces the M1 stub): vectorized per-mode adjustment with the full 3-step bounds procedure (Fortran lines 812–869) covering all four branches (drv_a/c zero vs positive). Helpers `_gather_per_slot`, `_adjusted_num_*`, `_compute_dgn_v2n`. Skips Aitken-accum transfer (PR-B); equivalent to Fortran `do_aitacc_transfer_in=.false.`.
- New constants in `mam4_jax/data.py`: `DGNUM_AMODE`, `DGNUMLO_AMODE`, `DGNUMHI_AMODE`, derived `ALNSG_AMODE`, `DUMFAC_AMODE`, `VOLTONUMB_AMODE`/`VOLTONUMBLO_AMODE`/`VOLTONUMBHI_AMODE` — all from `rad_constituents.F90:167-170` and `modal_aero_initialize_data.F90:428-435`.
- Validation (`tests/test_calcsize.py`, 4 tests): batched across all 60 timesteps. Max relative error in `dgncur_a` evolution = **2.12e-16** — bit-exact at machine ε across all 240 (60 × 4) data points. Number tracers (which never adjust in the box-model setup) pass through unchanged at machine ε.
- `tests/test_scaffolding.py`: dropped `calcsize` from the `PROCESS_MODULES` stub-raises list.
- Residual figure: `docs/figures/calcsize_residuals.png` (top: dgncur_a evolution per mode JAX vs Fortran; bottom: per-(timestep, mode) rel-err).
- Full suite: **39/39 green** (was 36).
- Documentation: `docs/DEFERRED.md` got a new entry calling out that the bounds-adjust + Aitken-accum-transfer branches are dead in the captured reference; `tests/reference/SCHEMA.md` mirrors the note.

## 2026-05-19 — Milestone 3.4 (PR-C) — Wateruptake driver + completion of M3.4

- PR: pending (`m3/wateruptake-driver`)
- Final piece of the wateruptake bottom-up chain. Replaces the M1 `NotImplementedError` stub at `mam4_jax/processes/wateruptake.py` with the full port of `modal_aero_wateruptake_dr` + `modal_aero_wateruptake_sub` (~250 lines vectorized).
- Added per-species and per-mode property tables to `mam4_jax/data.py`:
  - `SPECDENS_AMODE`, `SPECHYGRO_AMODE` (9 species types, from `rad_constituents.F90:96-103`).
  - `SIGMAG_AMODE`, `RHCRYSTAL_AMODE`, `RHDELIQUES_AMODE` (4 modes).
  - Pre-computed `PER_SLOT_DENSITY` / `PER_SLOT_HYGRO` (4 × 14) lookup tables and a `SLOT_VALID` mask for vectorized per-(mode, slot) gather.
  - `RHOH2O = 1000 kg/m³` added to `mam4_jax/constants.py`.
- `wateruptake(state, params, config)` (ADR-009 signature) takes a state dict with `q`, `dgncur_a`, `t`, `pmid`, `cldn` and returns a new state with `dgncur_awet`, `qaerwat`, `wetdens` updated. Internally: gather per-mode dry mass / volume / hygroscopicity using `INDEX_TABLES`, compute v2ncur_a / naer / dryrad / drymass per mode, compute RH from `qsat_water(t, pmid)` and the clear-sky cloud adjustment, call `modal_aero_kohler` per (column, level, mode), apply the deliquescence/crystallization hysteresis branches.
- Validation (`tests/test_wateruptake.py`, 4 tests): end-to-end against `tests/reference/per_process/wateruptake_{before,after}.npz`. Box-model meteorology (`t=273`, `pmid=1e5`, `cldn=0`) is pinned by the namelist + `driver.F90:591` so the test doesn't need additional instrumentation. Measured relative errors:
  - `dgncur_awet`: max 4.53e-16 (machine ε)
  - `qaerwat`: max 1.86e-7 — *but* at the 10⁻²⁰ absolute scale (primary-carbon mode where rwet ≈ rdry and qaerwat is essentially numerical noise). All other modes match at machine ε.
  - `wetdens`: max 2.07e-16 (machine ε)
- Test cleanup: `wateruptake` removed from the `PROCESS_MODULES` stub-raises tuple in `tests/test_scaffolding.py` — it's a real implementation now.
- Residual figure: `docs/figures/wateruptake_residuals.png` (4-panel: dry vs wet diameters, aerosol water content, wet density, per-(mode, var) rel-err).
- Full suite: **36/36 green** (was 33).

## 2026-05-19 — Milestone 3.4 (PR-B) — Port `modal_aero_kohler`

- PR: pending (`m3/kohler-solver`)
- Second bottom-up step of the wateruptake chain: the Köhler-equilibrium wet-radius solver itself, consuming the `makoh_cubic` / `makoh_quartic` polynomial root finders that landed in PR-A.
- Renamed `scripts/patches/expose_makoh.patch` → `scripts/patches/expose_internals.patch` and extended it to also expose `modal_aero_kohler` (single consolidated patch is cleaner than two competing ones touching the same source region).
- `scripts/reference_drivers/kohler_driver.F90`: sweeps a `(rdry, hygro, s)` grid of 7 × 4 × 6 = 168 points designed to exercise all four branches of the solver — insoluble particle (vol ≤ 1e-12 microns³), small-p approximation, generic quartic, near-saturation interpolation. `build_reference.sh --kohler` and `capture_reference.py --mode kohler` produce `tests/reference/kohler/reference.npz` (~6 KB).
- `mam4_jax/kohler.py`: added `modal_aero_kohler(rdry_in, hygro, s)` plus an internal `_pick_smallest_valid_real_root` helper. Vectorised over the batch axis; both polynomial families are solved unconditionally then masked to the appropriate branch via `jnp.where`. Skips the `verify_wateruptake` bisection branch (macro is off in the reference build).
- Constants embedded as literals (Fortran lines 533-539): `mw=18`, `surften=76`, `ugascon=8.3e7`, `tair=273`, `rhow=1` — these are the in-routine values the Fortran uses (the physically-derived alternatives are commented out at lines 525-531).
- Validation (`tests/test_kohler.py`, 4 tests): max relative error against Fortran is **9.77e-14** across all 168 grid points — 8 orders below ADR-003's tolerance. The worst-case is at small rdry near saturation, where root selection is fiddly.
- Residual figure: `docs/figures/kohler_residuals.png` shows Köhler growth-factor curves per hygroscopicity panel (JAX dashed over Fortran solid) plus a per-point rel-err panel.
- Full suite: **33/33 green** (was 29).

## 2026-05-19 — Milestone 3.4 (PR-A) — Port `makoh_cubic` and `makoh_quartic`

- PR: pending (`m3/makoh-polynomial-solvers`)
- First bottom-up step of the wateruptake port chain: the two analytical polynomial root finders that the Köhler solver consumes.
- `scripts/patches/expose_makoh.patch`: small overlay that adds `public :: makoh_cubic, makoh_quartic` to `modal_aero_wateruptake.F90` (the routines are otherwise private). Applied by `build_reference.sh --makoh` onto the transient build copy; vendored tree pristine.
- `scripts/reference_drivers/makoh_driver.F90`: standalone harness that feeds the makoh routines six representative cubic and six representative quartic test cases (well-conditioned plus the "insoluble particle" edge), writes complex roots to text. `scripts/capture_reference.py --mode makoh` parses to `tests/reference/makoh/reference.npz` (~2 KB).
- `mam4_jax/kohler.py` (new module): `makoh_cubic(p0, p1, p2)` and `makoh_quartic(p0, p1, p2, p3)` returning `complex128` roots. Line-by-line port of `modal_aero_wateruptake.F90:684-793`. NaN propagation faithfully matches Fortran (no `safe_cy` guards) so the algorithm's degenerate cases produce the same NaN they do in the reference. Naming rationale: this module will grow with the kohler solver in PR-B; the process-level entry point (the M1 stub at `mam4_jax/processes/wateruptake.py`) gets filled in by PR-C and will call into this module.
- Documented Fortran quirk preserved: `makoh_cubic` accepts `p2` but ignores it (Cardano's method on the depressed cubic). The JAX port exposes `p2` for signature parity with `del p2` and a docstring note.
- Validation (`tests/test_makoh.py`, 4 tests): max relative error **1.49e-14 (cubic)** and **3.47e-15 (quartic)** across all 6 + 6 test cases. Both ~8 orders below ADR-003's 1e-6 tolerance.
- Residual figure: `docs/figures/makoh_residuals.png` (4 panels — absolute and relative error per case for each root branch of cubic + quartic).
- Full suite: **29/29 green** (was 25).

## 2026-05-19 — Milestone 3.3 — Populate `IndexTables` from instrumented Fortran capture

- PR: pending (`m3/populate-index-tables`)
- Extended `scripts/patches/mam4_dump_state.F90` with a `dump_indices()` subroutine that writes `modal_aero_data`'s integer index tables (`numptr_amode`, `numptrcw_amode`, `lspectype_amode`, `lmassptr_amode`, `lmassptrcw_amode`, `nspec_amode`, `modename_amode`, `specname_amode`) to `mam4_indices.txt` once at init, right before `cambox_do_run`'s `main_time_loop`. The unified-diff patch (`driver_instrumentation.patch`) gains the corresponding `call dump_indices()` line via the existing `_generate_driver_patch.py` regenerator.
- `scripts/capture_reference.py --mode instrumented` now also parses `mam4_indices.txt` and writes `tests/reference/indices/reference.npz` (~4 KB, 11 arrays + 3 scalar dims, all 0-based with `-1` sentinels for unused slots).
- `mam4_jax/data.py`: replaced sentinel-filled `IndexTables` with hard-coded MAM4-MOM constants (`NUMPTR_AMODE`, `LMASSPTR_AMODE`, `LMASSPTRCW_AMODE`, `LSPECTYPE_AMODE` — all 0-based) and a module-level `INDEX_TABLES` instance. Accessors `get_number`, `get_mass`, and new `get_mass_by_species_name` now return actual `pcnst`-axis slices instead of raising. `make_sentinel_tables()` kept for tests of the sentinel-raise path.
- Reference-axis ordering: Python uses `(mode, slot)`. Fortran is `(slot, mode)` (column-major); the parser swaps. Documented in `tests/reference/SCHEMA.md`.
- Tests: scaffolding suite grew from 12 to 18 (+`test_index_tables_populated`, `test_index_tables_match_npz_reference`, `test_get_number_returns_slice`, `test_get_mass_returns_slice`, `test_get_mass_raises_on_unused_slot`, `test_get_mass_by_species_name`). Full suite: **25/25 green**.
- The `.npz` is committed as provenance; the Python constants are the source of truth. `tests/test_scaffolding.py::test_index_tables_match_npz_reference` fails loudly if they ever drift.

## 2026-05-18 — Milestone 3.2 — Ports: `qsat_water` and `qsat_ice` + physical constants

- PR: pending (`m3/qsat-functions`)
- Added `mam4_jax/constants.py` with the canonical physical constants (BOLTZ, AVOGAD, RGAS, MWDAIR, MWWV, LATICE, LATVAP, derived RDAIR/RH2O/EPSQS, plus `wv_saturation`-name aliases HLATV/HLATF/RGASV/EPSQS). Values transcribed verbatim from `mam4-original-src-code/e3sm_src/shr_const_mod.F90:33-61` so the JAX port uses the same numbers the Fortran sets through `gestbl()`.
- Built a reference driver (`scripts/reference_drivers/qsat_driver.F90`) that calls `gestbl` with box-model constants then sweeps `qsat_water` (Goff–Gratch via inline polysvp formula) and `qsat_ice` (Clausius–Clapeyron with combined latent heat of sublimation) over a 301-T × 5-p grid. New `--qsat` flag in `build_reference.sh`, `--mode qsat` in `capture_reference.py`. Output: `tests/reference/qsat/reference.npz` (~48 KB).
- Ported `qsat_water(T, p)` and `qsat_ice(T, p)` to `mam4_jax/saturation.py`, plus a `qs_from_es(es, p)` helper that captures the shared `qs = epsqs · es / (p − (1 − epsqs) · es)` formula and the Fortran's `qs < 0 → qs = 1` clamp. **Preserved the Fortran inconsistency**: `qsat_ice` uses Clausius–Clapeyron, not `polysvp_ice`. Documented in the saturation module docstring; callers wanting consistency can `qs_from_es(polysvp_ice(T), p)`.
- Validation (`tests/test_qsat.py`): max relative error against Fortran is **9.36e-14 (water)** and **7.81e-15 (ice)**. Both ~8+ orders below ADR-003's 1e-6 tolerance. Test suite total: 19/19 green.
- Residual figure: `docs/figures/qsat_residuals.png` (four panels — qs(T) per pressure level for water + ice, with rel-err vs T below).

## 2026-05-18 — Milestone 3.1 — First port: `polysvp` (saturation vapor pressure)

- PR: pending (`m3/polysvp-port`)
- Built a standalone Fortran reference driver (`scripts/reference_drivers/polysvp_driver.F90`) that calls `wv_saturation::polysvp` over a 170 K – 320 K sweep (1501 points, 0.1 K resolution). Linked against the existing baseline build's object files. `scripts/build_reference.sh --polysvp` produces `run/polysvp_driver.exe`; `scripts/capture_reference.py --mode polysvp` runs it and archives `tests/reference/polysvp/reference.npz` (~36 KB, arrays `T`, `esat_water`, `esat_ice`).
- Ported `polysvp` to `mam4_jax/saturation.py` as `polysvp_water(T)` and `polysvp_ice(T)` (plus a Fortran-parity `polysvp(T, type)` dispatcher). Direct line-by-line port of the Goff–Gratch polynomial — each Python line traces 1:1 to the Fortran source.
- Validation (`tests/test_polysvp.py`): max relative error against the Fortran reference is **4.31e-15 (water)** and **4.14e-15 (ice)** across 1501 points — eleven orders of magnitude below ADR-003's 1e-6 tolerance, essentially bit-equivalent in `float64`.
- Residual figure: `docs/figures/polysvp_residuals.png`, generated by `scripts/plot_polysvp_residuals.py`. Top panel overlays JAX and Fortran on log axes; bottom panel shows rel-err vs T with the 1e-6 tolerance line and the float64 ε floor.

## 2026-05-18 — Milestone 2 — Fortran reference output capture

- PR: pending (`m2/reference-capture`)
- Built the vendored MAM4 Fortran box model end-to-end via `scripts/build_reference.sh` (auto-detects `gfortran` + NetCDF via `nf-config`/`nc-config`; adds `-fallow-invalid-boz` for modern gfortran and two `-L` paths for Homebrew's split NetCDF prefixes). Vendored tree stays pristine; build artifacts live in gitignored `mam4-original-src-code/{build,run}/`.
- Captured the canonical 12-point convergence sweep (`1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800` substeps over 1800 s) into `tests/reference/sweep/*.nc` (12 NetCDF files, ~1.7 MB total). Discovered and worked around the upstream `run_test.csh`'s broken sweep loop and hard-coded outpath by reimplementing the sweep in `scripts/capture_reference.py`.
- Added the patch-overlay instrumentation (ADR-012): `scripts/patches/mam4_dump_state.F90` is a small Fortran helper module that writes binary state snapshots; `scripts/patches/driver_instrumentation.patch` inserts six `call dump_snapshot(...)` hooks around `calcsize`, `wateruptake`, and `amicphys` inside `cambox_do_run`. The build script applies both onto a transient copy of `driver.F90` and overrides `OBJ9` so the helper compiles before `driver.o`.
- `scripts/capture_reference.py --mode instrumented` rebuilds with the overlay, runs a single configurable-`nstep` integration, parses the six `mam4_dump_*.bin` files, and writes them as `tests/reference/per_process/*.npz` with a documented array contract.
- Authored `docs/REFERENCE_BUILD.md` (prereqs, build flag rationale, what the scripts do, missing-from-upstream `&size_parameters` namelist group, why the upstream `run_test.csh` is replaced) and `tests/reference/SCHEMA.md` (artifact layout for both sweep and per-process outputs, array shapes/dtypes, VMR-conversion caveat for `amicphys`).
- `git diff mam4-original-src-code/` is empty before, during, and after a build — the vendored tree contract from ADR-001 holds.

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
