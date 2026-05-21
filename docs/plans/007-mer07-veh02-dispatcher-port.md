# Plan 007 — M3.6 PR-F2: port the `mer07_veh02_nuc_mosaic_1box` dispatcher

> **Status:** approved 2026-05-21.

---

## Context

PR-F1 (PR #19, merged) ported the two leaf parameterizations (`binary_nuc_vehk2002` + `pbl_nuc_wang2008`) as pure JAX functions. PR-F2 ports the dispatcher (`modal_aero_newnuc.F90:598-1173`, ~580 LOC) that wraps those leaves with unit conversion, the Kerminen-Kulmala 2002 size correction, grown-particle composition logic, and the final `qh2so4_del / qso4a_del / qnuma_del` accounting. After PR-F2 lands, only PR-F3 (amicphys glue) remains.

## Scope

**IN:**
- Unit conversion `so4vol = qh2so4_avg * cair * avogad * 1e-6`.
- Binary path (calls PR-F1's `binary_nuc_vehk2002`).
- PBL path (calls PR-F1's `pbl_nuc_wang2008`) gated by `z <= max(pblh, 100)`.
- `adjust_factor_bin_tern_ratenucl` correction.
- Wet/dry volume ratio.
- Size-bin assignment (collapses to `isize_nuc = 1` for `nsize=1`).
- Grown-particle composition (for no-NH₃, pure sulfacid).
- KK2002 size correction.
- Final deltas via `freduce`.

**OUT** (deliberately not ported):
- Ternary nucleation (no NH₃ in MAM4-MOM → unreachable).
- `nsize > 1` (amicphys always passes 1 — JAX hardcodes scalar `dplom_sect / dphim_sect`).
- Diagnostic output blocks (lines 1074-1170).

## Subtasks

Each ≈ one commit; single PR titled `M3.6 (PR-F2): port mer07_veh02_nuc_mosaic_1box dispatcher`.

1. **Capture additional init constants** via the amicphys init dump:
   `mw_so4a_host`, `mw_nh4a_host`, `dens_so4a_host`. Add to `data.py` with parity test. Transcribe the dispatcher-internal `parameter`s (`_ACCOM_COEF_H2SO4=0.65`, `_DENS_{AMMSULF,AMMBISULF,SULFACID}=1770`, `_MW_{AMMSULF,AMMBISULF,SULFACID}=132/114/96`, `_MW_SO4A=96`, `_MW_NH4A=18`, `_ADJUST_FACTOR_BIN_TERN_RATENUCL=1.0`) directly into `mam4_jax/newnuc.py`.

2. **Standalone Fortran driver** `scripts/reference_drivers/mer07_veh02_driver.F90` — sweeps a 5D (T, RH, zm, qh2so4, h2so4_uptkrate) grid covering five regimes: subcutoff / low-rate / active no-PBL / active PBL / gas-limited. 2160 records total.

3. **Build flag + capture mode**: `--mer07-veh02` build flag, `--mode mer07-veh02` capture mode. Uses the existing `expose_internals.patch` overlay (already makes `mer07_veh02_nuc_mosaic_1box` public).

4. **JAX port** `mer07_veh02_nuc_mosaic_1box` in `mam4_jax/newnuc.py`. ~150 LOC after the simplifications (no ternary, no `nsize > 1`, no NH₃-aware composition). Early-return paths expressed as `jnp.where` masks.

5. **Test** `tests/test_newnuc.py` extended with `test_mer07_veh02_dispatcher_matches_fortran`. Rel-err < 1e-6 on `qnuma_del / qso4a_del / qh2so4_del / dnclusterdt`; exact-equal on the integer/zero outputs (`isize_nuc`, `qnh3_del`, `qnh4a_del`, `dens_nh4so4a`).

6. **Residual plot** → `docs/figures/mer07_veh02_residuals.png`.

7. **Docs** (rule #5): PROGRESS, PLANS (mark 5f.PR-F2 done), SCHEMA, REFERENCE_BUILD, FEATURES.

## Verification

- `python -m pytest -q` → 53/53 green (52 + 1 new).
- `python scripts/capture_reference.py --mode mer07-veh02` regenerates the fixture.
- `python scripts/plot_mer07_veh02_residuals.py` renders the figure (worst rel-err < 1e-6).

## Out of scope

- Amicphys orchestration (`mam_newnuc_1subarea`) — PR-F3.
- Ternary nucleation (no NH₃).
- `mam_pcarbon_aging_1subarea` — still M3.6-out-of-scope as flagged in `docs/UPSTREAM_FORTRAN_BUGS.md` 🔵.
