# Plan 026 — Primary-carbon aging (`mam_pcarbon_aging_1subarea`) + float32 coag mass transfer

**Status:** in progress (2026-08-25). PR open at [#75](https://github.com/reflective-org/MAM4-JAX/pull/75).
**Branch:** `feat/pcarbon-aging` → `main`.
**Contributor:** @duncanwp (motivation: jax-gcm BC lifetime ~21 d vs observed 5–8 d — [climate-analytics-lab/jax-gcm#721](https://github.com/climate-analytics-lab/jax-gcm/issues/721)).

> **Numbering note.** `feat/cam-driver` carries `docs/plans/025-cam-driver.md`; this plan takes 026 so the two branches don't collide at merge. 019 and 020 remain unused.

---

## 1. Scope

Closes the last amicphys sub-process left out of scope in `FEATURES.md`, plus the float32 defect that makes it wrong when a host runs in f32:

1. **Port `mam_pcarbon_aging_1subarea`** (`e3sm_src_modified/modal_aero_amicphys.F90:5111-5285`, called at `:2555`) — the sulfate-monolayer coating criterion that converts aged primary-carbon particles to the accumulation mode.
2. **Fix float32 intermodal coagulation mass transfer being identically zero** — `getcoags`' third-moment harmonic mean formed `nc3*fm3 ≈ 1e-39`, below float32 min-normal, so `betaij3` flushed to 0 for every pair. Same defect class as plan 023 / PR [#60](https://github.com/reflective-org/MAM4-JAX/pull/60), one level deeper.

Aging is what rescues the so4/soa the core condenses onto pcarbon each step: the pcarbon `LMAP_AER` row has no pcnst slot for those species, so without aging they are silently dropped at the repack — a standing sulfur/SOA sink. The two defects close together.

---

## 2. What's in scope vs. deferred

| Item | This PR | Deferred |
| --- | --- | --- |
| `_mam_pcarbon_aging_1subarea` (criterion + transfer) | ✅ | — |
| `mdo_pcarbonaging` toggle through `amicphys` / `run_step` / `run_timesteps` | ✅ | — |
| `configure_pcarbon_aging(n_so4_monolayers=…)` + per-call override | ✅ | — |
| Reciprocal-form harmonic mean, all 7 `getcoags` sites | ✅ | — |
| `qs21` prefactor cancellation rewrite (see ADR-019) | ✅ | — |
| `1-exp(-x)` → `-expm1(-x)` in coag mass transfer + number loss | ✅ | — |
| `qaer_del_cond` / `qaer_del_coag` budget attribution (F90:5169-5210) | — | ✅ `DEFERRED.md` |
| Marine-organic age pairs (`MODAL_AERO_9MODE`, F90:5931+) | — | ✅ out of build scope |
| Trajectory-level f32 acceptance bar | — | ✅ still open from ADR-018 §5 |

This PR **resolves** plan 023 §2's deferred row *"Audit other coag coefficients (qs11/qs22/qs12/qs21) for f32 magnitude bounds."* That row's claim — "only qv12 is f32-broken" — was wrong: all four second-moment coefficients also flushed to zero, and so did `betaij3` itself, which plan 023 believed it had fixed. The correction is appended as **plan 023 §8**; the measurement is in §5 below.

---

## 3. Physics

A pcarbon particle counts as aged once its hygroscopic shell — sulfate condensed from H₂SO₄ plus SOA, counted as so4-*equivalent* volume via `fac_eqvso4hyg_aer` — covers the mode's surface with `n_so4_monolayers` monolayers of so4:

```
xferfrac = min( vol_shell · dgn · fac_volsfc / (6 · n_mono · 4.76e-10 m · vol_core), 1 − 10ε )
```

where `6/(dgn·fac_volsfc)` is the log-normal mode's surface-area-to-volume ratio (F90:5219-5239). That fraction of the mode's **mapped** species (pom, bc, mom — `lmap_aer > 0`) and of the mode number moves to accum; the **unmapped** shell species (soa, so4, and any coagulated ncl/dst) move **entirely**, since accum has pcnst slots for them and pcarbon does not.

Build-specific simplifications, all verified against the vendored source:

- Shell is so4 + soa only. `iaer_nh4`/`iaer_no3`/`iaer_cl` are `-999888777` in this build (F90:5618-5642), and `aging_include_seasalt = .false.` (F90:200) excludes ncl from the coating (it still *moves* with the transfer).
- Single age pair (pcarbon → accum); the marine-organic pairs exist only in the 9-mode build.
- `deltat` is unused in the Fortran body — the transfer is per-call instantaneous.

---

## 4. Threshold default: 3.0, not 8.0

The amicphys path reads `n_so4_monolayers_pcage` at init through `phys_control`:

```
box_model_utils/phys_control.F90:26          n_so4_monolayers_pcage = 3.0_r8
modal_aero_initialize_data.F90:417,585   →   modal_aero_amicphys_init
```

The frequently-quoted `8.0` (`e3sm_src/modal_aero_gasaerexch.F90:37`) is a `parameter` belonging to the **legacy** `modal_aero_coag` aging path (`modal_aero_coag.F90:87` USEs it); amicphys never references it. The package default is therefore **3.0** — defaults reproduce the reference, deviations are opt-in — locked by `test_default_threshold_is_the_amicphys_reference_value`.

Measured sensitivity of the 60-step trajectory vs the canonical `per_process/` bundle:

| `n_so4_monolayers` | max aerosol rel-err | max gas rel-err |
| --- | --- | --- |
| 1.0 (ECHAM-HAM `m7_coat`) | 204 % | 10.5 % |
| **3.0 (reference, default)** | **2.07 %** | **5.30 %** |
| 8.0 (legacy coag path) | 434 % | 5.76 % |
| aging off | 1411 % | 32.6 % |

---

## 5. Float32 coagulation

`getcoags` formed every harmonic mean as `a*b/(a+b)`. In SI metres the operands are small enough that the *product* underflows float32 min-normal (1.18e-38) while the operands themselves are in normal range. Measured at box-typical diameters before this PR, four of the eight `getcoags_wrapper_f` outputs were identically zero in f32: `betaij3`, `betaij2i`, `betaij2j`, `betaii2`, `betajj2`. Only `betaij3` is consumed by `_mam_coag_1subarea` today, but the wrapper is public.

All seven harmonic means now route through a shared `_harmonic_mean_safe(a, b) = 1/(1/a + 1/b)` with a double-`where` guard so the dead branch stays benign for reverse-mode. Fixing those exposed a second, independent defect in `qs21`'s prefactor — see **ADR-019**.

---

## 6. Validation

`tests/test_pcarbon_aging.py` (9 tests):

- Hand-computed criterion and transfer fractions at machine ε; per-species mass and number conservation; saturation (`1−10ε`) and zero-core edge cases.
- **Sulfur closure through full `amicphys`**: with aging on, gas + aerosol S closes to the netprod source at 1e-9; with aging off the repack leak reproduces at >100× the closure residual.
- **End-to-end parity vs the canonical aging-ON bundle** (`tests/reference/per_process/`, previously unused for exactly this reason): `run_step` and the 60-step trajectory at the ADR-015 coarse-dt bars. Aerosol tracers ≤ 2.1 %, gas slots ≤ 5.3 % (the known diffrax soaexch offset). The two Fortran builds differ by up to 14× on pcarbon tracers, so passing both this and `test_driver.py`'s no-aging companion at 5 % is a real constraint.

`tests/test_coag.py`: `test_wrapper_all_outputs_nonzero_in_float32` locks all 8 wrapper outputs (nonzero, `rtol=1e-3` vs f64). The f64 reference sweep still passes at `rtol=1e-6` — with the `qs21` margin caveat recorded in ADR-019.

---

## 7. Open items at review time

- ADR-019's `qs21` decision (keep the more-accurate form and document, vs. scope it to f32).
- `test_coag.py`'s `RTOL = 1e-6` is only 8.4× above `qs21`'s measured error and the margin shrinks with diameter ratio — see ADR-019 §Consequences.
