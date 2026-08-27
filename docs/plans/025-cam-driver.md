# Plan 025 — CAM driver (plan 024 PR G), and what it can legitimately reuse

**Status:** IN PROGRESS. Written 2026-08-21 while the owner was away; every
assumption is listed in §5 for review rather than buried in code.

**Scope, per owner 2026-08-20:** box model only. Coagulation, condensation,
nucleation. **No deposition, no column transport, no evaporation** — evaporation
is an organics process, not a sulfate one (see `mam-box-fortran`
`docs/design/SO4_SCOPE.md`). SO4-only species set.

---

## 1. The question that had to be answered first

`coupling/amicphys.py` already contains standalone helpers for all four
processes — `_mam_gasaerexch_1subarea`, `_mam_rename_1subarea`,
`_mam_newnuc_1subarea`, `_mam_coag_1subarea`. The tempting move is to wire them
into CAM's sequence and call it a CAM driver.

**That would be CAM's *sequencing* with E3SM's *kernels*.** Whether it is
legitimate has to be settled per process, not assumed. Using the per-process
discrepancy reports in `mam-box-fortran/docs/reference/discrepancies/`, plus one
new numerical check:

| Process | Reusable for CAM? | Basis |
| --- | --- | --- |
| **Uptake rates** (`gas_aer_uptkrates`) | ❌ **No — corrected in §7** | CAM vs E3SM-legacy is bit-identical (58/58), but the JAX port implements the *amicphys* variant, a third one (Knudsen-dependent β vs CAM's fixed β=2, exact vs truncated constants, ac handling). CAM's must be ported |
| **Condensation, H2SO4, tropospheric** | ✅ **Yes — verified numerically** | See §2 |
| **Condensation, H2SO4, `sulfate_equilib`** | ❌ **No** | CAM-only. No E3SM counterpart exists to have been ported |
| **Coagulation** | ✅ **Yes** | `getcoags` is **byte-for-byte identical** between CAM and E3SM — comments and whitespace included — over 1519 lines |
| **Nucleation** | ✅ **Yes, in a box** | Leaf parameterisations are 0-diff with exact literal matches. The one answer-changing difference (V2) is **cloud handling**, and a box model runs `cld = 0`, where both reduce to the same thing. See assumption A4 |
| **Rename** | ❌ **No** | CAM's default path (**A1**) and E3SM's production path (**C**) are "a real algorithmic difference". The JAX port ports **C**. See §3 |

**So exactly two things need new science work: `sulfate_equilib` condensation,
and rename.** Everything else is already in this repo and correct for CAM.

## 2. Tropospheric H2SO4 condensation is the same maths

CAM (`modal_aero_gasaerexch.F90`):

```fortran
sum_uprt_so4 = Σ_n uptkratebb(n)                        ! :407-408
avg_uprt_so4 = (1 - exp(-dt*sum_uprt_so4))/dt           ! :492
sum_dqdt_so4 = q(l_so4g) * avg_uprt_so4                 ! :503
dqdt_so4(n)  = fgain_so4(n) * sum_dqdt_so4              ! :567-569
               where fgain_so4(n) = uptkratebb(n)/sum_uprt_so4
```

JAX `_linear_uptake_closed_form` solves `dg/dt = -Kg + src`,
`da_i/dt = uptk_i·g` with `K = Σ uptk_i` — the same competing-sink system. With
`src = 0` the two are algebraically identical, since
`uptk_i·g_avg·dt = (uptk_i/K)·g0·(1-e^{-K dt})`.

**Checked numerically rather than left as algebra:** 200 random cases over
`uptk ∈ [1e-6, 1e-2]`, `g0 ∈ [1e-14, 1e-9]`, `dt ∈ [1, 1000]` s — worst relative
difference **1.01e-14**. Identical to round-off.

`src = 0` is the right setting for CAM: CAM does not add gas production inside
gasaerexch. It arrives as `del_h2so4_gasprod`, which the driver hands to
*nucleation*, not to condensation. See assumption A2.

## 3. Rename is the real blocker

From the rename report's legend:

| Tag | Code |
| --- | --- |
| **A1** | `modal_aero_rename_no_acc_crs_sub` — **CAM's default** |
| **A2** | `modal_aero_rename_acc_crs_sub` — CAM, `modal_accum_coarse_exch=.true.` |
| **B** | E3SM **legacy** `modal_aero_rename_sub` (dead code in E3SM) |
| **C** | E3SM **production** `mam_rename_1subarea`, inside amicphys |
| **J** | the JAX port — **ports C** |

Verdict: *"A1 vs B is a non-difference. A1 vs C is a real algorithmic
difference."* A1 and B are line-for-line identical — 27 differing lines out of
243, **zero of which change the arithmetic**.

So the existing JAX rename cannot serve CAM. But because **A1 ≡ B**, porting
CAM's rename is a single well-defined job that also gives E3SM's legacy path for
free.

This matters more than it looks: rename is what produced the ×31 accumulation-
number jump in the Fortran box runs. A driver without it, or with the wrong one,
is not approximately right — it is qualitatively different.

## 4. What this PR does and does not do

**Does:** the CAM sequence — `gasaerexch` (tropospheric H2SO4) → `newnuc` →
`coag` — on grid-cell means, with sub-stepping exposed, reusing only the kernels
established above as legitimate.

**Does not:** rename, or the `sulfate_equilib` branch. Both raise rather than
silently substituting an E3SM equivalent, because a wrong-but-running driver is
worse than one that refuses.

**Sub-stepping is exposed and defaults ON**, against CAM's own behaviour. The
Fortran showed a **2.08× spread** in accumulation sulfate across dt 120 s → 1.875 s
with nucleation active, versus 1.3 % with it off. Faithfully reproducing CAM's
un-substepped splitting means reproducing an O(50 %) error at a 30 s step. See
assumption A6 — this is a deliberate divergence and needs owner sign-off.

## 5. Assumptions — all of them, for review

Numbered so they can be accepted or rejected individually.

| # | Assumption | Why | Risk if wrong |
| --- | --- | --- | --- |
| **A1** | Reusing E3SM-derived kernels for CAM is legitimate **only** where the discrepancy reports say the code is identical, and I have not re-verified those reports beyond the two checks in §2 | The reports carry `file:line` citations and were written from the sources | Moderate. A report being wrong would put wrong physics in a "CAM" driver |
| **A2** | `src = 0` inside CAM's gasaerexch; gas production reaches *nucleation* via `del_h2so4_gasprod`, not condensation | Matches `aero_model.F90`'s call sequence and the Fortran box driver already built | Low — verified in the coupling module |
| **A3** | SO4-only means the SOA differences (`opoa_frac`, volatility) are out of scope, so "the 58-line kernel is all that's shared" is pessimistic for our case | Owner scope 2026-08-04 and 2026-08-20 | Low |
| **A4** | Nucleation's cloud-handling difference vanishes because the box runs `cld = 0` | Both formulations weight by `(1-cld)` or area-weight by `fclea`; at `cld = 0` both give the full clear-sky result | **Should be tested**, not assumed. Listed as work below |
| **A5** | Rename must be CAM's A1, not E3SM's C, and A1 ≡ B makes it one port | Rename report verdict | Low on the verdict; the port itself is unwritten and untested |
| **A6** | Sub-stepping should default **ON**, diverging from CAM | dt-convergence measurement | **This is a science-policy call, not a technical one.** Needs owner sign-off |
| **A7** | `cam_mam4`/`cam_mam5` from PR #73 are the right topologies | Read out of an initialised CAM model; losslessness verified | Low |
| **A8** | Grid-cell means, no sub-areas, is right for CAM | CAM has no sub-area concept — `grep -rI amicphys` over CAM `src/` returns zero hits | Low |

## 6. Progress and remaining work

### Done

1. ✅ **A4 tested, and it turned up a defect instead.** The JAX port implements
   only the clear-sky sub-area, so at `cld = 0` it trivially matches CAM's
   `(1-0) × clear` — A4 holds. But `_mam_amicphys_1gridcell`'s docstring
   *claimed* a non-zero cloud fraction "raises a clear error"; the code declined
   to check and never read `cldn` at all, so a cloudy cell silently got
   clear-sky physics applied to the whole cell. Fixed, and the guard had to move
   **outside** the jit: `driver.run_step` is `@jax.jit`, so a check inside it
   could never fire.

2. ✅ **CAM rename implemented as a selector**, not a second implementation —
   E3SM's code contains CAM's algorithm as its `optaa /= 40` branch, and every
   quantity it needs was already computed. Default stays `"e3sm"`, asserted.

3. ✅ **Reference capture built** (`mam-box-fortran/tools/capture_rename`), five
   growth values. Two mistakes worth recording: rename is **growth-driven**, so
   the first capture with `dqdt = 0` produced an all-zero transfer that looked
   like a valid reference; and zsh's lack of word-splitting silently pinned the
   growth parameter, which made the mass transfer look constant across a sweep.

### Validated, and precisely how far

Compared on **dimensionless** transfer fractions rather than by mapping CAM's
`q`/`dqdt` into the amicphys-local view. That mapping is where a factor-of-1000
error produces a *fake* validation — either failing for the wrong reason, or
passing because two errors cancel. Both inputs (`v2n/voltonumb`, `deldryvol/dryvol`)
and outputs (`xferfrac_num`, `xferfrac_vol`) are unit-free, so no conversion
appears anywhere.

**Worst relative difference across five growth values: 9.6e-10** — the floor the
reference's 9 significant figures can resolve.

Those five points sit in a regime where **both branches agree exactly**, so
they validate the machinery the two algorithms *share* — erfc tail integrals,
`dp_cut`, `factoraa`/`factoryy`, the transfer-fraction clamps. Necessary, not
sufficient.

### ✅ And now validated where they DIVERGE — machine precision

The divergent window had to be **derived**, not searched for:

```
E3SM clamps when   dgn_t_old  >  dp_belowcut  (= 0.99 · dp_cut)
CAM   clamps when  dgn_t_new  >= dp_cut
      with         dgn_t_new  =  dgn_t_old · (1+growth)^(1/3)
```

so divergence needs an **oversized mode with small growth** — `dgn_t_old` just
above `dp_belowcut` while `dgn_t_new` stays below `dp_cut`, which bounds growth
under about 3 %. Confirmed behaviourally: at `dgn_old = 8.12e-8` the branches
diverge at growth 0.005 and 0.01 and then **converge again at 0.03**, exactly
where the derivation says they should.

Result over six points (two diameters × three growth values):

| | vs CAM's Fortran |
| --- | --- |
| **JAX `method="cam"`** | **worst 8.1e-15** — machine precision |
| JAX `method="e3sm"` | off by **68 % to 308 %** |

So the comparison genuinely discriminates: a `method="cam"` that silently fell
through to the E3SM path would fail by orders of magnitude, and there is a test
asserting that.

⚠ Getting there needed a **units fix in the capture tool**. Rename's
`dryvol_t_old` is in m³-AP/kmol-air — `q·(specmw/specdens)`, not raw `q`.
Parameterising by raw volume put the intended `v2n` an order of magnitude below
the `v2nhirlx` floor, so `num_t_oldbnd` clamped every input to the same value:
two visibly different states (v2n 4.090e20 and 9.695e20) produced
**byte-identical output with no error anywhere.**

### ✅ The sulfeq equilibrium cluster — ported and validated at machine precision

The `sulfate_equilib` work splits in two: the EQUILIBRIUM VALUE (computed per
mode inside CAM's water uptake) and its CONSUMPTION (the reversible
condensation branch in gasaerexch). The first half is done:

- `mam4_jax/physics/strat_sulfate.py` ports `calc_h2so4_wtpct` (Tabazadeh
  1997 composition) + `calc_h2so4_equilib_mixrat` (Ayers/Kulmala vapor
  pressure, Giauque enthalpy, dual Kelvin factors) + the CAM `qsat_water`
  they stand on — which is NOT the already-ported E3SM box `qsat_water`:
  CAM returns `qs = 1` whenever `p <= es`, the E3SM box clamps only a
  negative-denominator `qs`, and they disagree on `es ∈ [p, p/(1−ε)]`
  (reachable at the routine's own `t = 450 K` clamp).
- The routines were **private** to `modal_aero_wateruptake`; exposed by a
  visibility-only patch applied to a staged copy by the new
  `mam-box-fortran/tools/capture_sulfeq` (the box model's process masks
  cannot isolate them — they only run under `modal_strat_sulfate`, deep in
  the wateruptake driver).
- The capture grid pins every branch by construction: both T clamps
  (135→140, 460→450 K), all three Tabazadeh activity regimes plus both
  activity clamps (qh2o is BUILT as `activ_target × qs`, so regime coverage
  cannot drift with T/p), and Kelvin-strong→negligible diameters (1e-8 →
  9e-7 m, the MAM5 `coarse_strat` dgnum). 27 qsat + 189 wtpct + 567 full
  cases → `tests/reference/cam_sulfeq/sulfeq.json`.
- **Measured worst relative errors** (`tests/test_strat_sulfate.py`):
  wtpct **1.3e-15**, sulden **7.1e-16**, qh2so4_equilib **3.5e-14** (the
  ~100-magnitude exponent inside `exp` amplifies its last ULP by ~1e-14 —
  gated at 5e-13). Plus Kelvin monotonicity in diameter and reverse-mode
  gradient finiteness across every branch.
- ⚠ **Upstream defect found while porting, preserved faithfully**: the
  first surface-tension interpolation pairs knot `i−1`'s ordinate with knot
  `i`'s abscissa (`surf_tens = sig1 + dsigma_dwt*(wtpct_flat - stwtp(i))`,
  F90:1005), offsetting the whole segment by `−(sig2−sig1)`. Both sibling
  lookups in the same routine are correct. Written up in mam-box-fortran
  `docs/bugs/BUG-cam-wateruptake-surftens-interp.md`; the port keeps
  bit-parity with the bug.

### ✅ The reversible condensation branch (the cluster's consumer)

`h2so4_reversible_uptake` (same module) ports
modal_aero_gasaerexch.F90:523-566: exponential decay of the gas toward the
mode-weighted equilibrium `g_equ = Σ(uptk·sulfeq)/Σuptk`, per-mode
`dqdt = uptk·(g_avg − sulfeq)` with the `a_end ≥ 0` evaporation floor, the
`kxt < 1e-5` first-order branch, the `deltatxx = deltat·(1+1e-15)` nudge,
and the three-state `ido_so4a` mode classification (1 = has slot, 2 = CAM's
slotless pcarbon age-source, 0 = inactive). Validated against a verbatim
NumPy transcription of the Fortran (loops, `1-exp`, cycles) over 300
randomized states spanning both branches, condensation/evaporation, the
floor, and all ido classes — bar 5e-11, sized by the one documented
arithmetic deviation (`-expm1(-kxt)`, repo-standard per plan 026/ADR-019,
worth ~2e-11 at the branch threshold). Exact-zero equilibrium fixed point,
floor semantics, branch continuity, and reverse-mode grads locked in.
When the driver lands, the end-to-end comparison (item 2 below) validates
this against the actual Fortran box rather than a transcription. It needs
the mode-mean `dmean = dgncur_awet·exp(1.5·alnsg²)` from the PREVIOUS step
— the lagged carried state plan 024 §6 describes.

### Remaining, in order

1. **The driver itself** — CAM's sequence on grid-cell means, sub-stepping
   exposed.
2. **Reference comparison** against `mam-box-fortran` at a pinned tag, for both
   `cam_mam4` and `cam_mam5`.

### Assumptions added since §5 was written

| # | Assumption | Status |
| --- | --- | --- |
| **A9** | Comparing dimensionless fractions is sufficient to validate the algorithm | ✅ **Confirmed.** Avoids the unit mapping entirely, and the divergent-regime points show it discriminates |
| **A10** | The five original capture points are representative | ❌ **Was wrong, now fixed.** They all sat in the saturated regime where both branches agree. Six divergent-regime points added |
| **A11** | `qaer_cur` is post-growth and the delta is informational | Verified against the Fortran reference: it conserves against `qaer_cur` to 0.0 and against `qaer_cur + delta` to 2.4e-3 |

---

## 7. The driver itself — sub-plan (started 2026-08-26, owner go-ahead "Let's do it")

**A6 is resolved**: owner approved proceeding with the driver including
sub-stepping exposed and defaulting ON (2026-08-26). The default substep
COUNT is still to be picked empirically in G5 (smallest n bringing a 30 s
step within ~1 % of the converged answer on the reference scenario).

### Findings from the source read that reshape §1's table

- **A1 correction — CAM's `gas_aer_uptkrates` is a THIRD variant, not the
  ported one.** The "bit-identical 58/58" claim compared CAM against
  E3SM-*legacy*. The JAX port (`_gas_aer_uptkrates_1box1gas`) implements
  the *amicphys* variant: Knudsen-dependent β, caller-supplied
  accommodation/diffusivity/free path, exact √π/√2. CAM's
  (modal_aero_gasaerexch.F90:953-1086): **fixed β = 2**, hardcoded
  ac = 0.65 literals (`0.4875`, `1.184`), truncated `tworootpi = 3.5449077`
  / `root2 = 1.4142135`, its own `gasdiffus = 0.557e-4·T^1.75/p`,
  `gasspeed = 14.70·√T`, and the result is multiplied by number
  concentration and gasdiffus inside. Must be ported, not reused.
- **The legacy 8.0-monolayer aging runs INSIDE CAM's gasaerexch** (:719-806,
  `modefrm_pcage` block, `dr_so4_monolayers_pcage` from the gasaerexch
  module parameter = **8.0**), and AGAIN inside `modal_aero_coag_sub` for
  the coagulated shell (modal_aero_coag.F90:502-546). This closes the
  #75-review attribution question for good: the CAM code line genuinely
  ages at 8.0 through the legacy path; E3SM/amicphys receives 3.0 via
  phys_control. Both are now facts of their respective drivers, not a knob
  disagreement.
- **`DGNUM` pbuf-initialises to 0.0** (modal_aero_calcsize.F90:131), so a
  reference run with calcsize/wateruptake off is garbage — END-TO-END
  parity requires topology-threaded calcsize + wateruptake (the deferred
  plan-024 PR C "~66 call sites" job, scoped to what the driver calls).
  The MICROPHYSICS sequence (gasaerexch → newnuc → coag) can be validated
  first against isolated captures, which need no calcsize.
- ~~The box's `troplev = pver` asymmetry means Köhler-only water uptake~~
  **CORRECTED during G5**: under `strat` the box driver calls
  `tropopause_set_box_level(pver+1)` (mam_box_driver_cam.F90:182), so
  wateruptake's `k < troplev` strat branch IS live — the wt%-composition
  solution volume replaces Köhler for every mode. Ported as
  `wateruptake(strat=...)`.
- **MAM5 reference runs pin `nl_acc_crs = 0`** — the box defaults
  `modal_accum_coarse_exch` to ON under MAM5, but rename-A2 (636 lines)
  stays deferred per plan 024 (measured inert below qso2 ~1e-5); the
  reference must be captured with the same setting the port implements.
- **PR #73's branch (`feat/cam-mam5-topology`) is merged into this branch**
  — the driver is the first real consumer of `CAM_MAM4`/`CAM_MAM5`. The
  topologies carry no molecular weights; `specmw_amode`/`adv_mass` must be
  dumped from the initialised box model like the index tables were (same
  argument: `adv_mass` comes from the chemistry preprocessor).

### Commit-sized steps

| Step | Content | Validation |
| --- | --- | --- |
| **G0** ✅ | Extend `mam-box-fortran/tools/dump_tables` to emit per-slot `SPECMW_AMODE`, the gas-window `ADV_MASS`, `CNST_NAMES`, `MWDRY`; regenerate both topologies; land the values as `mam4_jax/core/cam_params.py` (generated, sha-stamped) | **Done.** `tests/test_cam_params.py`: alignment, every pointer resolving to the right tracer NAME, plan 024 §3 census values, mechanism gas MWs |
| **G1** ✅ | `mam4_jax/coupling/cam_driver.py`: CAM `gas_aer_uptkrates` (third variant) + SO4-only `gasaerexch_cam` — fgain/avg_uprt trop path, reversible strat path (ported), the gasaerexch aging block at 8.0, rename-A1 call, tendency application. All topology-threaded (`Topology` argument; tables cached per name) | **Done.** `tools/capture_gasaerexch` (24 branch-pinning cases × both topologies, real `modal_aero_gasaerexch_sub`, prescribed diameters): worst rel-err **1.1e-15** on every quantity that is not a cancellation sliver of its own input. The post-aging pcarbon number (a `q·10ε` remnant) differs up to ~5% *of the sliver* (~1e-15 of the tracer): the reference binary FMA-contracts `q + dqdt·Δt` (gfortran `-O2`, arm64 `-ffp-contract=fast`) — verified bit-for-bit by reproducing both chains — so the bar is rtol 5e-13 + per-slot atol `1e-13·q_in`. Parity over the FULL Fortran subroutine also confirms **A13** (zero SOA is a soaexch fixed point) empirically |
| **G2** ✅ | CAM `modal_aero_newnuc_sub` wrapper over the ported nucleation leafs (`del_h2so4_gasprod`/`aeruptk` semantics; `mw_so4a_host` threaded into the dispatcher as an optional arg exactly as the Fortran passes it — E3SM default untouched). Includes `physics/cam_saturation.py`: CAM's generic `qsat` is the mixed-phase TABLE (`estblf`, 250 entries, water/ice blend over 20 K) — NOT the direct over-water formula (~2× apart at 200 K) | **Done.** `tools/capture_newnuc` (96 cases × both topologies): worst rel-err **2.0e-11**, the ~1-ulp form differences (`expm1`, table interp) amplified by nucleation's ~10th-power H2SO4/RH sensitivity; gated 1e-10. Sulfur closure exact per case; cutoff/floor gates exercised both ways |
| **G3** ✅ | CAM `modal_aero_coag_sub` port (pair_option 3: ait→acc, pca→acc, ait→pca-effective-accum + coag-side 8-monolayer aging), reusing `getcoags_wrapper_f` | **Done.** `tools/capture_coag` (32 cases × both topologies, all three number-solve branches, saturated + fractional aging): worst non-sliver rel-err **6e-16**. Found en route: `shr_const_rgas` is the PRODUCT `6.02214e26·1.38065e-23 = 8314.467591` — the rounded `8314.46` was 9.1e-7 off and the implicit number solves amplified it to ~2e-6. Conservation/monotonicity/untouched-mode (coarse, coarse_strat) invariants asserted |
| **G4a** ✅ | `mam_microphysics_cam`: the exact one-call chain gasaerexch → newnuc → coag with the `del_h2so4_aeruptk` positive-down bookkeeping (aero_model.F90:1191-1214). Sub-stepping deliberately NOT here — it wraps the whole per-step physics in the driver so `n_substeps = n` ≡ running the box at `deltat/n`, the exact quantity the Fortran dt-study varied | **Done.** `tools/capture_microphys` (12 cases × both topologies, trop + both strat kinds): worst non-sliver rel-err **1.5e-12** (the G1-G3 ulp differences compounded through three chained stages). Sulfur closure through the chain at 1e-13; a test proves the aeruptk bookkeeping is live (zeroing it changes the answer) |
| **G4b** ✅ | calcsize + wateruptake threaded via optional `tables` bundles (`CalcsizeTables`/`WateruptakeTables`; `None` default = the E3SM module constants, bit-identical — suite proves it); wateruptake gains `qv=` (CAM keeps water vapor outside the aerosol window) and `strat=` (the wt%-composition solution-volume branch, wateruptake_sub:583-591 — live in the box because the driver sets its tropopause above the single level); `cam_run_step`/`cam_run_timesteps` assemble SO2 stub → calcsize → sulfeq → wateruptake → mmr↔vmr → microphysics, substep loop wrapping the WHOLE step | **Done.** See G5 |
| **G5** ✅ | End-to-end vs the `mam-box-fortran` **fixdumfac** builds: {cam_mam4, cam_mam5} × {trop, strat}, all-default namelist, 120 steps × dt 30 s | **Done — first MAM5 physics against an independent reference.** Every printed tracer within the reference's own 7-significant-digit print floor (~5e-7; gated 2e-6); the one full-precision column, total sulfur, agrees at **4.5e-15** on all four trajectories. `tests/test_cam_driver.py` + `tests/reference/cam_box/` |
| **G5** | End-to-end vs `mam-box-fortran` at a pinned tag: `cam_mam4` and `cam_mam5` (`nl_acc_crs=0`), trop + strat scenarios; pick the substep default empirically | acceptance bar proposed after measuring, ADR to record it |


### G5 findings (2026-08-26)

1. **The lagged-`dgncur_awet` feedback never survives a step in the box
   reference**: the vendored time-manager shim's `is_first_step()` is TRUE
   every step (`time_manager.F90:13-22`; the driver never advances it), so
   wateruptake re-seeds `dgncur_awet = dgncur_a` each step and sulfeq is
   computed from fresh post-calcsize DRY diameters. Production CAM lags
   genuinely. The port exposes `reseed_dgnwet_each_step` (default True =
   the reference's behaviour); before this was found, the strat trajectory
   was 2x off by step 9.
2. **A6 substep measurement** (dt=30 s, default scenario, vs n=32):

   | n_substeps | num_a2 | num_a1 | so4_a1 | h2so4 |
   |---|---|---|---|---|
   | 1 (CAM-faithful) | 26% | 60% | 57% | 78% |
   | 2 | 16% | 34% | 31% | 51% |
   | 4 | 9% | 17% | 16% | 27% |
   | 8 | 4% | 8% | 7% | 13% |
   | 16 | 1.4% | 2.5% | 2.2% | 4.4% |

   First-order splitting, exactly the Fortran study's 2.08x finding. **The
   shipped default is `n_substeps = 1`** — A6 said "default ON", but the
   #75-review convention (defaults reproduce the reference) takes
   precedence: the reference is un-substepped, and a default that makes
   plain runs ~60% different from every parity fixture repeats the
   n_so4_monolayers mistake. Hosts should pass `n_substeps >= 8`; the
   docstring and this table say so. ⚠ OWNER CALL: if A6's
   "default ON" should win instead, it is a one-line change plus an ADR.
3. The E3SM path is bit-unchanged by the threading (the `tables=None`
   defaults are the same module constants; full suite green throughout).
