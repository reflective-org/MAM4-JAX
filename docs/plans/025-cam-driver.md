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
| **Uptake rates** (`gas_aer_uptkrates`) | ✅ **Yes** | CAM vs E3SM-legacy **bit-identical, 58/58 normalised lines** (condensation report). The JAX port already ports it |
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

### Remaining, in order

1. **`sulfate_equilib` condensation** — the `sulfeq` branch. CAM-only, so no
   existing port to lean on.
2. **The driver itself** — CAM's sequence on grid-cell means, sub-stepping
   exposed.
3. **Reference comparison** against `mam-box-fortran` at a pinned tag, for both
   `cam_mam4` and `cam_mam5`.

### Assumptions added since §5 was written

| # | Assumption | Status |
| --- | --- | --- |
| **A9** | Comparing dimensionless fractions is sufficient to validate the algorithm | ✅ **Confirmed.** Avoids the unit mapping entirely, and the divergent-regime points show it discriminates |
| **A10** | The five original capture points are representative | ❌ **Was wrong, now fixed.** They all sat in the saturated regime where both branches agree. Six divergent-regime points added |
| **A11** | `qaer_cur` is post-growth and the delta is informational | Verified against the Fortran reference: it conserves against `qaer_cur` to 0.0 and against `qaer_cur + delta` to 2.4e-3 |
