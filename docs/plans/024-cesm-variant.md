# Plan 024 — CESM/CAM variant of MAM4-JAX

**Status:** **PROPOSED** (2026-08-03). Not approved; nothing here is in progress.
Per `CLAUDE.md` §"Nothing moves from proposed to in progress without owner approval".
**Branch:** `docs/m15-cesm-variant-plan` → `main` (docs only; no code).
**Evidence base:** sibling repo `mam-cesm-box`, in particular `docs/CESM_VS_E3SM.md` and the
seven per-process reports under `docs/discrepancies/` (~6,200 lines, every claim carrying
`file:line` citations into both Fortran trees).

## 0. Verification provenance

Every claim about *this* repo below was re-verified against **`origin/main` @ `5ed4656`**
("Include package data (*.npz) in the built wheel", #62) after a full `git fetch`, not against
a stale local checkout.

`git diff 9a68e18..origin/main` touches only `docs/DEFERRED.md`, `docs/FEATURES.md`,
`docs/PROGRESS.md`, `mam4_jax/processes/amicphys.py`, `pyproject.toml`, and
`tests/test_amicphys.py`. The files this plan's findings rest on —
`mam4_jax/processes/calcsize.py`, `mam4_jax/data.py`, `mam4_jax/config.py`,
`mam4_jax/constants.py`, and the vendored `mam4-original-src-code/` tree — are **unchanged**.
Spot-checked directly on `origin/main`: `calcsize.py:470-472` still has no `1e6` bound
turn-off; `data.py:136` is still `0.010` for p-organic and `:521` still
`MW_SO4A_HOST = 115.0`.

CAM side is pinned to `cam6_4_187` (`d38ad70`), the exact SHA recorded for `components/cam`
in the CESM checkout — see `mam-cesm-box/PROVENANCE.md`.

**Re-verify before implementing.** If M15 is approved well after this date, redo the
`git diff <plan-base>..origin/main` check above before trusting §3 or §5.

---

## 1. Why this plan exists

MAM4-JAX is a faithful port of **E3SMv1** MAM4 via the PNNL `MAM_box_model`. CESM3/CAM
ships a *different* MAM code line. A full source-level investigation of both established a
result that determines the whole strategy:

> **The leaf kernels are the same code. The parameter values differ far more than the code
> does. The orchestration layer shares nothing.**

Concretely, verified:

| Kernel | CAM vs E3SM |
| --- | --- |
| `getcoags` + `getcoags_wrapper_f` + Whitby tables | **byte-for-byte identical**, 1519 lines, matching MD5s. All **4030** tabulated values bit-exact — *including against this repo's `_coag_tables.npz`* |
| `binary_nuc_vehk2002` | 0 line diffs, 214/214 literals exact |
| `ternary_nuc_merik2007` | 0 line diffs, 433/433 literals exact |
| Kerminen–Kulmala survival correction | byte-identical (constant +38-line offset) |
| `makoh_cubic` / `makoh_quartic` | byte-identical, whitespace included |
| `modal_aero_kohler` | zero CAM-only lines; CGS constants hardcoded identically |
| `gas_aer_uptkrates` | arithmetically identical (58/58 lines) |
| calcsize prognostic kernel | 92 diffs, 72 of them a mechanical `renamexf`→`csizxf` rename; **no formula differs** |
| rename default path | 27 diffs, **none** affecting arithmetic |

So this is **not** a re-port. It is a variant axis plus a second driver.

---

## 2. Scope

### In scope

1. A `variant: "e3sm" | "cesm"` configuration axis threaded through `data.py` / `config.py`.
2. CAM-faithful parameter values (§3), including the mode/species topology change.
3. A **CAM driver** implementing CAM's sequential grid-cell-mean coupling, as an alternative
   to `processes/amicphys.py`.
4. Fixing two defects found while comparing against the Fortran (§5). These are bugs in the
   **existing E3SM path** and should land first, independently.

### Deferred

| Item | Why |
| --- | --- |
| CAM's `sulfeq` reversible stratospheric-sulfate uptake | Off by default (`modal_strat_sulfate`); large standalone feature |
| CAM's `modal_aero_rename_acc_crs_sub` (636 lines) | Off by default (`modal_accum_coarse_exch`); a third rename algorithm |
| Generalised VBS (`ntot_soaspec = 2/5/15`) | CAM default for MAM4 is `nsoa = 1` |
| MAM5 / `coarse_strat` | Second topology; doubles the config surface |
| CAM's stratospheric water uptake | **Not a pure function** — see §6 |

### Explicitly not in scope

`processes/amicphys.py` is **not** portable to CAM and must not be modified to try. CAM has
no equivalent — `grep -rI amicphys` over all of CAM `src/` returns zero hits.

---

## 3. The parameter axis — the substance of this plan

All values verified in source on both sides. `mam-cesm-box/docs/CESM_VS_E3SM.md` has the
citations.

| Parameter | E3SM (current) | CESM/CAM | Factor | Where in this repo |
| --- | --- | --- | --- | --- |
| p-organic hygroscopicity | 0.010 | **1.0e-10** | **10⁸** | `data.py:138` |
| `opoa_frac` | 0.1 | **0.0** | 0 vs 0.1 | gasaerexch |
| `n_so4_monolayers_pcage` | 3.0 | **8.0** | 2.67× aging | config |
| `mw_soa` (internal) | 150.0 | **250.0** | 1.67× | gasaerexch/data |
| `delh_vap_soa` | 156.0e3 | **131.0e3** | → **~11× C\*** at 250 K | gasaerexch |
| `p0_soa_298` | 1.0e-10 | **9.7831e-11** | ~2 % | gasaerexch |
| `MW_SO4A_HOST` | 115.0 | **115.107340** | 0.093 % | `data.py:521` |
| sulfate/ammonium `specmw_amode` | 115.0 | **115.107340** | 0.093 % | `data.py` |
| p/s-organic, black-c `specmw_amode` | 12.0 | **12.011000** | 0.092 % | `data.py` |
| seasalt `specmw_amode` | 58.5 | **58.442468** | 0.099 % | `data.py` |
| dust `specmw_amode` | 135.0 | **135.064039** | 0.047 % | `data.py` |
| **aitken mode species** | so4, soa, ncl | so4, soa, ncl, **dst** | `nspec_amode` 3→4 | `data.py` topology |

### 3.1 These belong in a parameter module, not scattered through the science code

Owner direction (2026-08-03): rather than hardcoding a second set of values, **separate them
out into a parameter file so the hardcoded values have one known home and can be modified.**
That is the right call and it generalises past `delh_vap_soa` — the values above currently
live in at least four places (`data.py` tables, `config.py` dataclasses, module-level
constants in `processes/gasaerexch.py`, and inline literals).

A companion five-layer design already exists in the Fortran box model at
`mam-cesm-box/docs/PARAMETER_ABSTRACTION.md`; the layering is deliberately chosen to map onto
this repo:

| Layer | What | This repo |
| --- | --- | --- |
| 1 Physical constants | π, R, Avogadro, g, mwdry, ρ_water | `constants.py` |
| 2 Mode/species properties | sigma, dgnum, density, hygroscopicity, MW, topology | `data.py` |
| 3 **Process options** | `opoa_frac`, `p0_soa_298`, **`delh_vap_soa`**, `mw_soa`, `n_so4_monolayers_pcage`, scheme selectors | `config.py` |
| 4 **Numerical guards** | clipping floors, `nh3ppt < 0.1` switch, `cld ≥ 0.99` skip, tolerances | **no home yet — add one** |
| 5 Coefficient tables | Vehkamäki, Merikanto, Whitby | `_coag_tables.npz` + inline |

Two consequences for this plan:

- **Layer 4 has no home in this repo today.** Those thresholds are inlined, so they cannot be
  swept or varied on either side. Worth adding a `numerics.py` as part of PR B, because they
  are precisely the values that make two "identical" implementations diverge invisibly — the
  formulas match, so a line diff shows nothing.
- Every value in §3 should become a *named, documented* entry with its CAM and E3SM values
  side by side and a citation, so the variant axis is a table lookup rather than scattered
  `if variant == "cesm"` branches.

`mam-cesm-box/docs/discrepancies/constants-inventory.md` (1,309 lines) is the full census of
where the hardcoded values currently are, in both codebases, with `file:line` for each — use
it as the worklist for populating the parameter module.

### 3.2 Two notes that matter for sequencing

- **`delh_vap_soa` is the dangerous one.** It makes the disagreement strongly
  temperature-dependent, so any validation done only at 298 K will look fine and fail in the
  cold upper troposphere. Every SOA test in this plan must run at ≥2 temperatures.
- **The 0.093 % MW offsets are ~1000× the repo's 1e-6 tolerance.** They must be part of the
  variant axis, not treated as noise. They *do* cancel to < 2e-5 in coagulation's
  `xferfrac_pcage`, but not in condensation or nucleation mass budgets.

### What does NOT need a variant axis — assert these as invariants instead

Verified identical: every per-mode size parameter (`sigmag` 1.8/1.6/1.8/1.6, `dgnum`,
`dgnumlo`, `dgnumhi`, `rhcrystal` 0.35, `rhdeliques` 0.80); all species densities (so4 1770,
pom 1000, soa 1000, bc 1700, ncl 1900, dst 2600); black-carbon hygroscopicity (1.0e-10 both);
`newnuc_method_flagaa = 11`; `pair_option_acoag = 3`; `method_soa = 2`; the ternary→binary
threshold (`nh3ppt < 0.1`, i.e. `qnh3 < 1e-13`); and all 4030 Whitby / 113 Vehkamäki / 157
Merikanto coefficients.

**Do not "fix" the Köhler CGS island.** `pi = 3.14159`, `ugascon = 8.3e7`, `mw = 18.0`,
`tair = 273.0`, `surften = 76.0` dyn cm⁻¹ are deliberately imprecise private copies,
identical in both Fortran codes. E3SM contains a commented-out attempt to replace them with
`physconst` values that was **backed out**. Normalising them changes answers.

Caveat on exactness: CAM's physprop files carry float32 round-trip noise (mode 4
`sigmag = 1.60000002`, not 1.6). Bit-identical CAM/E3SM agreement is impossible by
construction; ~1e-8 is the floor.

---

## 4. Proposed PR breakdown

Sized so each PR is independently reviewable and testable.

| PR | Content | Depends on |
| --- | --- | --- |
| **A** | Fix the two E3SM-path defects in §5. No variant work. | — |
| **B** | Introduce the `variant` axis plumbing with `"e3sm"` as the only value, **plus the parameter-module consolidation of §3.1**: pull the scattered Layer-3 values into `config.py` and add a `numerics.py` for Layer 4. Pure refactor; E3SM path must be **bit-unchanged**. | A |
| **C** | CAM species properties: hygroscopicity, the four `specmw_amode` sets, `MW_SO4A_HOST`. | B |
| **D** | CAM topology: dust in the aitken mode (`nspec_amode` 3→4, index tables, `noxf_acc2ait`). Highest-risk PR — touches every index table. | C |
| **E** | CAM SOA parameters (`opoa_frac`, `p0_soa_298`, `delh_vap_soa`, `mw_soa`) + `n_so4_monolayers_pcage`. Tests at ≥2 temperatures. | C |
| **F** | `processes/cam_driver.py` — CAM's sequential coupling: gasaerexch (calling rename internally) → newnuc → coag, on grid-cell means. No sub-areas, no sub-stepping. | D, E |
| **G** | Reference-data capture from `mam-cesm-box` + a `tests/reference/cesm/` suite mirroring the existing E3SM structure. | F |

**Non-negotiable invariant across B–G:** the E3SM path stays bit-for-bit unchanged. The
existing test suite is the only guard against the variant axis silently altering it, so every
PR must show it green.

---

## 5. Defects to fix first (PR A) — bugs in the current E3SM path

Both found by diffing this repo against its own Fortran reference; both verified.

### 5.1 `processes/calcsize.py` omits the `do_aitacc_transfer` bound turn-off

`box_model_utils/modal_aero_calcsize.F90:756-762`:

```fortran
if ( do_aitacc_transfer ) then
   if (n == nait) v2nxx = v2nxx/1.0e6_r8    ! effectively turn off the bound
   if (n == nacc) v2nyy = v2nyy*1.0e6_r8
   v2nxxrl = v2nxx/frelaxadj
   v2nyyrl = v2nyy*frelaxadj
end if
```

`calcsize.py:470-473` sets `v2nxx`/`v2nyy`/`v2nxxrl`/`v2nyyrl` with **no such branch**, and
there is no `1e6` anywhere in the package. Since `do_aitacc_transfer` defaults to `True`, the
port **enforces** the aitken upper and accum lower bounds where the Fortran deliberately
disables them — clamping number the reference does not clamp.

Not caught today because the port's own docstring notes the transfer never triggers in the
reference run, i.e. this path is untested. PR A should add a case that exercises it.

### 5.2 Single-precision literal caps achievable tolerance

`modal_aero_amicphys.F90:4030` has `sqrt( 0.5 )` — a single-precision literal, ~1.71e-8
relative error, feeding `factoryy` and thence both `erfc` arguments. The JAX port uses
float64 throughout, so it cannot reproduce the Fortran below ~1e-8, putting a floor under
`test_rename.py`'s 1e-6 tolerance. Decide explicitly: match the Fortran's `sqrt(0.5f)` or
document the floor. Related: CAM clips per-species dry volume with `max(0, ·)`;
amicphys and this port do not.

---

## 6. Known blocker for any future CAM stratospheric work

CAM's water-uptake driver is **not a pure function**. Above the tropopause it abandons Köhler
theory for an H2SO4–H2O binary solution (Tabazadeh 1997 composition, CARMA σ/ρ tables,
Giauque 1959 enthalpy, Ayers 1980 + Kulmala 1990 vapour pressure) and carries a **lagged
feedback**: `dgncur_awet(t-1)` → Kelvin term → `wtpct`/`sulden` → `dgncur_awet(t)`. E3SM has
no tropopause concept at all.

A JAX port would need this as explicit carried state, which conflicts with the current
functional design. Hence §2 defers it and this plan is **tropospheric-only**. That deferral
should be an ADR if the plan is approved.

---

## 7. Acceptance criteria

1. E3SM path bit-for-bit unchanged; full existing suite green on every PR.
2. `variant="cesm"` reproduces `mam-cesm-box` Fortran output to 1e-6 relative per process, at
   ≥2 temperatures for anything SOA-related.
3. Every parameter in §3 covered by a test asserting the variant-specific value is actually
   in force — not merely present in a table.
4. §3's invariant list asserted at import or first call, so a future edit cannot silently
   diverge them.
5. `docs/DEFERRED.md` records every §2 deferral with its reason.

## 8. Open questions

### Resolved

- **`delh_vap_soa` 131 kJ vs 156 kJ.** Owner direction 2026-08-03: treat it as a
  parameter-file value rather than a hardcoded choice, so both can be selected and the
  provenance is visible. See §3.1. The underlying *scientific* question — which value is more
  defensible — stays open, but it no longer blocks the plan, because the answer becomes a
  config default rather than a code edit.
- **Branch naming / base.** This plan sits on `docs/m15-cesm-variant-plan`, cut from
  `main` @ `5ed4656` after a full fetch.

### Still open

1. Should the CAM variant land on `main` behind the `variant` flag, or on a long-lived
   branch? Recommend **`main` behind the flag** — long-lived branches drift against the
   diffrax/solver work, and this repo has already paid that cost once (`diffrax`,
   `diffrax-cloud`, `merge-back/diffrax-to-main`).
2. Is `nspec_amode` 3→4 (PR D) acceptable as a breaking change to the public data tables, or
   does it need a compatibility shim?
3. Should Layer 4 (`numerics.py`) land in PR B as proposed, or as its own PR ahead of the
   variant work? It has standalone value for the E3SM path.
