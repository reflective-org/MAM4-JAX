# Upstream Fortran findings

Bugs, sloppiness, and porter-surprising patterns found in the vendored MAM4 Fortran reference (`mam4-original-src-code/`, originally [`kaizhangpnl/MAM_box_model`](https://github.com/kaizhangpnl/MAM_box_model) at commit `4150e2d`) while building the JAX port. Filed here so the upstream authors can fix or ack/wontfix.

**Status legend**

- 🔴 **Bug** — almost certainly wrong; should be fixed upstream.
- 🟡 **Lint** — defensible but worth cleaning up.
- 🔵 **Observation** — not a bug, but porter-surprising; worth documenting upstream if not already.

Entries below are sorted by file, then line. Each links to the line in the vendored copy.

---

## 🔴 Stray debug `print *` statements in production code

**File:** `e3sm_src_modified/modal_aero_amicphys.F90`
**Lines:** 2553, 5265, 5266

Three live `print *` statements that emit to stdout on every timestep of every column where the conditional fires:

```fortran
! Line 2553 (inside mam_amicphys_1subarea_clear, pcarbon-aging block):
if (iaer == 2) print *, 'so4 condensed from H2SO4 to Primary carbon mode: ', qaer_delsub_cond(2,4)

! Lines 5265-5266 (inside mam_pcarbon_aging_1subarea):
if (iaer == 2) print *, 'so4 transfer from Primary carbon mode (coagulated from Aitken mode) to Accumulation mode due to aging: ', qaer_del_coag(iaer,nfrm)
if (iaer == 2) print *, 'so4 transfer from Primary carbon mode (condensed H2SO4) to Accumulation mode due to aging: ', qaer_del_coag(iaer,nfrm)
```

There are also two **commented-out** prints at lines 4997 and 5079 — same flavour, presumably the dev intended to remove all of them and missed three.

**Issues:**
1. Pollutes stdout (annoying for production runs, breaks anything that parses stdout).
2. At line 2553, `iaer` is not the active loop variable in the enclosing scope — the surrounding `do_pcarbon_aging_block` doesn't iterate on `iaer`. Whatever value `iaer` has at that point is a leftover from a prior loop, so the conditional fires based on a stale value. The behaviour depends on the prior loop's structure, which is fragile.
3. The hard-coded indices `qaer_delsub_cond(2,4)` (so4 in pcarbon mode) and `iaer == 2` (so4) are MAM4-MOM-specific and would silently break under a different aerosol-species ordering.

**Suggested fix:** delete all three live prints. If kept as a diagnostic, wrap them in `#if defined(CAMBOX_DEBUG)` or similar.

---

## 🔴 Hard-coded developer path in `run_test.csh`

**File:** `run_test.csh`
**Line:** the `outpath` assignment near the top.

```csh
set outpath = /Users/sunj695/...
```

References a specific developer's home directory; the post-run `mv mam_output.nc $outpath/...` fails for anyone else. Anyone who clones the repo has to either edit `run_test.csh` or skip its post-processing step.

**Suggested fix:** default `outpath` to something like `./run` or `${OUTPATH:-./run}` so it works for fresh clones; document the override via an environment variable.

---

## 🔴 `run_test.csh` sweep loop has a premature `exit`

**File:** `run_test.csh`

The loop iterates over the canonical `(1 2 4 9 18 30 60 120 180 360 900 1800)` timestep counts, but an `exit` statement inside the loop body fires after the first iteration. Only `mam_dt1800_ndt1.nc` ever gets written; the rest of the sweep never runs. The `mv` to `$outpath` happens AFTER the `exit`, so even that single iteration's output isn't archived.

Net effect: as shipped, `run_test.csh` produces nothing the user can post-process. We wrote our own Python sweep driver (`scripts/capture_reference.py`) to work around this.

**Suggested fix:** remove the spurious `exit`. Possibly also tighten the convergence test into a documented Python or shell harness.

---

## 🔴 Missing `&size_parameters` namelist group in `run_test.csh`

**File:** `run_test.csh` (the inline `cat << EOF > namelist` block)

The Fortran driver reads five namelist groups (`&time_input`, `&cntl_input`, `&met_input`, `&chem_input`, `&size_parameters`) but the upstream `run_test.csh` writes only the first four. The mandatory `&size_parameters` group (`dgnum`, `sigmag` per mode — defaults from `box_model_utils/rad_constituents.F90:167-170`) is absent, so the driver crashes on startup with "namelist not found" or uses uninitialized values, depending on the gfortran version.

The omission was introduced when the `CUSTOM_SIZE` compile flag was removed — `&size_parameters` was previously synthesized at compile time and no one updated the runtime harness.

**Suggested fix:** add the `&size_parameters` block to `run_test.csh`'s namelist heredoc with the rad_constituents defaults.

---

## 🟡 Legacy BOZ literals in `infnan.F90` reject under modern gfortran

**File:** `box_model_utils/infnan.F90`

Encodes IEEE Inf/NaN via octal BOZ literals (e.g. `O'0777...'`) assigned to `real(r8)` variables. gfortran 10+ rejects this by default with `Error: BOZ literal at (1) used to initialize non-integer variable`. Compilable only with `-fallow-invalid-boz`.

**Suggested fix:** replace the BOZ-literal initializations with `ieee_value(x, ieee_quiet_nan)` and `ieee_value(x, ieee_positive_inf)` from the `ieee_arithmetic` module (Fortran 2003 standard). Should work on every modern compiler without a permissive flag.

---

## 🟡 `intent(inout)` declared on unused arguments

**File:** `e3sm_src_modified/modal_aero_amicphys.F90`

Multiple subroutines declare arguments `intent(inout)` but never modify them:

| Subroutine | Argument | Lines |
| --- | --- | --- |
| `mam_rename_1subarea` | `qwtr_cur` | 3958–3960 |
| `mam_soaexch_1subarea` | `qnum_cur` | 3629–3630 |
| `mam_soaexch_1subarea` | `qwtr_cur` | 3631–3632 |

The Fortran standard permits this, but it misleads readers, prevents the compiler from catching real `intent` violations, and forces callers to pass mutable buffers unnecessarily.

**Suggested fix:** declare these as `intent(in)`. (Strictly: if a future revision plans to modify them, mark the intent then; don't pre-declare.)

---

## 🟡 Driver `mmr ↔ vmr` round-trip introduces ULP drift on untouched tracers

**File:** `test_drivers/driver.F90`
**Lines:** 1224 (mmr → vmr) and 1321 (vmr → mmr)

```fortran
! Line 1224 (every timestep, before amicphys):
vmr(...) = mmr(...) * mwdry/adv_mass(l2)
! Line 1321 (every timestep, after amicphys):
mmr(...) = vmr(...) * adv_mass(l2)/mwdry
```

The two conversion factors `mwdry/adv_mass` and `adv_mass/mwdry` are computed inline at each step; their FP product isn't exactly 1.0, so any tracer that amicphys *doesn't* touch drifts by 1 ULP per round-trip. For a number tracer of magnitude ~8e7 over 60 steps, that accumulates to ~28000 in absolute terms — small but easily large enough to make sub-process isolation testing surprising.

**Suggested fix:** precompute `mmr_to_vmr = mwdry / adv_mass` once at init, store as a module-level array, then write `mmr = vmr / mmr_to_vmr` for the inverse. The single divide undoes the original multiply exactly (within FP) for any normal value.

(In `mam4_jax/data.py` we use both `MMR_TO_VMR = mwdry/adv_mass` and a separately computed `VMR_TO_MMR = adv_mass/mwdry` deliberately, to reproduce Fortran's drift bit-for-bit. If upstream fixes this, the JAX side becomes `q * MMR_TO_VMR / MMR_TO_VMR` — bit-exact identity.)

---

## 🔵 `mam_pcarbon_aging_1subarea` isn't gated by any `mdo_*` flag

**File:** `e3sm_src_modified/modal_aero_amicphys.F90`
**Lines:** 1986, 2555 (the two call sites)

The four "headline" sub-processes inside amicphys (gasaerexch, rename, newnuc, coag) are each individually gated by `mdo_gasaerexch / mdo_rename / mdo_newnuc / mdo_coag` namelist flags. But `mam_pcarbon_aging_1subarea` runs unconditionally as long as `n_agepair > 0`. It transfers so4 mass from primary-carbon to accumulation, which is correct physics, but **per-process testing** (e.g. running with `mdo_gasaerexch=1, others=0` to isolate gasaerexch's effect) gets surprising results because pcarbon aging keeps running.

This isn't a bug per se — aging conceptually has nowhere else to live in the timestep — but a porter or someone validating gasaerexch in isolation will spend time wondering why "gasaerexch-only" doesn't actually isolate gasaerexch. We worked around it with a `skip_pcarbon_aging.patch` overlay for our single-toggle reference captures.

**Suggested:** either gate pcarbon-aging behind a separate `mdo_aging` namelist flag, or at least document in `modal_aero_amicphys.F90`'s header comment that aging runs unconditionally.

---

## 🔵 `update_aerosol_props` mutates size fields during the cond sub-stepping loop

**File:** `e3sm_src_modified/modal_aero_amicphys.F90`
**Line:** 2411–2412 (inside `mam_amicphys_1subarea_clear`)

When `do_cond_wateruptake = .true.` (the default), gasaerexch's sub-stepping loop re-runs `update_aerosol_props` after every substep, which itself re-runs the wateruptake (Köhler) solver on the updated `qaer`. So `dgncur_awet`, `qaerwat`, and `wetdens` change during gasaerexch — even though these are conceptually wateruptake's outputs, not gasaerexch's.

This is correct physics (the new SOA mass condenses into wetter particles whose diameter shifts) but it surprises anyone validating gasaerexch's outputs at machine ε on the size fields. We documented this in `tests/test_amicphys.py::test_orchestration_gasaerexch_matches_fortran` (which uses 1e-3 tolerance on the size fields instead of 1e-6).

**Suggested:** add a comment near the call site noting that the size fields are now amicphys-output fields too, not just wateruptake's.

---

## 🔵 The standalone `modal_aero_{rename,gasaerexch,newnuc,coag}.F90` are partly dead code

**Files:** `e3sm_src/modal_aero_{rename,gasaerexch,newnuc,coag}.F90`

The box-model driver calls `modal_aero_amicphys_intr`, which has its own self-contained internal copies (`mam_rename_1subarea`, `mam_gasaerexch_1subarea`, `mam_newnuc_1subarea`, `mam_coag_1subarea`) of all four sub-processes. The top-level subroutines in the standalone files (e.g. `modal_aero_rename_sub`) are never reached from the box-model driver.

**But** — the standalone files still carry *some* live code that's used by amicphys via `use … , only :` imports of lower-level helpers. For example:

- `modal_aero_newnuc.F90`: top-level `modal_aero_newnuc_sub` is dead, but `mer07_veh02_nuc_mosaic_1box`, `binary_nuc_vehk2002`, `pbl_nuc_wang2008`, and the parameter `qh2so4_cutoff` ARE used by `mam_newnuc_1subarea`.
- `modal_aero_calcsize.F90`: lower-level helpers may be used too — we haven't audited.

The mix is confusing for porters. We initially treated the whole `modal_aero_newnuc.F90` as dead code (documented in this repo's PR #12) and only realized our error during PR-F planning when we read `mam_newnuc_1subarea`'s `use` statement.

**Suggested:** add a header comment to each standalone `modal_aero_*.F90` file noting which symbols are used by the box-model driver path vs which symbols are reachable only from other (non-box-model) configurations. Or split the live helpers into a separate module.

---

## 🔵 Hard-to-spot 3-digit-exponent format dropping the `e` separator

**Not specifically a bug in MAM4** — a generic Fortran formatting gotcha — but worth flagging because it affects anyone wiring instrumentation around the model.

The default `es24.16` format leaves only 6 chars for sign, decimal point, `e`, exponent sign, and exponent digits. When a value has a 3-digit negative exponent (e.g. Vehkamäki nucleation rates ≈ `1e-100`), gfortran silently drops the `e` separator, writing `1.33-116` instead of `1.33e-116`. Python's `float()` then can't parse the result.

**Workaround:** use `1pe27.16e3` or wider whenever the value range includes 3-digit exponents. We did this in `scripts/reference_drivers/newnuc_helpers_driver.F90`; the existing `makoh_driver.F90` etc. didn't hit it because their value ranges are narrower.

---

## How to update this doc

Add a new entry when porting reveals a new finding. Keep the categories ordered (🔴 then 🟡 then 🔵). Reference specific files and lines from the vendored snapshot — if those line numbers shift in a future upstream version, note that in the entry.

If a finding gets fixed upstream and we pull a refreshed snapshot, move that entry to a "Resolved upstream" section at the bottom (rather than deleting) so the historical record stays.
