# Plan 019 — M8: Cloud chemistry port (`cloudchem_simple_sub`)

**Status:** proposed (2026-05-28). Awaiting owner approval to move M8 from proposed → in progress per CLAUDE.md rule #3.
**Branch:** `diffrax-cloud` (the M8 integration branch off `diffrax`).
**GitHub milestone:** [M8 #3](https://github.com/reflective-org/MAM4-JAX/milestone/3).
**Supersedes / supplements:** the M8 stub section in `docs/PLANS.md` written under PR #50.

---

## 1. Scope

Port `cloudchem_simple_sub` from `mam4-original-src-code/box_model_utils/cloudchem_simple.F90` (137 LOC of which ~50 LOC are physics). This is the box-model's parameterized aqueous SO₂ → SO₄ conversion: callable from `driver.F90:1265`, gated on `mdo_cloudchem > 0 AND maxval(cld_ncol) > 1e-6`, currently a no-op stub in `mam4_jax/driver.py:70-79`.

### Scope decisions (settled 2026-05-28; ground for each in the owner discussion log)

| Decision | Value | Why |
| --- | --- | --- |
| Q1 — physics scope | `cloudchem_simple_sub` only | `cloudchem_simple_sub` *is* the box-model's aqueous SO₂→SO₄ path. The "full intr" alternative (`mam_amicphys_1subarea_cloudy`, ~555 LOC at `modal_aero_amicphys.F90:1504-2059`) is structurally orthogonal (cloudy-subarea amicphys orchestration) — tracked separately as M14 (milestone [#9](https://github.com/reflective-org/MAM4-JAX/milestone/9)). |
| Q2 — fixture `cldn` | **`cldn = 0.5`** constant across the column | `cloudchem_simple_sub` internally cycles when `cldn ≤ 0.009`, so the driver-level gate `cldn > 1e-6` is necessary but not sufficient. A realistic cloud fraction (0.3–0.8) exercises the body. 0.5 picked as mid-cloud baseline. Time-varying `cldn` deferred to M9 (calibration demo). |
| Q3 — sulfur scope flip | **Bounded flip in FEATURES.md** in PR-K3 | `cloudchem_simple_sub` *is* sulfur chemistry. FEATURES.md's "sulfur chemistry beyond stubs is out of scope" line becomes "cloudchem_simple's parameterized SO₂→SO₄ ported; explicit-kinetics aqueous chemistry (H₂O₂/O₃ pathways) remains out of scope." |
| Q4 — validation bar | **Start at ADR-015's 3 % / 24 h / dt ≤ 5 s; tighten empirically** if 60-step residual proves it | `cloudchem_simple_sub` is algebraic (no ODE), so the diffrax structural offset doesn't apply directly. BUT downstream amicphys reads modified `q[H2SO4]` (nucleation consumes `qgas_avg[H2SO4]`), so trajectory diff could amplify through nucleation. Honest: start permissive, tighten if measured residual allows. |

### Out of scope (deferred)

- `mam_amicphys_1subarea_cloudy` (cloudy-subarea amicphys orchestration) → **M14**.
- Time-varying `cldn` profiles → **M9** (calibration demo's natural target).
- Multi-column `cldn` variation → **M12** (multi-column / multi-level).
- Explicit-kinetics aqueous chemistry (H₂O₂ / O₃ in-cloud SO₂ oxidation) → **out of scope indefinitely**; not in the reference Fortran tree's box-model path.

---

## 2. Physics surface

`cloudchem_simple_sub` body (per gridcell `(i, k)`, per timestep `deltat`):

```
if (cldn(i,k) <= 0.009)  CYCLE        ! sub-cloud threshold
tmpf = min(1.0, cldn(i,k))             ! cloud-fraction weight

! num distribution (Aitken vs accum)
tmpd = max(qqcw[num_c1], 1.0)          ! accum number  (denom protect)
tmpe = max(qqcw[num_c2], 0.0)          ! Aitken number
tmpd = tmpd / (tmpd + tmpe)            ! accum fraction
tmpe = max(0.0, 1.0 - tmpd)            ! Aitken fraction

! gases  (q is mol/mol-air)
tmpa = tmpf * q[SO2]   * exp(-deltat / tau)     ! SO2 lost (tau = 1800 s)
tmpb = tmpf * q[H2SO4]                          ! all in-cloud H2SO4 transfers
q[SO2]    -= tmpa
q[H2SO4]  -= tmpb

! cloud-borne sulfate
qqcw[so4_c1] += tmpd * (tmpa + tmpb)
qqcw[so4_c2] += tmpe * (tmpa + tmpb)

! cloud-borne ammonium (only if NH3 species exists in this config)
if (l_nh3g > 0 ...) then
   tmpc = min(tmpa + tmpb, tmpf * q[NH3])
   q[NH3]      -= tmpc
   qqcw[nh4_c1] += tmpd * tmpc
   qqcw[nh4_c2] += tmpe * tmpc
end if
```

### Tracer indices touched

- **Gas pcnst slots**: `l_h2so4g = LMAP_GAS[1]`, `l_so2g`, `l_nh3g` (the last two **not yet in `mam4_jax/data.py`** — need to capture via instrumentation in PR-K1; see open questions in §6).
- **Cloud-borne number** (per mode): `l_num_c1 = NUMPTRCW_AMODE[accum]`, `l_num_c2 = NUMPTRCW_AMODE[aitken]`.
- **Cloud-borne sulfate** (per mode): `l_so4_c1 = LPTR_SO4_CW_AMODE[accum]`, `l_so4_c2 = LPTR_SO4_CW_AMODE[aitken]`.
- **Cloud-borne ammonium** (per mode): `l_nh4_c1`, `l_nh4_c2` — only used if NH3 exists.

The `_CW_AMODE` and `LPTR_*_CW_AMODE` index tables touch the `qqcw` (cloud-borne) array, which has not been a write target in the JAX port to date. **Risk:** these tables may be sentinel-filled in `mam4_jax/data.py` (PR-K1 verifies and extends if needed).

### Constants

- `tau_cloudchem_simple = 1800.0` s (30-min e-folding for SO₂ uptake in cloud water). Module-level constant in `mam4_jax/processes/cloudchem.py`.

---

## 3. Sub-PR breakdown

Three commit-sized sub-PRs per CLAUDE.md rule #2. Each lands on its own feature branch off `diffrax-cloud` and targets `diffrax-cloud`.

### PR-K1 — Reference capture (`mdo_cloudchem=1, cldn=0.5`)

Capture the Fortran reference the JAX port validates against.

**Deliverables**
- `scripts/patches/cloudchem_set_cld.patch` — one-hunk patch changing `driver.F90:591` from `cld = 0.0_r8` to `cld = 0.5_r8`. Preserves the read-only Fortran tree per ADR-001 / ADR-012 (patches applied to transient build copies).
- `scripts/capture_reference.py` extended with a new mode: `--mode instrumented-cloudchem-only` (analogous to `instrumented-amicphys-off`) — full physics + cloudchem; namelist `mdo_cloudchem=1`; overlay applies `cloudchem_set_cld.patch` + `skip_pcarbon_aging.patch`.
- Instrumented dumps around the `cloudchem_simple_sub` call at `driver.F90:1265` → `tests/reference/per_process_cloudchem/cloudchem_{before,after}.npz`. Captures the `q` and `qqcw` arrays before/after the cloudchem call.
- End-to-end trajectory NetCDFs for the 4-dt 24 h sweep at `mdo_cloudchem=1, cldn=0.5` → `tests/reference/sweep_24h_cloudchem/mam_dt{1,5,30,300}_*.nc`. Used by PR-K3's end-to-end test.
- Possibly extend `dump_indices()` to capture `LPTR_SO4_CW_AMODE`, `LPTR_NH4_CW_AMODE`, `NUMPTRCW_AMODE`, gas pcnst indices for SO2/NH3 — refresh `tests/reference/indices/reference.npz` and hard-code into `mam4_jax/data.py` (only if currently sentinel-filled; see open questions §6).
- `tests/reference/SCHEMA.md` adds sections for the new cloudchem reference + sweep fixtures.

**Expected diff size**: ~150 LOC (patch + capture mode + SCHEMA), plus ~30–50 MB of LFS NetCDFs.

### PR-K2 — Port `cloudchem_simple_sub` to JAX (per-process validation)

Implement the JAX port. Pure function, JIT-friendly.

**Deliverables**
- `mam4_jax/processes/cloudchem.py`:
  - `cloudchem_simple_sub(state) -> state` — the core JAX function.
  - Module constant `TAU_CLOUDCHEM_SIMPLE = 1800.0`.
  - Index look-ups via the `mam4_jax.data` accessors (no string-based lookups inside the JIT scope).
  - `jnp.where(cldn > 0.009, body_tendencies, 0.0)` instead of the Fortran cycle (JIT-friendly).
  - Safe-division pattern (`jnp.maximum(qqcw[num_c1], 1.0)`) mirroring the Fortran's `max(qqcw, 1.0)`.
- `tests/test_cloudchem.py::test_cloudchem_simple_matches_fortran_per_process` — per-process validation against PR-K1's `cloudchem_{before,after}.npz`. Bar = ADR-003's 1e-6 (algebraic step, no ODE — expect ε-level residual).
- `docs/figures/cloudchem_residuals.png` — residual plot for the seven tracers cloudchem modifies (H2SO4, SO2, NH3, so4_c1, so4_c2, nh4_c1, nh4_c2).
- `docs/PROGRESS.md` entry summarizing the port.

**Expected diff size**: ~100 LOC of JAX + ~80 LOC of test + 1 plot.

### PR-K3 — Wire into driver, end-to-end trajectory, FEATURES flip

Replace the `cloud_chem_simple_sub` no-op in `driver.py` with the real implementation, validate the full operator-splitting trajectory, and update docs to reflect the new capability.

**Deliverables**
- `mam4_jax/driver.py`:
  - `cloud_chem_simple_sub` becomes a thin wrapper around `cloudchem.cloudchem_simple_sub`, gated on `state.get("mdo_cloudchem", 0) > 0` to preserve existing fixture tests (they default to 0 = no-op).
  - The module docstring's "Cloud-chem placement" section updated to reflect that cloud-chem is now live (not a structural no-op).
- `tests/test_driver.py`:
  - New `test_run_timesteps_60step_with_cloudchem_matches_fortran` against PR-K1's end-to-end fixture (`mdo_cloudchem=1, cldn=0.5`). Bar starts at ADR-015's 3 %.
  - Existing trajectory tests (the cldn=0 fixture) unchanged — verifies the no-op default preserves backwards compatibility.
- `tests/test_sweep.py` extended (or sibling `test_sweep_cloudchem.py` added) with the new 4-dt 24 h sweep at cldn=0.5. ADR-015's 3 % / dt ≤ 5 s bar; dt = 30, 300 diagnostic-only per the existing convention.
- `docs/FEATURES.md`:
  - "Cloud chemistry (parameterized aqueous SO₂ → SO₄)" added as a new row in the "Microphysical processes" table with status `ported (validated)` + this PR's link.
  - "Out of scope" section updated: "Sulfur chemistry beyond the placeholder `cloudchem_simple` stub" → "Explicit-kinetics aqueous sulfur chemistry (H₂O₂/O₃ pathways)" remains out of scope.
- `docs/PLANS.md` M8 status: proposed → done (this PR is the closing one for the milestone).
- `docs/PROGRESS.md` end-to-end validation entry.

**Expected diff size**: ~80 LOC of driver wiring + ~120 LOC of integration tests + docs.

---

## 4. Validation strategy

| Layer | PR | Fixture | Bar | Expected |
| --- | --- | --- | --- | --- |
| Per-process | PR-K2 | `per_process_cloudchem/cloudchem_{before,after}.npz` (single-step) | ADR-003 1e-6 | Machine ε (algebraic) |
| End-to-end 60-step | PR-K3 | new 60-step fixture from PR-K1 | ADR-015 3 % | TBD — measure, tighten if proven |
| 24 h sweep | PR-K3 | `sweep_24h_cloudchem/mam_dt*.nc` | ADR-015 3 % at dt ≤ 5 s; diagnostic at dt ≥ 30 | TBD |
| No-regression | PR-K3 | existing cldn=0 fixtures | ADR-003 1e-6 | Unchanged (`mdo_cloudchem=0` default keeps no-op) |

If PR-K3's measured 60-step rel-err is ≤ 1e-6, document the tighter bar in the PR description and keep the 24h sweep at 3 % (24h is where the diffrax structural offset accumulates regardless of cloudchem).

---

## 5. Risks / known unknowns

1. **`_CW_AMODE` index tables.** `LPTR_SO4_CW_AMODE`, `LPTR_NH4_CW_AMODE`, `NUMPTRCW_AMODE` may be sentinel-filled in `mam4_jax/data.py` — we've never written `qqcw` outside the all-zero IC. PR-K1 verifies and extends the `dump_indices()` overlay if needed. **First load-bearing investigation in PR-K1.**
2. **Cloud-borne tracer downstream coupling.** This is the first time the JAX port produces non-zero `qqcw[so4_c1/c2, nh4_c1/c2]`. Need to verify amicphys doesn't read these values via a path we haven't exercised (clear-subarea path shouldn't, but worth confirming with a single-toggle test).
3. **Gas-tracer amplification through nucleation.** `cloudchem_simple_sub` modifies `q[H2SO4]`. Next driver step's amicphys reads `q[H2SO4]` (nucleation uses `qgas_avg[H2SO4]`). Algebraic ε-level diff in cloudchem could compound through nucleation. Bar choice (Q4 = 3 % start) hedges this.
4. **`l_nh3g` may be -1 in this MAM4-MOM config.** The `if (l_nh3g > 0 ...)` Fortran branch guards against missing NH3. JAX port needs the equivalent. PR-K1's `dump_indices()` extension settles whether NH3 is present.
5. **`cldn = 0.5` constant choice.** Picked as mid-cloud; could be 0.3 or 0.8. If physically motivated, document the choice in PR-K1's plan. If chosen for convenience, flag it. (Recommendation: 0.5 unless owner has a specific physical regime in mind.)

---

## 6. Open questions before PR-K1 starts

The following need owner answers (or my recommended defaults can stand if unaddressed):

1. **Cldn value**: lock in 0.5? (Default: yes.)
2. **NH3 presence**: confirm via `dump_indices()` whether NH3 (`l_nh3g`) exists in this MAM4-MOM config. (Default: PR-K1 instruments and reports; if NH3 is absent, the JAX port's NH3 branch becomes structurally dead and a per-process test exercises only the SO2/H2SO4 path.)
3. **Cw index tables**: extend `dump_indices()` patch to capture `LPTR_*_CW_AMODE` and `NUMPTRCW_AMODE`? (Default: yes, if they're sentinel-filled — settled by PR-K1's first commit.)
4. **NetCDF schema**: does PR-K1's new sweep fixture's NetCDF need to add cloud-borne tracer columns to the output? (Default: yes — `mam_output.nc` already has slots; this is a Fortran-side question about what variables get written. If the existing schema doesn't expose `qqcw[so4_c1]` etc., we'd need an instrumentation extension.)
5. **Acceptance of bar relaxation**: if PR-K3's measured 60-step bar is between 1e-6 and 3 %, do we tighten to the measured value, leave at 3 %, or pick a round intermediate (1 %, 1e-3)? (Default: tighten to the measured value, documented in the PR.)

---

## 7. Why NOT include the full `modal_aero_cloudchem_intr` in M8

(Captured here as the load-bearing scope rationale; M14 milestone has the full detail.)

- `cloudchem_simple_sub` IS aqueous SO₂ → SO₄, just parameterized as a relaxation (`τ = 1800 s`) rather than explicit H₂O₂ / O₃ kinetics. It's what the box-model authors chose to ship as the cloud-chem path; porting it is honest and bounded.
- `mam_amicphys_1subarea_cloudy` (~555 LOC at `modal_aero_amicphys.F90:1504-2059`) is structurally orthogonal — it runs the same four sub-processes (gasaerexch, rename, newnuc, coag) on the cloudy fraction of the gridcell, separately from the clear-subarea path already ported in M3.6. Different call site, different physics. **M14** is where it goes.
- Doing both in M8 would conflate two structurally separable scopes and bury the cloudchem-simple deliverable in a much larger PR sequence.

---

## 8. Decision log

Logged per CLAUDE.md rule #1 ("plan first") and rule #11 ("transparency").

| Date | Decision | Notes |
| --- | --- | --- |
| 2026-05-28 | M8 scope = `cloudchem_simple_sub` only | M14 created for the structurally orthogonal `mam_amicphys_1subarea_cloudy`. Owner explicitly flagged the latter not to be forgotten. |
| 2026-05-28 | Fixture = `cldn = 0.5`, `mdo_cloudchem = 1`, `skip_pcarbon_aging` maintained | Threshold ≥ 0.009 needed to exercise the body; 0.5 picked as mid-cloud baseline. |
| 2026-05-28 | Validation bar = 3 % start, empirical tightening allowed | Algebraic step suggests tighter bar is achievable but downstream amicphys coupling could amplify; honest start permissive. |
| 2026-05-28 | FEATURES.md sulfur clause = bounded flip in PR-K3 | "Explicit-kinetics aqueous chemistry (H₂O₂/O₃) remains out of scope." |

---

## 9. Pointers

- `docs/PLANS.md` Milestone 8 (this plan supersedes its scope sketch).
- `docs/PLANS.md` Milestone 14 (the structural follow-up).
- `docs/KEY_DECISIONS.md` ADR-001 (read-only Fortran tree), ADR-003 (1e-6 default bar), ADR-012 (patch-overlay pattern), ADR-015 (3 % / 24 h bar revision).
- `mam4-original-src-code/box_model_utils/cloudchem_simple.F90` (137 LOC source).
- `mam4-original-src-code/test_drivers/driver.F90:591, 1037, 1263-1270` (cld hardcode, cld_ncol propagation, call site).
- `mam4_jax/driver.py:70-79` (current no-op stub being replaced).
- GitHub: [M8 milestone #3](https://github.com/reflective-org/MAM4-JAX/milestone/3), [M14 milestone #9](https://github.com/reflective-org/MAM4-JAX/milestone/9).
