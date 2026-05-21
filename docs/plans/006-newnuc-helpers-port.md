# Plan 006 — M3.6 PR-F1: leaf nucleation parameterizations

> **Status:** approved 2026-05-21.

---

## Context

PR-F (the full `mam_newnuc_1subarea` port) splits into three sub-PRs after the owner-approved scope split (2026-05-21):

- **PR-F1** (this plan) — leaf parameterizations only: `binary_nuc_vehk2002` and `pbl_nuc_wang2008`. Validated via standalone Fortran driver, no amicphys involvement.
- **PR-F2** — `mer07_veh02_nuc_mosaic_1box` (dispatcher + Kerminen-Kulmala 2002 size correction).
- **PR-F3** — `mam_newnuc_1subarea` (amicphys orchestration) + wiring + end-to-end test.

PR-F1 mirrors the polysvp/qsat/makoh/kohler pattern — pure-scalar math, standalone Fortran driver sweep, JAX port, 1e-6 validation. Lowest-risk, lowest-LOC sub-PR.

## Scope

| Fortran function | Lines | Role |
| --- | --- | --- |
| `binary_nuc_vehk2002` | `modal_aero_newnuc.F90:1256-1448` (~193) | Vehkamäki et al. (2002) H₂SO₄–H₂O binary nucleation rate + critical-cluster size. Pure polynomial parameterization in (T, ln RH, ln [H₂SO₄]). |
| `pbl_nuc_wang2008`    | `modal_aero_newnuc.F90:1179-1255` (~77)  | Wang & Penner (2008) boundary-layer overlay: candidate rate from `1e-6 * so4vol` (flagaa=11) or `1e-12 * so4vol²` (flagaa=12); wins if higher than the prior. |

Out of scope for PR-F1 (handled later):
- `mer07_veh02_nuc_mosaic_1box` (PR-F2) — case dispatcher + KK2002 correction.
- `mam_newnuc_1subarea` (PR-F3) — amicphys orchestration.
- Ternary nucleation (no NH₃ in MAM4-MOM; entire branch dead).
- The `adjust_factor_pbl_ratenucl` constant — hard-coded as `1.0` in JAX (matches Fortran's `parameter` default; runtime-configurable via namelist but the box-model namelist doesn't touch it).

## Subtasks

Each ≈ one commit; single PR titled `M3.6 (PR-F1): port nucleation leaf parameterizations`.

1. **Extend `scripts/patches/expose_internals.patch`** with a second hunk that makes `binary_nuc_vehk2002` and `pbl_nuc_wang2008` public from `modal_aero_newnuc` (they're inside the module's `contains` block; need explicit `public ::` declaration for the standalone driver to call them).

2. **Standalone Fortran driver** `scripts/reference_drivers/newnuc_helpers_driver.F90` — sweeps a 3D (T, RH, [H₂SO₄]) grid (16 × 10 × 12 = 1920 records). Output format `1pe27.16e3` (wider than the makoh/kohler `es24.16` to accommodate Vehkamäki's huge dynamic range; rates can hit `exp(-200)` ≈ `1e-87`). Captures binary outputs once and chains both PBL flagaa=11 and flagaa=12 paths after.

3. **Build script** `--newnuc-helpers` flag, mirroring `--makoh`/`--kohler`. `expose_internals.patch` overlay extended to cover newnuc.

4. **`scripts/capture_reference.py --mode newnuc-helpers`** → `tests/reference/newnuc_helpers/reference.npz`. Parser handles four data sections (binary_inputs, binary_outputs, pbl11_outputs, pbl12_outputs) — pbl rows mix one int + 6 floats so the parser splits column-wise.

5. **JAX port** in new module `mam4_jax/newnuc.py`:
   - `binary_nuc_vehk2002(temp, rh, so4vol)` → 5-tuple. Batch-friendly via standard JAX broadcasting.
   - `pbl_nuc_wang2008(so4vol, flagaa, ratenucl, rateloge, cnum_tot, cnum_h2so4, cnum_nh3, radius_cluster)` → 7-tuple (adds `flagaa2` sentinel). `flagaa` is a Python int (static at trace time); the early-return path becomes a `jnp.where` mask.

6. **Tests** in `tests/test_newnuc.py` (3 tests): binary, PBL flagaa=11, PBL flagaa=12. Rel-err < 1e-6 across the full grid.

7. **Residual plot** → `docs/figures/newnuc_helpers_residuals.png`. Two-panel: top binary nucleation rate vs [H₂SO₄] on a few (T, RH) slices (10-order-of-magnitude log-log); bottom per-record |rel-err| for all 7 outputs vs ADR-003 1e-6.

8. **Docs** (rule #5): PROGRESS (PR-F1 entry), PLANS (split 5f into 5f.PR-F1/F2/F3; mark PR-F1 done), SCHEMA (new `newnuc_helpers/` directory), REFERENCE_BUILD (new `newnuc-helpers` capture mode row), FEATURES (new "Nucleation parameterizations" supporting-physics row).

## Verification

- `python -m pytest -q` → 52/52 green (49 + 3 new).
- `python scripts/capture_reference.py --mode newnuc-helpers` regenerates the fixture.
- `python scripts/plot_newnuc_helpers_residuals.py` renders the figure; worst rel-err < 1e-6.

## Out of scope

- Dispatcher and orchestration (PR-F2, PR-F3).
- Ternary nucleation (`ternary_nuc_napari`-equivalent code paths).
- Runtime-configurable `adjust_factor_pbl_ratenucl` (treat as hard-coded `1.0` until a fixture configures it).
