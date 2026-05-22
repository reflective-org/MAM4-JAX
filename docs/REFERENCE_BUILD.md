# Building and capturing the Fortran reference

The vendored Fortran reference under `mam4-original-src-code/` is the ground truth for the JAX port. This doc explains how to build it and how to (re)generate the reference outputs that live under `tests/reference/`.

The committed vendored tree is treated as read-only (ADR-001). Build/run artifacts go into `mam4-original-src-code/{build,run}/`, which are gitignored.

## Prerequisites

| Tool | Minimum | Tested |
| --- | --- | --- |
| `gfortran` | 10+ | 15.2.0 (Homebrew GCC) |
| `netcdf` (C library) | 4.6+ | 4.9.3 |
| `netcdf-fortran` | 4.5+ | 4.6.2 |
| `cpp` | any | (system) |
| Python | 3.10+ | 3.12 |

### macOS

```bash
brew install gcc netcdf netcdf-fortran
```

That installs `gfortran` (from `gcc`), `nc-config`, and `nf-config` on `PATH`.

### Linux

Use your distribution's `gfortran`, `libnetcdf-dev`, and `libnetcdff-dev` (Debian/Ubuntu names). The scripts auto-detect paths via `nf-config` / `nc-config`, so as long as those are on `PATH` no further configuration is needed.

## Quick start

```bash
# Baseline: build + 12-point convergence sweep → tests/reference/sweep/*.nc
python scripts/capture_reference.py

# Instrumented: build with the ADR-012 overlay, capture per-process I/O
# at the three microphysics call sites → tests/reference/per_process/*.npz
python scripts/capture_reference.py --mode instrumented
python scripts/capture_reference.py --mode instrumented --nstep 60   # longer
```

Sweep mode writes 12 NetCDF files (~1.7 MB total) under `tests/reference/sweep/`, one per timestep count `(1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800)` over a fixed 1800 s integration window. Instrumented mode writes six `.npz` archives under `tests/reference/per_process/`, one per (process, before|after) hook point. See `tests/reference/SCHEMA.md` for the exact data contracts.

## Capture modes

`scripts/capture_reference.py --mode <mode>` accepts the following values. All write under `tests/reference/`; the schema for each subdirectory is in `tests/reference/SCHEMA.md`.

| `--mode` | Purpose | Build flavour | Output |
| --- | --- | --- | --- |
| `sweep` (default) | 12-point convergence sweep for end-to-end validation (canonical full-physics namelist, pcarbon aging on). | baseline (no overlay) | `tests/reference/sweep/mam_dt<DT>_ndt<N>.nc` |
| `sweep-no-pcarbon-aging` | 12-point convergence sweep with `skip_pcarbon_aging.patch` applied at build time. Matches the JAX port's M3.6 scope (pcarbon aging deferred). Used by M5's `test_sweep_matches_fortran`. Build-script constraint relaxed (2026-05-22) so `--skip-pcarbon-aging` no longer requires `--instrumented`. | baseline + skip_pcarbon_aging | `tests/reference/sweep_no_pcarbon_aging/mam_dt<DT>_ndt<N>.nc` |
| `instrumented` | Per-process I/O dumps around calcsize / wateruptake / amicphys (full-physics fixture), plus the amicphys-internal rename hook (M3.6 PR-B). | ADR-012 overlay + `scripts/patches/rename_hook.patch` | `tests/reference/per_process/{calcsize,wateruptake,amicphys,rename}_{before,after}.npz` + (one-time) `tests/reference/indices/reference.npz` |
| `instrumented-no-aitacc` | Same hooks as `instrumented` but with `do_aitacc_transfer_in=.false.` so calcsize PR-A could be validated without the Aitken↔accum transfer block confounding it. | overlay + `scripts/patches/disable_aitacc_transfer.patch` | `tests/reference/per_process_no_aitacc/*.npz` |
| `instrumented-amicphys-off` | Same hooks as `instrumented` but writes the namelist with `mdo_gasaerexch=mdo_rename=mdo_newnuc=mdo_coag=0`, forcing `modal_aero_amicphys_intr` into bit-exact passthrough. Used by M3.6 PR-A to validate the orchestration shell in isolation from the four physics sub-routines. | overlay (no extra patch) | `tests/reference/per_process_amicphys_off/*.npz` |
| `instrumented-rename-only` | Same hooks but with `mdo_gasaerexch=mdo_newnuc=mdo_coag=0, mdo_rename=1` — only rename runs in the Fortran amicphys. Used by M3.6 PR-C's `test_orchestration_rename_only_matches_fortran`. In this fixture rename ends up being a no-op every step (the optaa=40 guard trips because Aitken `dgn` stays small without gasaerexch growth) — so the test ends up validating the state-dict ↔ amicphys-local-view round-trip end-to-end. | overlay (no extra patch) | `tests/reference/per_process_rename_only/*.npz` |
| `instrumented-gasaerexch-with-soaexch-only` | `mdo_gasaerexch=1, others=0` plus the `skip_pcarbon_aging.patch` overlay (pcarbon aging is a separate sub-process out of M3.6 scope). SOA exchange runs as in unmodified Fortran. Used by M3.6 PR-E's `test_orchestration_gasaerexch_matches_fortran`. | overlay + skip_pcarbon_aging | `tests/reference/per_process_gasaerexch/*.npz` |
| `instrumented-gasaerexch-and-newnuc-only` | `mdo_gasaerexch=1, mdo_newnuc=1, others=0` plus `skip_pcarbon_aging.patch`. Newnuc requires gasaerexch to produce `qgas_avg[h2so4]`. Used by M3.6 PR-F3's `test_orchestration_gasaerexch_and_newnuc_matches_fortran`. | overlay + skip_pcarbon_aging | `tests/reference/per_process_gasaerexch_and_newnuc/*.npz` |
| `instrumented-coag-only` | `mdo_coag=1, others=0` plus `skip_pcarbon_aging.patch`. Coag operates on the current state's `dgncur_a`/`dgncur_awet`/`wetdens` (set by calcsize + wateruptake upstream of amicphys); unlike newnuc it does not require any other sub-process to fire first. Used by M3.6 PR-G3's `test_orchestration_coag_only_matches_fortran`. | overlay + skip_pcarbon_aging | `tests/reference/per_process_coag/*.npz` |
| `instrumented-full-minus-pcarbon-aging` | All `mdo_*=1` (canonical full-physics namelist) but `skip_pcarbon_aging.patch` applied at build time so the pcarbon-aging sub-process is no-op'd. Matches the JAX port's M3.6 scope (pcarbon aging deferred). Used by M4 PR-A's `test_run_step_one_step_matches_fortran`; PR-M4-B's 60-step trajectory test uses the same fixture. | overlay + skip_pcarbon_aging | `tests/reference/per_process_full_minus_pcarbon_aging/*.npz` |
| `polysvp` | Standalone Goff–Gratch driver — sweeps temperature for both water and ice branches. | `scripts/reference_drivers/polysvp_driver.F90` | `tests/reference/polysvp/reference.npz` |
| `qsat` | Standalone `qsat_water` / `qsat_ice` driver over a (T, p) grid. | `scripts/reference_drivers/qsat_driver.F90` | `tests/reference/qsat/reference.npz` |
| `makoh` | Standalone `makoh_cubic` / `makoh_quartic` driver over hand-picked polynomial test cases. | `scripts/reference_drivers/makoh_driver.F90` | `tests/reference/makoh/reference.npz` |
| `kohler` | Standalone `modal_aero_kohler` driver across a (rdry, hygro, s) grid. | `scripts/reference_drivers/kohler_driver.F90` | `tests/reference/kohler/reference.npz` |
| `newnuc-helpers` | Standalone `binary_nuc_vehk2002` + `pbl_nuc_wang2008` driver across a (T, RH, [H₂SO₄]) grid; both PBL flagaa branches captured. | `scripts/reference_drivers/newnuc_helpers_driver.F90` | `tests/reference/newnuc_helpers/reference.npz` |
| `mer07-veh02` | Standalone `mer07_veh02_nuc_mosaic_1box` dispatcher driver across a 5D (T, RH, zm, [H₂SO₄], H₂SO₄ uptake rate) grid covering 5 regimes (subcutoff / low-rate / active no-PBL / active PBL / gas-limited). | `scripts/reference_drivers/mer07_veh02_driver.F90` | `tests/reference/mer07_veh02/reference.npz` |
| `coag-coefficients` | Standalone `getcoags` + `getcoags_wrapper_f` driver across a (T, P, dgnumA, dgnumB) grid (4×2×5×6 = 240 records) for fixed MAM4-MOM sigmas (1.6 / 1.8) and densities (1770 / 1770). Captures both functions' outputs in the same `.npz` so PR-G2 reuses the PR-G1 fixture. | `scripts/reference_drivers/coag_coefficients_driver.F90` | `tests/reference/coag_coefficients/reference.npz` |

`--nstep` applies to the two `instrumented*` modes. Defaults: `1` for `instrumented`, `60` for `instrumented-no-aitacc`. Values outside the canonical sweep `(1, 2, 4, 9, 18, 30, 60, 120, 180, 360, 900, 1800)` print a warning but still run. The standalone-driver modes (`polysvp`, `qsat`, `makoh`, `kohler`, `newnuc-helpers`, `mer07-veh02`, `coag-coefficients`) have no `--nstep` knob — they sweep their own input grids.

### Standalone reference drivers

The standalone driver modes (`polysvp`, `qsat`, `makoh`, `kohler`, `newnuc-helpers`, `mer07-veh02`, `coag-coefficients`) compile a tiny Fortran main against the relevant upstream module (`box_model_utils/wv_saturation.F90`, `e3sm_src_modified/modal_aero_wateruptake.F90`, `e3sm_src/modal_aero_newnuc.F90`, or `e3sm_src/modal_aero_coag.F90`) and dump tabulated outputs. They do **not** run the box-model driver, so they bypass the namelist, `cambox_config.*`, and the ADR-012 overlay entirely. Each driver source lives in `scripts/reference_drivers/`; `capture_reference.py` builds and runs it directly with `gfortran`. The `makoh`, `kohler`, `newnuc-helpers`, `mer07-veh02`, and `coag-coefficients` modes additionally apply `scripts/patches/expose_internals.patch` to make module-private helpers (`makoh_cubic/quartic`, `modal_aero_kohler`, `binary_nuc_vehk2002`, `pbl_nuc_wang2008`, `getcoags`) public so the standalone drivers can call them.

## What the scripts do

### `scripts/build_reference.sh`

Builds the Fortran executable from the vendored source.

1. Verifies `gfortran`, `nf-config`, `nc-config`, `cpp` are on `PATH`.
2. Sets `NETCDF_INCLUDE` = `$(nf-config --includedir)` and exports library paths for both NetCDF Fortran and NetCDF C (which Homebrew installs under separate prefixes).
3. Wipes and recreates `mam4-original-src-code/{build,run}/`.
4. Copies the vendored sources (`box_model_utils/`, `e3sm_src/`, `e3sm_src_modified/`, `test_drivers/`) and `Makefile` into `build/`.
5. Runs `make` with overridden `FCFLAGS`/`LDFLAGS` (see "Build flag tweaks" below).
6. Verifies `mam4-original-src-code/run/mam_box_test.exe` is produced.

The committed vendored source is **never modified** — `git diff mam4-original-src-code/` should be empty after a build.

### `scripts/capture_reference.py`

Runs the executable across the 12-point sweep.

1. If `mam_box_test.exe` is missing, invokes `build_reference.sh`.
2. For each `(dt, nstep)` in the sweep, writes a `namelist` file matching the Fortran driver's five namelist groups (`&time_input`, `&cntl_input`, `&met_input`, `&chem_input`, `&size_parameters`) into `mam4-original-src-code/run/`.
3. Runs `./mam_box_test.exe` from `run/`.
4. Copies the produced `mam_output.nc` to `tests/reference/sweep/mam_dt<DT>_ndt<N>.nc`.

With `--mode instrumented`:

1. Invokes `build_reference.sh --instrumented` (overlay applied; mam4_dump_state.o built before driver.o).
2. Wipes any prior `mam4_dump_*.bin` from the run directory.
3. Runs the executable once for the requested `--nstep`.
4. Parses the six `mam4_dump_*.bin` files (binary record layout in `scripts/patches/mam4_dump_state.F90`'s header) into `tests/reference/per_process/<tag>.npz`, with arrays `istep`, `q`, `qqcw`, `dgncur_a`, `dgncur_awet`, `qaerwat`, `wetdens`. Schema: `tests/reference/SCHEMA.md`.

## Build flag tweaks

Two adjustments to the upstream Makefile setup are required to build with modern gfortran on Homebrew macOS:

| Flag | Why |
| --- | --- |
| `-fallow-invalid-boz` | `infnan.F90` encodes IEEE Inf / NaN via octal BOZ literals. gfortran 10+ rejects this by default; the flag permits the legacy pattern. |
| Two `-L` paths | The upstream Makefile assumes `libnetcdf` and `libnetcdff` share a prefix. Homebrew installs them under separate prefixes; both `-L` paths are needed at link time. |

Both adjustments are applied via `FCFLAGS=...` and `LDFLAGS=...` overrides on the `make` invocation — the vendored `cambox_config.make.in` is **not** modified.

## Namelist details

The Fortran driver reads five namelist groups; one of them (`&size_parameters`) is **not** included in the upstream `run_test.csh` despite being mandatory in this snapshot (it was added when the repo's CUSTOM_SIZE compile flag was removed). The defaults written by `scripts/capture_reference.py` match the MAM4-MOM defaults from `mam4-original-src-code/box_model_utils/rad_constituents.F90:167-170`:

| Mode | `dgnum` (m) | `sigmag` |
| --- | --- | --- |
| 1 (accumulation) | `0.1100e-6` | `1.800` |
| 2 (Aitken) | `0.0260e-6` | `1.600` |
| 3 (coarse) | `2.000e-6` | `1.800` |
| 4 (primary carbon) | `0.050e-6` | `1.600` |

Process toggles, meteorology, and chemistry initial conditions match `run_test.csh` verbatim.

## Notes on the upstream `run_test.csh`

The script committed under `mam4-original-src-code/run_test.csh` is **not used by these wrappers** for two reasons:

1. Its loop body contains an `exit` before the `mv mam_output.nc ...` line, so only the first sweep iteration ever runs and no outputs are archived.
2. It references an `outpath` hard-coded to a previous developer's home directory.

`scripts/capture_reference.py` reimplements the sweep cleanly without touching the vendored copy. The vendored `run_test.csh` is preserved exactly as it shipped, for provenance.

## Reproducibility

The reference outputs under `tests/reference/sweep/` and `tests/reference/per_process/` are committed to the repo. To regenerate them on a fresh checkout, run `python scripts/capture_reference.py` and `python scripts/capture_reference.py --mode instrumented` and verify `git diff tests/reference/` is empty. Bit-exact reproducibility depends on `gfortran` version and `netcdf-fortran` ABI; small differences are expected across compiler versions. The exact data contracts are in `tests/reference/SCHEMA.md`.
