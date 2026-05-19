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
