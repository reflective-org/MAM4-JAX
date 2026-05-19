#!/usr/bin/env bash
# Build the vendored MAM4 Fortran box model.
#
# Detects gfortran and NetCDF (Fortran + C bindings) via nf-config /
# nc-config. Build artifacts land in mam4-original-src-code/{build,run}/,
# which are gitignored — the committed vendored source is never modified.
#
# Usage:
#   scripts/build_reference.sh                 # baseline build
#   scripts/build_reference.sh --instrumented  # adds per-process I/O dump hooks
#   scripts/build_reference.sh --polysvp       # also builds the polysvp reference driver
#
# Instrumented build overlays scripts/patches/mam4_dump_state.F90 and applies
# scripts/patches/driver_instrumentation.patch to the build/ copy of
# driver.F90. The committed vendored tree is never modified in either mode.
#
# --polysvp can combine with the baseline build to also produce
# run/polysvp_driver.exe, a standalone harness that calls wv_saturation::polysvp
# over a temperature sweep (see scripts/reference_drivers/polysvp_driver.F90).
#
# Prereqs (macOS):
#   brew install gcc netcdf netcdf-fortran

set -euo pipefail

INSTRUMENTED=0
BUILD_POLYSVP=0
for arg in "$@"; do
  case "$arg" in
    --instrumented) INSTRUMENTED=1 ;;
    --polysvp)      BUILD_POLYSVP=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/mam4-original-src-code"
PATCH_DIR="$REPO_ROOT/scripts/patches"
BUILD_DIR="$SRC_DIR/build"
RUN_DIR="$SRC_DIR/run"
EXE="$RUN_DIR/mam_box_test.exe"

# --- Toolchain detection ----------------------------------------------------

require() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: $tool not found in PATH." >&2
    echo "  On macOS install via: brew install gcc netcdf netcdf-fortran" >&2
    exit 1
  fi
}

require gfortran
require nf-config
require nc-config
require cpp

export NETCDF_INCLUDE="$(nf-config --includedir)"
export NETCDF_LIB="$(nf-config --prefix)/lib"
NETCDF_C_LIB="$(nc-config --prefix)/lib"

# --fallow-invalid-boz is required for gfortran 10+ because infnan.F90
# encodes IEEE Inf/NaN via octal BOZ literals (legacy pattern). Both
# -L paths are needed because Homebrew installs netcdf and netcdf-fortran
# under separate prefixes.
FCFLAGS="-O2 -fno-range-check -ffree-line-length-none -fallow-invalid-boz -I${NETCDF_INCLUDE}"
LDFLAGS="-L${NETCDF_LIB} -L${NETCDF_C_LIB} -lnetcdff -lnetcdf"

echo "gfortran:        $(gfortran --version | head -1)"
echo "NETCDF_INCLUDE:  $NETCDF_INCLUDE"
echo "NETCDF_LIB:      $NETCDF_LIB"
echo "NETCDF_C_LIB:    $NETCDF_C_LIB"
echo ""

# --- Fresh build/run directories --------------------------------------------

rm -rf "$BUILD_DIR" "$RUN_DIR"
mkdir -p "$BUILD_DIR" "$RUN_DIR"

cp "$SRC_DIR/box_model_utils/"* "$BUILD_DIR/"
cp "$SRC_DIR/e3sm_src/"* "$BUILD_DIR/"
cp "$SRC_DIR/e3sm_src_modified/"* "$BUILD_DIR/"
cp "$SRC_DIR/test_drivers/"* "$BUILD_DIR/"
cp "$SRC_DIR/Makefile" "$BUILD_DIR/"

# --- Apply instrumentation overlay (if requested) ---------------------------

if [[ "$INSTRUMENTED" == "1" ]]; then
  echo ""
  echo "Applying instrumentation overlay..."
  cp "$PATCH_DIR/mam4_dump_state.F90" "$BUILD_DIR/"
  ( cd "$BUILD_DIR" && patch -p1 < "$PATCH_DIR/driver_instrumentation.patch" )
fi

# --- Build ------------------------------------------------------------------

if [[ "$INSTRUMENTED" == "1" ]]; then
  # mam4_dump_state.o must be built before driver.o so its .mod file is
  # available when driver.F90 references the module.
  OBJ9_OVERRIDE="mam4_dump_state.o gaschem_simple.o cloudchem_simple.o driver.o main.o"
  ( cd "$BUILD_DIR" && make FCFLAGS="$FCFLAGS" LDFLAGS="$LDFLAGS" OBJ9="$OBJ9_OVERRIDE" )
else
  ( cd "$BUILD_DIR" && make FCFLAGS="$FCFLAGS" LDFLAGS="$LDFLAGS" )
fi

if [[ ! -x "$EXE" ]]; then
  echo "Error: make exited cleanly but $EXE is missing." >&2
  exit 1
fi

echo ""
echo "Built: $EXE"

# --- Optional: polysvp standalone driver ------------------------------------

if [[ "$BUILD_POLYSVP" == "1" ]]; then
  POLYSVP_EXE="$RUN_DIR/polysvp_driver.exe"
  POLYSVP_SRC="$REPO_ROOT/scripts/reference_drivers/polysvp_driver.F90"
  # Link against every object the main build produced except the box-model
  # entry points (main, driver) so we get a single program with our own main.
  POLYSVP_OBJS=$(ls "$RUN_DIR"/*.o | grep -Ev "/(main|driver|mam4_dump_state)\.o$" | tr '\n' ' ')
  ( cd "$RUN_DIR" && gfortran $FCFLAGS -o "$POLYSVP_EXE" "$POLYSVP_SRC" $POLYSVP_OBJS $LDFLAGS )
  if [[ ! -x "$POLYSVP_EXE" ]]; then
    echo "Error: polysvp driver link failed." >&2
    exit 1
  fi
  echo "Built: $POLYSVP_EXE"
fi
