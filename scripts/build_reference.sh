#!/usr/bin/env bash
# Build the vendored MAM4 Fortran box model.
#
# Detects gfortran and NetCDF (Fortran + C bindings) via nf-config /
# nc-config. Build artifacts land in mam4-original-src-code/{build,run}/,
# which are gitignored — the committed vendored source is never modified.
#
# Usage:
#   scripts/build_reference.sh
#
# Prereqs (macOS):
#   brew install gcc netcdf netcdf-fortran

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/mam4-original-src-code"
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

# --- Build ------------------------------------------------------------------

( cd "$BUILD_DIR" && make FCFLAGS="$FCFLAGS" LDFLAGS="$LDFLAGS" )

if [[ ! -x "$EXE" ]]; then
  echo "Error: make exited cleanly but $EXE is missing." >&2
  exit 1
fi

echo ""
echo "Built: $EXE"
