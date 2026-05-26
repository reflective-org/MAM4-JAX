#!/usr/bin/env bash
# Build the vendored MAM4 Fortran box model.
#
# Detects gfortran and NetCDF (Fortran + C bindings) via nf-config /
# nc-config. Build artifacts land in mam4-original-src-code/{build,run}/,
# which are gitignored — the committed vendored source is never modified.
#
# Usage:
#   scripts/build_reference.sh                       # baseline build
#   scripts/build_reference.sh --instrumented        # adds per-process I/O dump hooks
#   scripts/build_reference.sh --instrumented --no-aitacc-transfer
#                                                    # additionally flips the calcsize
#                                                    # call to do_aitacc_transfer_in=.false.
#   scripts/build_reference.sh --polysvp             # also builds the polysvp reference driver
#   scripts/build_reference.sh --qsat                # also builds the qsat reference driver
#   scripts/build_reference.sh --makoh               # also builds the makoh reference driver
#   scripts/build_reference.sh --kohler              # also builds the kohler reference driver
#
# Instrumented build overlays scripts/patches/mam4_dump_state.F90 and applies
# scripts/patches/driver_instrumentation.patch to the build/ copy of
# driver.F90. The committed vendored tree is never modified in either mode.
#
# --makoh and --kohler additionally apply scripts/patches/expose_internals.patch
# to the build copy of modal_aero_wateruptake.F90, which makes makoh_cubic,
# makoh_quartic, and modal_aero_kohler public so the standalone driver(s)
# can call them.
#
# --polysvp / --qsat / --makoh / --kohler can combine with the baseline
# build to also produce run/<name>_driver.exe, standalone harnesses that
# drive specific entry points (see scripts/reference_drivers/).
#
# Prereqs (macOS):
#   brew install gcc netcdf netcdf-fortran

set -euo pipefail

INSTRUMENTED=0
BUILD_POLYSVP=0
BUILD_QSAT=0
BUILD_MAKOH=0
BUILD_KOHLER=0
BUILD_NEWNUC_HELPERS=0
BUILD_MER07_VEH02=0
BUILD_COAG_COEFFICIENTS=0
NO_AITACC_TRANSFER=0
SKIP_SOAEXCH=0
SKIP_PCARBON_AGING=0
for arg in "$@"; do
  case "$arg" in
    --instrumented)         INSTRUMENTED=1 ;;
    --polysvp)              BUILD_POLYSVP=1 ;;
    --qsat)                 BUILD_QSAT=1 ;;
    --makoh)                BUILD_MAKOH=1 ;;
    --kohler)               BUILD_KOHLER=1 ;;
    --newnuc-helpers)       BUILD_NEWNUC_HELPERS=1 ;;
    --mer07-veh02)          BUILD_MER07_VEH02=1 ;;
    --coag-coefficients)    BUILD_COAG_COEFFICIENTS=1 ;;
    --no-aitacc-transfer)   NO_AITACC_TRANSFER=1 ;;
    --skip-soaexch)         SKIP_SOAEXCH=1 ; SKIP_PCARBON_AGING=1 ;;
    --skip-pcarbon-aging)   SKIP_PCARBON_AGING=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$NO_AITACC_TRANSFER" == "1" && "$INSTRUMENTED" != "1" ]]; then
  echo "Error: --no-aitacc-transfer must be combined with --instrumented" >&2
  exit 2
fi

# --skip-pcarbon-aging modifies amicphys to no-op the pcarbon-aging
# call; the change is independent of the instrumentation overlay.
# M5's sweep-no-pcarbon-aging mode applies it to a baseline (non-
# instrumented) build, so the previous "must be combined with
# --instrumented" constraint has been lifted.
#
# --skip-soaexch applies the gasaerexch_skip_soaexch.patch which
# bypasses the call to mam_soaexch_1subarea. Also independent of
# the instrumentation overlay; lifted for the H2SO4-only 24h sweep
# (sweep-24h-skip-soaexch-no-pcarbon-aging mode in
# scripts/capture_reference.py).

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
  ( cd "$BUILD_DIR" && patch -p1 < "$PATCH_DIR/rename_hook.patch" )
  ( cd "$BUILD_DIR" && patch -p1 < "$PATCH_DIR/amicphys_init_dump.patch" )
  ( cd "$BUILD_DIR" && patch -p1 < "$PATCH_DIR/amicphys_after_writeback.patch" )
fi

if [[ "$NO_AITACC_TRANSFER" == "1" ]]; then
  echo ""
  echo "Applying disable_aitacc_transfer overlay..."
  ( cd "$BUILD_DIR" && patch -p1 < "$PATCH_DIR/disable_aitacc_transfer.patch" )
fi

if [[ "$SKIP_SOAEXCH" == "1" ]]; then
  echo ""
  echo "Applying gasaerexch_skip_soaexch overlay..."
  ( cd "$BUILD_DIR" && patch -p1 < "$PATCH_DIR/gasaerexch_skip_soaexch.patch" )
fi

if [[ "$SKIP_PCARBON_AGING" == "1" ]]; then
  echo ""
  echo "Applying skip_pcarbon_aging overlay..."
  ( cd "$BUILD_DIR" && patch -p1 < "$PATCH_DIR/skip_pcarbon_aging.patch" )
fi

if [[ "$BUILD_MAKOH" == "1" || "$BUILD_KOHLER" == "1" || "$BUILD_NEWNUC_HELPERS" == "1" || "$BUILD_MER07_VEH02" == "1" || "$BUILD_COAG_COEFFICIENTS" == "1" ]]; then
  echo ""
  echo "Applying expose_internals overlay..."
  ( cd "$BUILD_DIR" && patch -p1 < "$PATCH_DIR/expose_internals.patch" )
fi

# --- Build ------------------------------------------------------------------

if [[ "$INSTRUMENTED" == "1" ]]; then
  # mam4_dump_state.o must be built before any consumer of the module.
  # Consumers: driver.o (OBJ9) and modal_aero_amicphys.o (OBJ5, via the
  # rename_hook patch). Compile it into OBJ4 so its .mod is available
  # before OBJ5 (modal_aero_amicphys) and OBJ9 (driver) are built.
  OBJ4_OVERRIDE="rad_constituents.o mam4_dump_state.o"
  ( cd "$BUILD_DIR" && make FCFLAGS="$FCFLAGS" LDFLAGS="$LDFLAGS" \
       OBJ4="$OBJ4_OVERRIDE" )
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

build_ref_driver() {
  local name="$1"
  local exe="$RUN_DIR/${name}_driver.exe"
  local src="$REPO_ROOT/scripts/reference_drivers/${name}_driver.F90"
  # Link against every object the main build produced except the box-model
  # entry points (main, driver) so we get a single program with our own main.
  local objs
  objs=$(ls "$RUN_DIR"/*.o | grep -Ev "/(main|driver|mam4_dump_state)\.o$" | tr '\n' ' ')
  ( cd "$RUN_DIR" && gfortran $FCFLAGS -o "$exe" "$src" $objs $LDFLAGS )
  if [[ ! -x "$exe" ]]; then
    echo "Error: $name driver link failed." >&2
    exit 1
  fi
  echo "Built: $exe"
}

if [[ "$BUILD_POLYSVP" == "1" ]]; then
  build_ref_driver polysvp
fi

if [[ "$BUILD_QSAT" == "1" ]]; then
  build_ref_driver qsat
fi

if [[ "$BUILD_MAKOH" == "1" ]]; then
  build_ref_driver makoh
fi

if [[ "$BUILD_KOHLER" == "1" ]]; then
  build_ref_driver kohler
fi

if [[ "$BUILD_NEWNUC_HELPERS" == "1" ]]; then
  build_ref_driver newnuc_helpers
fi

if [[ "$BUILD_MER07_VEH02" == "1" ]]; then
  build_ref_driver mer07_veh02
fi

if [[ "$BUILD_COAG_COEFFICIENTS" == "1" ]]; then
  build_ref_driver coag_coefficients
fi
