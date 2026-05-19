! mam4_dump_state.F90 — instrumentation helper for MAM4-JAX reference capture.
!
! Provides dump_snapshot(), called from the patched driver.F90 immediately
! before and after each microphysics process call. Writes one binary record
! per invocation to mam4_dump_<tag>.bin in the run directory, appending across
! timesteps so a sweep produces one file per (process, before|after).
!
! Binary record format (stream-access, native endianness):
!   integer(4) :: istep
!   integer(4) :: ncol, pver, pcnst, ntot_amode
!   real(r8)   :: q(ncol, pver, pcnst)
!   real(r8)   :: qqcw(ncol, pver, pcnst)
!   real(r8)   :: dgncur_a(ncol, pver, ntot_amode)
!   real(r8)   :: dgncur_awet(ncol, pver, ntot_amode)
!   real(r8)   :: qaerwat(ncol, pver, ntot_amode)
!   real(r8)   :: wetdens(ncol, pver, ntot_amode)
!
! Consumed by scripts/capture_reference.py --mode instrumented, which
! reassembles records into per-process .npz archives under
! tests/reference/per_process/. Schema documented in
! tests/reference/SCHEMA.md.
!
! This file is NOT part of the vendored Fortran tree; it is copied into
! the transient build/ directory by scripts/build_reference.sh when called
! with --instrumented.

      module mam4_dump_state

      use shr_kind_mod, only: r8 => shr_kind_r8

      implicit none
      private
      public :: dump_snapshot

      contains

      subroutine dump_snapshot(tag, istep, ncol, pver, pcnst, ntot_amode, &
                               q, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens)

         character(len=*), intent(in) :: tag
         integer,          intent(in) :: istep, ncol, pver, pcnst, ntot_amode
         real(r8),         intent(in) :: q(:,:,:), qqcw(:,:,:)
         real(r8),         intent(in) :: dgncur_a(:,:,:), dgncur_awet(:,:,:)
         real(r8),         intent(in) :: qaerwat(:,:,:), wetdens(:,:,:)

         integer            :: unit
         character(len=256) :: filename
         logical            :: exists

         filename = 'mam4_dump_' // trim(tag) // '.bin'

         inquire(file=filename, exist=exists)
         if (exists) then
            open(newunit=unit, file=filename, form='unformatted', &
                 access='stream', position='append', action='write')
         else
            open(newunit=unit, file=filename, form='unformatted', &
                 access='stream', action='write')
         end if

         write(unit) istep
         write(unit) ncol, pver, pcnst, ntot_amode
         write(unit) q
         write(unit) qqcw
         write(unit) dgncur_a
         write(unit) dgncur_awet
         write(unit) qaerwat
         write(unit) wetdens

         close(unit)

      end subroutine dump_snapshot

      end module mam4_dump_state
