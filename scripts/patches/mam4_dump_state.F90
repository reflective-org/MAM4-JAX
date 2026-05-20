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
      public :: dump_snapshot, dump_indices, dump_rename_snapshot

      contains

      subroutine dump_indices()
         !
         ! Write modal_aero_data's integer index tables to mam4_indices.txt
         ! once at init. Called by the patched driver.F90 before the istep
         ! loop. Output is plain text with '%' section markers; parsed by
         ! scripts/capture_reference.py --mode instrumented into
         ! tests/reference/indices/reference.npz.
         !
         use modal_aero_data, only: ntot_amode, ntot_aspectype, maxd_aspectype, &
                                     numptr_amode, numptrcw_amode, &
                                     lspectype_amode, lmassptr_amode, lmassptrcw_amode, &
                                     nspec_amode, specname_amode, modename_amode

         integer :: unit, i, j

         open(newunit=unit, file='mam4_indices.txt', status='replace', action='write')

         write(unit,'(a)') '# MAM4 index tables, captured by mam4_dump_state::dump_indices.'
         write(unit,'(a)') "# Section markers begin with '%'. 2D arrays are written"
         write(unit,'(a)') '# column-major (Fortran memory order).'

         write(unit,'(/a)')   '% ntot_amode'
         write(unit,'(i0)')   ntot_amode

         write(unit,'(/a)')   '% ntot_aspectype'
         write(unit,'(i0)')   ntot_aspectype

         write(unit,'(/a)')   '% maxd_aspectype'
         write(unit,'(i0)')   maxd_aspectype

         write(unit,'(/a)')   '% numptr_amode (shape: ntot_amode)'
         write(unit,'(*(i0,1x))') (numptr_amode(i), i=1, ntot_amode)

         write(unit,'(/a)')   '% numptrcw_amode (shape: ntot_amode)'
         write(unit,'(*(i0,1x))') (numptrcw_amode(i), i=1, ntot_amode)

         write(unit,'(/a)')   '% nspec_amode (shape: ntot_amode)'
         write(unit,'(*(i0,1x))') (nspec_amode(i), i=1, ntot_amode)

         write(unit,'(/a)')   '% lspectype_amode (shape: maxd_aspectype, ntot_amode)'
         do j = 1, ntot_amode
            write(unit,'(*(i0,1x))') (lspectype_amode(i, j), i=1, maxd_aspectype)
         end do

         write(unit,'(/a)')   '% lmassptr_amode (shape: maxd_aspectype, ntot_amode)'
         do j = 1, ntot_amode
            write(unit,'(*(i0,1x))') (lmassptr_amode(i, j), i=1, maxd_aspectype)
         end do

         write(unit,'(/a)')   '% lmassptrcw_amode (shape: maxd_aspectype, ntot_amode)'
         do j = 1, ntot_amode
            write(unit,'(*(i0,1x))') (lmassptrcw_amode(i, j), i=1, maxd_aspectype)
         end do

         write(unit,'(/a)')   '% modename_amode (shape: ntot_amode, strings)'
         do i = 1, ntot_amode
            write(unit,'(a)') trim(modename_amode(i))
         end do

         write(unit,'(/a)')   '% specname_amode (shape: ntot_aspectype, strings)'
         do i = 1, ntot_aspectype
            write(unit,'(a)') trim(specname_amode(i))
         end do

         close(unit)

      end subroutine dump_indices

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

      subroutine dump_rename_snapshot(tag, istep, i, k, jsub, &
                                      mtoo_renamexf, qnum_cur, qaer_cur, &
                                      qaer_delsub_grow4rnam, qwtr_cur, &
                                      fac_m2v_aer)
         !
         ! Dump the local-view inputs and outputs of mam_rename_1subarea,
         ! captured from inside mam_amicphys_1subarea_clear via the rename
         ! hook patch. One record per call (per col, level, sub-area, step).
         !
         ! fac_m2v_aer is amicphys-private module data populated at
         ! amicphys init; we dump it alongside each record (constant across
         ! the run but having it inline makes the .npz self-contained for
         ! tests).
         !
         ! Binary record format (stream-access, native endianness):
         !   integer(4) :: istep, i, k, jsub
         !   integer(4) :: max_mode, max_aer
         !   integer(4) :: mtoo_renamexf(max_mode)
         !   real(r8)   :: qnum_cur(max_mode)
         !   real(r8)   :: qaer_cur(max_aer, max_mode)
         !   real(r8)   :: qaer_delsub_grow4rnam(max_aer, max_mode)
         !   real(r8)   :: qwtr_cur(max_mode)
         !   real(r8)   :: fac_m2v_aer(max_aer)
         !
         character(len=*), intent(in) :: tag
         integer,          intent(in) :: istep, i, k, jsub
         integer,          intent(in) :: mtoo_renamexf(:)
         real(r8),         intent(in) :: qnum_cur(:), qwtr_cur(:)
         real(r8),         intent(in) :: qaer_cur(:,:), qaer_delsub_grow4rnam(:,:)
         real(r8),         intent(in) :: fac_m2v_aer(:)

         integer            :: unit
         character(len=256) :: filename
         logical            :: exists
         integer            :: max_mode_loc, max_aer_loc

         filename = 'mam4_dump_' // trim(tag) // '.bin'

         inquire(file=filename, exist=exists)
         if (exists) then
            open(newunit=unit, file=filename, form='unformatted', &
                 access='stream', position='append', action='write')
         else
            open(newunit=unit, file=filename, form='unformatted', &
                 access='stream', action='write')
         end if

         max_mode_loc = size(qnum_cur)
         max_aer_loc  = size(qaer_cur, dim=1)

         write(unit) istep, i, k, jsub
         write(unit) max_mode_loc, max_aer_loc
         write(unit) mtoo_renamexf
         write(unit) qnum_cur
         write(unit) qaer_cur
         write(unit) qaer_delsub_grow4rnam
         write(unit) qwtr_cur
         write(unit) fac_m2v_aer

         close(unit)

      end subroutine dump_rename_snapshot

      end module mam4_dump_state
