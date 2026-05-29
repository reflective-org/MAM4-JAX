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
      public :: dump_snapshot, dump_snapshot_vmr, dump_indices, dump_rename_snapshot

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
         use constituents,    only: cnst_get_ind

         integer :: unit, i, j

         ! Gas constituent pcnst slots for cloudchem / amicphys. cnst_get_ind
         ! returns -1 for absent species (the .false. arg suppresses endrun).
         ! Added 2026-05-28 (M8 PR-K1) so JAX can map H2SO4/SO2/NH3/HCL/HNO3/
         ! SOAG into q[gas_pcnst] without re-running cnst_get_ind on the JAX
         ! side. The H2SO4 / SOAG slots are also available via pcnst_lmap_gas
         ! (amicphys gas list); the others (SO2/NH3/HCL/HNO3) are *not* in
         ! lmap_gas — this is the only way to discover them.
         integer :: l_h2so4g, l_so2g, l_nh3g, l_hclg, l_hno3g, l_soag

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

         ! Gas pcnst slots — 1-based as returned by cnst_get_ind. Converted to
         ! 0-based with -1 sentinel for absent species on the Python side.
         ! Species names are case-sensitive — they match the cnst registry
         ! exactly. Configs that register species under different casing
         ! (e.g. 'h2so4') will silently return -1 from these lookups.
         call cnst_get_ind( 'H2SO4', l_h2so4g, .false. )
         call cnst_get_ind( 'SO2',   l_so2g,   .false. )
         call cnst_get_ind( 'NH3',   l_nh3g,   .false. )
         call cnst_get_ind( 'HCL',   l_hclg,   .false. )
         call cnst_get_ind( 'HNO3',  l_hno3g,  .false. )
         call cnst_get_ind( 'SOAG',  l_soag,   .false. )

         write(unit,'(/a)')   '% gas_pcnst_indices (1-based; -1 if species absent)'
         write(unit,'(a,1x,i0)') 'h2so4', l_h2so4g
         write(unit,'(a,1x,i0)') 'so2',   l_so2g
         write(unit,'(a,1x,i0)') 'nh3',   l_nh3g
         write(unit,'(a,1x,i0)') 'hcl',   l_hclg
         write(unit,'(a,1x,i0)') 'hno3',  l_hno3g
         write(unit,'(a,1x,i0)') 'soag',  l_soag

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

      subroutine dump_snapshot_vmr(tag, istep, ncol, pver, gas_pcnst, ntot_amode, &
                                   vmr, vmrcw, dgncur_a, dgncur_awet, qaerwat, wetdens)
         !
         ! Same binary record format as dump_snapshot, but for the
         ! amicphys-internal vmr / vmrcw arrays (volume mixing ratios with
         ! gas_pcnst third dimension, typically 30 for MAM4-MOM vs pcnst=35
         ! for mass-mixing-ratio q / qqcw).
         !
         ! The output .bin layout matches dump_snapshot byte-for-byte; the
         ! distinction lives at the call site (and in the Python parser,
         ! which renames the keys 'q'/'qqcw' -> 'vmr'/'vmrcw' for tags
         ! produced by this routine). Used by cloudchem_hook.patch to
         ! capture the state across cloudchem_simple_sub, which operates
         ! on vmr / vmrcw in driver.F90's amicphys-vmr-context block.
         !
         ! ***Binary format MUST stay in lock-step with dump_snapshot***
         ! Any record-layout change (e.g., adding a new written field)
         ! must be mirrored to both subroutines, or the Python parser
         ! (_read_dump in capture_reference.py, used for both formats)
         ! will mis-parse one or the other. The Python side currently
         ! reads format-blindly from the header dims; if that diverges,
         ! either add a magic byte to differentiate or split _read_dump
         ! into two parsers. See M14 follow-up: when cloudy-subarea
         ! amicphys lands additional vmr-mode dump tags, this is the
         ! moment to also enumerate the vmr-tag prefixes explicitly in
         ! capture_reference.py rather than relying on `startswith
         ! ("cloudchem_")`.
         !
         character(len=*), intent(in) :: tag
         integer,          intent(in) :: istep, ncol, pver, gas_pcnst, ntot_amode
         real(r8),         intent(in) :: vmr(:,:,:), vmrcw(:,:,:)
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
         write(unit) ncol, pver, gas_pcnst, ntot_amode
         write(unit) vmr
         write(unit) vmrcw
         write(unit) dgncur_a
         write(unit) dgncur_awet
         write(unit) qaerwat
         write(unit) wetdens

         close(unit)

      end subroutine dump_snapshot_vmr

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
