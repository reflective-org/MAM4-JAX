! newnuc_helpers_driver.F90 — reference-data harness for the leaf-level
! nucleation parameterizations in modal_aero_newnuc:
!   * binary_nuc_vehk2002 (Vehkamäki 2002 H2SO4-H2O binary nucleation)
!   * pbl_nuc_wang2008    (Wang 2008 boundary-layer first/second order)
!
! Writes a text file ./newnuc_helpers_reference.txt that
! scripts/capture_reference.py --mode newnuc-helpers parses into
! tests/reference/newnuc_helpers/reference.npz.
!
! Requires the extended expose_internals.patch overlay to make
! binary_nuc_vehk2002 and pbl_nuc_wang2008 PUBLIC from modal_aero_newnuc
! (they're inside the module's `contains` block by default).

      program newnuc_helpers_driver

      use shr_kind_mod,      only: r8 => shr_kind_r8
      use modal_aero_newnuc, only: binary_nuc_vehk2002, pbl_nuc_wang2008

      implicit none

      ! Grid sizes — kept modest so the .npz fits in memory comfortably
      ! but large enough to exercise non-trivial coverage of the
      ! parameterizations.
      integer, parameter :: n_temp   = 16   ! T sweep
      integer, parameter :: n_rh     = 10   ! RH sweep
      integer, parameter :: n_so4    = 12   ! [H2SO4] sweep (log-spaced)
      integer, parameter :: ntot     = n_temp * n_rh * n_so4

      real(r8) :: temp_grid(n_temp), rh_grid(n_rh), so4_grid(n_so4)

      real(r8) :: temp_out(ntot), rh_out(ntot), so4_out(ntot)
      real(r8) :: ratenucl(ntot), rateloge(ntot)
      real(r8) :: cnum_h2so4(ntot), cnum_tot(ntot), radius_cluster(ntot)

      real(r8) :: pbl_ratenucl11(ntot), pbl_rateloge11(ntot)
      real(r8) :: pbl_radius11(ntot), pbl_cnum_h2so411(ntot)
      real(r8) :: pbl_cnum_tot11(ntot),  pbl_cnum_nh311(ntot)

      real(r8) :: pbl_ratenucl12(ntot), pbl_rateloge12(ntot)
      real(r8) :: pbl_radius12(ntot), pbl_cnum_h2so412(ntot)
      real(r8) :: pbl_cnum_tot12(ntot),  pbl_cnum_nh312(ntot)

      integer :: flagaa2_11(ntot), flagaa2_12(ntot)

      integer :: it, ir, is, idx
      real(r8) :: t_lo, t_hi, rh_lo, rh_hi, so4_lo, so4_hi
      real(r8) :: log_so4_lo, log_so4_hi

      integer :: flagaa2_local
      real(r8) :: rate_l, log_l, ch_l, ct_l, cn_l, rad_l

      ! T sweep: 230..300 K
      t_lo = 230.0_r8 ; t_hi = 300.0_r8
      do it = 1, n_temp
         temp_grid(it) = t_lo + (t_hi - t_lo) * real(it-1, r8) / real(n_temp-1, r8)
      end do

      ! RH sweep: 0.05..0.95
      rh_lo = 0.05_r8 ; rh_hi = 0.95_r8
      do ir = 1, n_rh
         rh_grid(ir) = rh_lo + (rh_hi - rh_lo) * real(ir-1, r8) / real(n_rh-1, r8)
      end do

      ! so4vol sweep: 1e4..1e10 molec/cm3 (log-spaced)
      log_so4_lo = log(1.0e4_r8) ; log_so4_hi = log(1.0e10_r8)
      do is = 1, n_so4
         so4_grid(is) = exp( log_so4_lo + (log_so4_hi - log_so4_lo) * &
                             real(is-1, r8) / real(n_so4-1, r8) )
      end do

      ! Flatten the 3D grid into 1D arrays, ready for output.
      idx = 0
      do it = 1, n_temp
         do ir = 1, n_rh
            do is = 1, n_so4
               idx = idx + 1
               temp_out(idx) = temp_grid(it)
               rh_out(idx)   = rh_grid(ir)
               so4_out(idx)  = so4_grid(is)

               call binary_nuc_vehk2002(temp_out(idx), rh_out(idx), so4_out(idx), &
                                        ratenucl(idx), rateloge(idx), &
                                        cnum_h2so4(idx), cnum_tot(idx), &
                                        radius_cluster(idx))

               ! flagaa = 11 (first-order PBL): seed with the binary result;
               ! the helper writes through if the PBL rate is higher.
               flagaa2_local = -1
               rate_l = ratenucl(idx) ; log_l = rateloge(idx)
               ch_l   = cnum_h2so4(idx) ; ct_l = cnum_tot(idx)
               cn_l   = 0.0_r8 ; rad_l = radius_cluster(idx)
               call pbl_nuc_wang2008(so4_out(idx), 11, flagaa2_local, &
                                     rate_l, log_l, ct_l, ch_l, cn_l, rad_l)
               flagaa2_11(idx)       = flagaa2_local
               pbl_ratenucl11(idx)   = rate_l
               pbl_rateloge11(idx)   = log_l
               pbl_cnum_h2so411(idx) = ch_l
               pbl_cnum_tot11(idx)   = ct_l
               pbl_cnum_nh311(idx)   = cn_l
               pbl_radius11(idx)     = rad_l

               ! flagaa = 12 (second-order PBL): same seed.
               flagaa2_local = -1
               rate_l = ratenucl(idx) ; log_l = rateloge(idx)
               ch_l   = cnum_h2so4(idx) ; ct_l = cnum_tot(idx)
               cn_l   = 0.0_r8 ; rad_l = radius_cluster(idx)
               call pbl_nuc_wang2008(so4_out(idx), 12, flagaa2_local, &
                                     rate_l, log_l, ct_l, ch_l, cn_l, rad_l)
               flagaa2_12(idx)       = flagaa2_local
               pbl_ratenucl12(idx)   = rate_l
               pbl_rateloge12(idx)   = log_l
               pbl_cnum_h2so412(idx) = ch_l
               pbl_cnum_tot12(idx)   = ct_l
               pbl_cnum_nh312(idx)   = cn_l
               pbl_radius12(idx)     = rad_l
            end do
         end do
      end do

      ! ----- Write output ---------------------------------------------------
      open(unit=10, file='newnuc_helpers_reference.txt', status='replace', action='write')

      write(10,'(a)') '# newnuc helpers reference data'
      write(10,'(a)') "# Sections marked with '%' header lines."

      write(10,'(/a,i0)') '% n_total ', ntot

      write(10,'(/a)') '% binary_inputs (rows: ntot; cols: temp rh so4vol)'
      do idx = 1, ntot
         write(10,'(3es27.16e3)') temp_out(idx), rh_out(idx), so4_out(idx)
      end do

      write(10,'(/a)') '% binary_outputs (rows: ntot; cols: ratenucl rateloge cnum_h2so4 cnum_tot radius_cluster)'
      do idx = 1, ntot
         write(10,'(5es27.16e3)') ratenucl(idx), rateloge(idx), &
                                cnum_h2so4(idx), cnum_tot(idx), &
                                radius_cluster(idx)
      end do

      write(10,'(/a)') '% pbl11_outputs (rows: ntot; cols: flagaa2 ratenucl rateloge cnum_h2so4 cnum_tot cnum_nh3 radius_cluster)'
      do idx = 1, ntot
         write(10,'(i6,6es27.16e3)') flagaa2_11(idx), pbl_ratenucl11(idx), pbl_rateloge11(idx), &
                                   pbl_cnum_h2so411(idx), pbl_cnum_tot11(idx), &
                                   pbl_cnum_nh311(idx), pbl_radius11(idx)
      end do

      write(10,'(/a)') '% pbl12_outputs (rows: ntot; cols: flagaa2 ratenucl rateloge cnum_h2so4 cnum_tot cnum_nh3 radius_cluster)'
      do idx = 1, ntot
         write(10,'(i6,6es27.16e3)') flagaa2_12(idx), pbl_ratenucl12(idx), pbl_rateloge12(idx), &
                                   pbl_cnum_h2so412(idx), pbl_cnum_tot12(idx), &
                                   pbl_cnum_nh312(idx), pbl_radius12(idx)
      end do

      close(10)

      write(*,'(a,i0,a)') 'newnuc_helpers_driver: wrote ', ntot, ' records to newnuc_helpers_reference.txt'

      end program newnuc_helpers_driver
