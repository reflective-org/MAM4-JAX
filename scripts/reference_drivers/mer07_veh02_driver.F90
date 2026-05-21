! mer07_veh02_driver.F90 — reference-data harness for
! mer07_veh02_nuc_mosaic_1box (the dispatcher in modal_aero_newnuc).
!
! Sweeps (temp, rh, zm, qh2so4, h2so4_uptkrate) with qnh3=0 (matches
! MAM4-MOM). Covers four regimes:
!   1) Subcutoff: qh2so4_avg < qh2so4_cutoff → early return with zeros.
!   2) Low-rate: above cutoff but rateloge <= -13.82 → early return.
!   3) Active no-PBL: zm > pblh → no PBL nuc.
!   4) Active PBL:    zm < pblh → PBL nuc activates.
!
! Writes a text file ./mer07_veh02_reference.txt that
! scripts/capture_reference.py --mode mer07-veh02 parses into
! tests/reference/mer07_veh02/reference.npz.
!
! Requires the existing expose_internals.patch overlay (which already
! makes mer07_veh02_nuc_mosaic_1box public from modal_aero_newnuc).

      program mer07_veh02_driver

      use shr_kind_mod,      only: r8 => shr_kind_r8
      use modal_aero_newnuc, only: mer07_veh02_nuc_mosaic_1box

      implicit none

      ! Grid sizes (kept modest so the .npz fits in memory comfortably).
      integer, parameter :: n_temp     = 6
      integer, parameter :: n_rh       = 5
      integer, parameter :: n_zm       = 3
      integer, parameter :: n_so4      = 8
      integer, parameter :: n_uptk     = 3
      integer, parameter :: ntot       = n_temp * n_rh * n_zm * n_so4 * n_uptk

      ! Constants matching MAM4-MOM defaults.
      integer, parameter :: newnuc_method_flagaa = 11
      real(r8), parameter :: dtnuc          = 30.0_r8         ! 1 timestep (s)
      real(r8), parameter :: press_in       = 1.0e5_r8        ! Pa
      real(r8), parameter :: pblh_in        = 1000.0_r8       ! m
      real(r8), parameter :: qnh3_cur       = 0.0_r8          ! no NH3
      real(r8), parameter :: mw_so4a_host   = 115.0_r8        ! captured

      integer,  parameter :: nsize          = 1
      integer,  parameter :: maxd_asize     = 1
      ! Aitken mode bounds for MAM4-MOM (rad_constituents.F90:167-170).
      real(r8), parameter :: dplom_sect(1)  = 0.0087e-6_r8
      real(r8), parameter :: dphim_sect(1)  = 0.0520e-6_r8

      ! Grid arrays.
      real(r8) :: temp_g(n_temp), rh_g(n_rh), zm_g(n_zm)
      real(r8) :: so4_g(n_so4), uptk_g(n_uptk)

      ! Output arrays.
      real(r8) :: temp_o(ntot), rh_o(ntot), zm_o(ntot)
      real(r8) :: qh2so4_o(ntot), uptkrate_o(ntot)
      integer  :: isize_nuc_o(ntot)
      real(r8) :: qnuma_del_o(ntot), qso4a_del_o(ntot), qnh4a_del_o(ntot)
      real(r8) :: qh2so4_del_o(ntot), qnh3_del_o(ntot)
      real(r8) :: dens_nh4so4a_o(ntot), dnclusterdt_o(ntot)

      integer  :: it, ir, iz, iq, iu, idx, isize_nuc_loc
      real(r8) :: t_lo, t_hi, rh_lo, rh_hi
      real(r8) :: log_so4_lo, log_so4_hi
      real(r8) :: qnuma_del_loc, qso4a_del_loc, qnh4a_del_loc
      real(r8) :: qh2so4_del_loc, qnh3_del_loc
      real(r8) :: dens_nh4so4a_loc, dnclusterdt_loc

      ! T: 235..295 K (stay within Vehkamäki's valid range).
      t_lo = 235.0_r8 ; t_hi = 295.0_r8
      do it = 1, n_temp
         temp_g(it) = t_lo + (t_hi - t_lo) * real(it-1, r8) / real(n_temp-1, r8)
      end do

      ! RH: 0.05..0.95.
      rh_lo = 0.05_r8 ; rh_hi = 0.95_r8
      do ir = 1, n_rh
         rh_g(ir) = rh_lo + (rh_hi - rh_lo) * real(ir-1, r8) / real(n_rh-1, r8)
      end do

      ! zm: 100, 800, 1500 m (covers below/near/above pblh=1000).
      zm_g(1) = 100.0_r8
      zm_g(2) = 800.0_r8
      zm_g(3) = 1500.0_r8

      ! qh2so4: 1e-17..1e-10 mol/mol (covers subcutoff to highly active).
      log_so4_lo = log(1.0e-17_r8) ; log_so4_hi = log(1.0e-10_r8)
      do iq = 1, n_so4
         so4_g(iq) = exp( log_so4_lo + (log_so4_hi - log_so4_lo) * &
                          real(iq-1, r8) / real(n_so4-1, r8) )
      end do

      ! h2so4_uptkrate: 0, 1e-5, 1e-3 s⁻¹.
      uptk_g(1) = 0.0_r8
      uptk_g(2) = 1.0e-5_r8
      uptk_g(3) = 1.0e-3_r8

      ! Sweep.
      idx = 0
      do it = 1, n_temp
         do ir = 1, n_rh
            do iz = 1, n_zm
               do iq = 1, n_so4
                  do iu = 1, n_uptk
                     idx = idx + 1
                     temp_o(idx)     = temp_g(it)
                     rh_o(idx)       = rh_g(ir)
                     zm_o(idx)       = zm_g(iz)
                     qh2so4_o(idx)   = so4_g(iq)
                     uptkrate_o(idx) = uptk_g(iu)

                     call mer07_veh02_nuc_mosaic_1box( &
                        newnuc_method_flagaa, dtnuc, &
                        temp_g(it), rh_g(ir), press_in, &
                        zm_g(iz), pblh_in, &
                        so4_g(iq), so4_g(iq), qnh3_cur, uptk_g(iu), &
                        mw_so4a_host, &
                        nsize, maxd_asize, dplom_sect, dphim_sect, &
                        isize_nuc_loc, &
                        qnuma_del_loc, qso4a_del_loc, qnh4a_del_loc, &
                        qh2so4_del_loc, qnh3_del_loc, &
                        dens_nh4so4a_loc, &
                        -1, dnclusterdt_loc )

                     isize_nuc_o(idx)    = isize_nuc_loc
                     qnuma_del_o(idx)    = qnuma_del_loc
                     qso4a_del_o(idx)    = qso4a_del_loc
                     qnh4a_del_o(idx)    = qnh4a_del_loc
                     qh2so4_del_o(idx)   = qh2so4_del_loc
                     qnh3_del_o(idx)     = qnh3_del_loc
                     dens_nh4so4a_o(idx) = dens_nh4so4a_loc
                     dnclusterdt_o(idx)  = dnclusterdt_loc
                  end do
               end do
            end do
         end do
      end do

      ! Write output.
      open(unit=10, file='mer07_veh02_reference.txt', status='replace', action='write')
      write(10,'(a)') '# mer07_veh02_nuc_mosaic_1box reference data'
      write(10,'(a)') "# Sections marked with '%' header lines."

      write(10,'(/a,i0)') '% n_total ', ntot

      write(10,'(/a)') '% inputs (rows: ntot; cols: temp rh zm qh2so4 uptkrate)'
      do idx = 1, ntot
         write(10,'(5es27.16e3)') temp_o(idx), rh_o(idx), zm_o(idx), &
                                  qh2so4_o(idx), uptkrate_o(idx)
      end do

      write(10,'(/a)') '% outputs (rows: ntot; cols: isize_nuc qnuma_del qso4a_del qnh4a_del qh2so4_del qnh3_del dens_nh4so4a dnclusterdt)'
      do idx = 1, ntot
         write(10,'(i6,7es27.16e3)') isize_nuc_o(idx), &
                                     qnuma_del_o(idx), qso4a_del_o(idx), &
                                     qnh4a_del_o(idx), qh2so4_del_o(idx), &
                                     qnh3_del_o(idx), dens_nh4so4a_o(idx), &
                                     dnclusterdt_o(idx)
      end do

      close(10)
      write(*,'(a,i0,a)') 'mer07_veh02_driver: wrote ', ntot, ' records to mer07_veh02_reference.txt'

      end program mer07_veh02_driver
