! kohler_driver.F90 — reference-data harness for modal_aero_kohler.
!
! Sweeps over a (rdry, hygro, s) grid that exercises all four branches of
! the kohler solver:
!
!   (a) "very small" particle (vol <= 1e-12 microns^3 → r = rdry)
!   (b) small-p approximation (|p31|/rdry^2 < 1e-4)
!   (c) generic quartic solution
!   (d) near-saturation interpolation (s > 1-eps with eps=1e-4)
!
! Writes the input grid plus rwet to ./kohler_reference.txt, which
! scripts/capture_reference.py --mode kohler parses into
! tests/reference/kohler/reference.npz.
!
! Requires the expose_internals patch overlay to make modal_aero_kohler
! callable from outside the module. Not part of the vendored tree —
! lives under scripts/reference_drivers/.

      program kohler_driver

      use shr_kind_mod,            only: r8 => shr_kind_r8
      use modal_aero_wateruptake,  only: modal_aero_kohler

      implicit none

      ! Grid sizes (must fit within makoh's imax=200 internal buffer).
      integer, parameter :: nr = 7    ! dry radii
      integer, parameter :: nh = 4    ! hygroscopicities
      integer, parameter :: ns = 6    ! relative humidities

      integer, parameter :: npts = nr * nh * ns   ! = 168 points

      real(r8) :: rdry_vals(nr)
      real(r8) :: hygro_vals(nh)
      real(r8) :: s_vals(ns)

      real(r8) :: rdry_in(npts), hygro(npts), s_in(npts), rwet(npts)

      integer :: i, j, k, n

      ! Dry radii spanning insoluble (1e-13 m → vol~1e-39) through coarse-
      ! mode (1e-5 m = 10 microns). Logarithmic spacing.
      rdry_vals = (/ 1.0e-13_r8, 1.0e-9_r8, 1.0e-8_r8, 1.0e-7_r8, &
                     1.0e-6_r8,  3.0e-6_r8, 1.0e-5_r8 /)

      ! Hygroscopicity: 0 (insoluble), 0.1, 0.5 (typical), 1.4 (sea salt).
      hygro_vals = (/ 0.0_r8, 0.1_r8, 0.5_r8, 1.4_r8 /)

      ! Relative humidities spanning dry through near-saturation.
      ! 0.9999 lies above the 1-eps threshold so it exercises the
      ! near-saturation interpolation branch.
      s_vals = (/ 0.1_r8, 0.5_r8, 0.9_r8, 0.99_r8, 0.999_r8, 0.9999_r8 /)

      n = 0
      do i = 1, nr
         do j = 1, nh
            do k = 1, ns
               n = n + 1
               rdry_in(n) = rdry_vals(i)
               hygro(n)   = hygro_vals(j)
               s_in(n)    = s_vals(k)
            end do
         end do
      end do

      call modal_aero_kohler(rdry_in, hygro, s_in, rwet, npts)

      open(unit=10, file='kohler_reference.txt', status='replace', action='write')
      write(10,'(a)') '# kohler reference data'
      write(10,'(a,i0)') '# npts: ', npts
      write(10,'(a)') '# columns: rdry_in (m)   hygro (-)   s (-)   rwet (m)'
      write(10,'(a)') '% kohler_grid'
      do n = 1, npts
         write(10,'(4es24.16)') rdry_in(n), hygro(n), s_in(n), rwet(n)
      end do
      close(10)

      write(*,'(a,i0,a)') 'kohler_driver: wrote ', npts, &
         ' rows to kohler_reference.txt'

      end program kohler_driver
