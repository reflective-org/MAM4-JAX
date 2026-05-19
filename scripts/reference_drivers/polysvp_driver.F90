! polysvp_driver.F90 — standalone reference-data harness for wv_saturation::polysvp.
!
! Compiles against the same wv_saturation object file used by the box model,
! so the reference data this produces are byte-equivalent to what the box
! model would compute for the same temperatures. Writes a plain-text table
! (T, esat_water, esat_ice) to ./polysvp_reference.txt, which
! scripts/capture_reference.py --mode polysvp then converts to .npz.
!
! Sweep range covers the full atmospheric column of interest for MAM4
! (180 K to 320 K) plus a margin at the cold end where the Goff-Gratch
! parameterization is documented as "uncertain below -70 C".
!
! Not part of the vendored Fortran tree — lives under scripts/reference_drivers/.

      program polysvp_driver

      use shr_kind_mod,   only: r8 => shr_kind_r8
      use wv_saturation,  only: polysvp

      implicit none

      ! Sweep: 170 K to 320 K in 0.1 K steps (1501 points)
      real(r8), parameter :: t_min = 170.0_r8
      real(r8), parameter :: t_max = 320.0_r8
      real(r8), parameter :: dt    = 0.1_r8

      integer  :: i, npts
      real(r8) :: t, esat_water, esat_ice

      npts = nint((t_max - t_min) / dt) + 1

      open(unit=10, file='polysvp_reference.txt', status='replace', action='write')
      write(10,'(a)')   '# polysvp reference table'
      write(10,'(a)')   '# columns: T (K), esat_water (Pa), esat_ice (Pa)'
      write(10,'(a,i0)') '# n_points: ', npts
      write(10,'(a,3es24.16)') '# t_min, t_max, dt: ', t_min, t_max, dt
      do i = 0, npts - 1
         t = t_min + real(i, r8) * dt
         esat_water = polysvp(t, 0)
         esat_ice   = polysvp(t, 1)
         write(10,'(3es24.16)') t, esat_water, esat_ice
      end do
      close(10)

      write(*,'(a,i0,a)') 'polysvp_driver: wrote ', npts, &
         ' rows to polysvp_reference.txt'

      end program polysvp_driver
