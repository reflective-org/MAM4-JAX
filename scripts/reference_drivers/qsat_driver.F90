! qsat_driver.F90 — reference-data harness for wv_saturation::qsat_water_fn / qsat_ice.
!
! Sweeps over a (T, p) grid and writes a four-column text table
! (T, p, qs_water, qs_ice) to ./qsat_reference.txt, which
! scripts/capture_reference.py --mode qsat then converts to .npz.
!
! qsat_water_fn uses Goff-Gratch (matches polysvp_water).
! qsat_ice uses a Clausius-Clapeyron-style approximation, NOT polysvp_ice —
! the JAX port preserves that for fidelity (see mam4_jax/saturation.py).
!
! The functions depend on module-level state in wv_saturation (epsqs, hlatv,
! hlatf, rgasv) which is populated by gestbl(). We call gestbl with the
! canonical values used by the box model (transcribed from shr_const_mod.F90).
!
! Not part of the vendored Fortran tree — lives under scripts/reference_drivers/.

      program qsat_driver

      use shr_kind_mod,  only: r8 => shr_kind_r8
      use wv_saturation, only: qsat_water, qsat_ice, gestbl

      implicit none

      ! Sweep: T over 170 K – 320 K in 0.5 K steps (301 points)
      !        p over 5 representative atmospheric pressures (Pa)
      real(r8), parameter :: t_min = 170.0_r8
      real(r8), parameter :: t_max = 320.0_r8
      real(r8), parameter :: dt    = 0.5_r8
      real(r8), parameter :: p_vals(5) = (/ &
         1.0e3_r8, 1.0e4_r8, 5.0e4_r8, 1.0e5_r8, 1.1e5_r8 /)

      ! Canonical box-model constants (from shr_const_mod.F90).
      real(r8), parameter :: tmn    = 173.16_r8   ! gestbl tmin (table lower bound, K)
      real(r8), parameter :: tmx    = 375.16_r8   ! gestbl tmax (table upper bound, K)
      real(r8), parameter :: trice  = 20.0_r8     ! ice/water transition width
      real(r8), parameter :: epsil  = 18.016_r8 / 28.966_r8  ! mwwv/mwdair
      real(r8), parameter :: latvap = 2.501e6_r8
      real(r8), parameter :: latice = 3.337e5_r8
      real(r8), parameter :: rh2o   = 6.02214e26_r8 * 1.38065e-23_r8 / 18.016_r8
      real(r8), parameter :: cp_in  = 1.00464e3_r8 ! shr_const_cpdair
      real(r8), parameter :: tmlt   = 273.15_r8

      logical  :: ip
      integer  :: i, j, n_t, n_p
      real(r8) :: t, p, qsi
      real(r8) :: t_arr(1), p_arr(1), es_arr(1), qs_arr(1)

      ip = .true.

      ! Populate wv_saturation module-level state.
      call gestbl(tmn, tmx, trice, ip, epsil, latvap, latice, rh2o, cp_in, tmlt)

      n_t = nint((t_max - t_min) / dt) + 1
      n_p = size(p_vals)

      open(unit=10, file='qsat_reference.txt', status='replace', action='write')
      write(10,'(a)')    '# qsat reference table'
      write(10,'(a)')    '# columns: T (K), p (Pa), qs_water (kg/kg), qs_ice (kg/kg)'
      write(10,'(a,i0,a,i0)') '# n_T x n_p: ', n_t, ' x ', n_p

      do i = 0, n_t - 1
         t = t_min + real(i, r8) * dt
         do j = 1, n_p
            p   = p_vals(j)
            t_arr(1) = t
            p_arr(1) = p
            ! qsat_water is a public subroutine that operates on size-1 arrays.
            call qsat_water(t_arr, p_arr, es_arr, qs_arr)
            qsi = qsat_ice(t, p)
            write(10,'(4es24.16)') t, p, qs_arr(1), qsi
         end do
      end do
      close(10)

      write(*,'(a,i0,a,i0,a)') 'qsat_driver: wrote ', n_t, ' x ', n_p, &
         ' rows to qsat_reference.txt'

      end program qsat_driver
