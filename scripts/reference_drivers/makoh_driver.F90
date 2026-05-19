! makoh_driver.F90 — reference-data harness for makoh_cubic / makoh_quartic.
!
! Calls modal_aero_wateruptake's private polynomial root finders on a small
! batch of representative inputs. Writes complex roots (real, imag) to
! ./makoh_reference.txt; scripts/capture_reference.py --mode makoh parses
! it into tests/reference/makoh/reference.npz.
!
! Requires the expose_makoh patch overlay (scripts/patches/expose_makoh.patch)
! to make makoh_cubic and makoh_quartic accessible from outside the module.
!
! Not part of the vendored Fortran tree — lives under scripts/reference_drivers/.

      program makoh_driver

      use shr_kind_mod,            only: r8 => shr_kind_r8
      use modal_aero_wateruptake,  only: makoh_cubic, makoh_quartic

      implicit none

      integer, parameter :: imx  = 200          ! Matches the makoh imx parameter
      integer, parameter :: ncub = 6            ! Number of cubic test cases
      integer, parameter :: nqua = 6            ! Number of quartic test cases

      ! Polynomial coefficient buffers (sized to imx so we can call the
      ! existing makoh routines, which expect statically-sized arrays).
      real(r8) :: cub_p0(imx), cub_p1(imx), cub_p2(imx)
      real(r8) :: qua_p0(imx), qua_p1(imx), qua_p2(imx), qua_p3(imx)
      complex(r8) :: cx3(3, imx), cx4(4, imx)

      integer :: i

      ! --------------------------------------------------------------------
      ! Cubic test cases.
      ! makoh_cubic's algorithm ignores p2 entirely (matches Cardano on a
      ! depressed cubic). Coefficients chosen to exercise:
      !   (1,2) general path with O(1) coefficients
      !   (3)   the "insoluble" branch (p1 == 0)
      !   (4)   Köhler-like magnitudes
      !   (5)   negative p0 / positive p1
      !   (6)   very small p1, large p0
      ! --------------------------------------------------------------------
      cub_p0(:) = 0.0_r8
      cub_p1(:) = 0.0_r8
      cub_p2(:) = 0.0_r8

      cub_p0(1) = -1.0_r8       ; cub_p1(1) = 1.0_r8       ; cub_p2(1) = 0.5_r8
      cub_p0(2) = 2.0_r8        ; cub_p1(2) = -5.0_r8      ; cub_p2(2) = 3.0_r8
      cub_p0(3) = -8.0_r8       ; cub_p1(3) = 0.0_r8       ; cub_p2(3) = 0.0_r8
      cub_p0(4) = -1.234e-12_r8 ; cub_p1(4) = 5.6e-8_r8    ; cub_p2(4) = 0.0_r8
      cub_p0(5) = -2.0_r8       ; cub_p1(5) = 3.0_r8       ; cub_p2(5) = -1.5_r8
      cub_p0(6) = 6.0_r8        ; cub_p1(6) = -11.0_r8     ; cub_p2(6) = 0.0_r8 ! roots 1,2,3 (depressed-cubic form)

      call makoh_cubic(cx3, cub_p2, cub_p1, cub_p0, ncub)

      ! --------------------------------------------------------------------
      ! Quartic test cases.
      ! --------------------------------------------------------------------
      qua_p0(:) = 0.0_r8
      qua_p1(:) = 0.0_r8
      qua_p2(:) = 0.0_r8
      qua_p3(:) = 0.0_r8

      qua_p0(1) =  1.0_r8 ; qua_p1(1) = -2.0_r8 ; qua_p2(1) =  1.0_r8 ; qua_p3(1) =  0.5_r8
      qua_p0(2) = 24.0_r8 ; qua_p1(2) =-50.0_r8 ; qua_p2(2) = 35.0_r8 ; qua_p3(2) =-10.0_r8 ! roots near 1,2,3,4
      qua_p0(3) =  0.0_r8 ; qua_p1(3) =  1.0_r8 ; qua_p2(3) =  0.5_r8 ; qua_p3(3) =  0.0_r8
      qua_p0(4) = -1.0e-6_r8; qua_p1(4) = 1.0e-3_r8 ; qua_p2(4) = 0.0_r8 ; qua_p3(4) = 0.0_r8
      qua_p0(5) =  1.0_r8 ; qua_p1(5) =  0.0_r8 ; qua_p2(5) =  0.0_r8 ; qua_p3(5) =  0.0_r8 ! x^4 + 1 = 0 (4 complex roots)
      qua_p0(6) = -2.0_r8 ; qua_p1(6) =  1.0_r8 ; qua_p2(6) = -3.0_r8 ; qua_p3(6) =  2.0_r8

      call makoh_quartic(cx4, qua_p3, qua_p2, qua_p1, qua_p0, nqua)

      ! --------------------------------------------------------------------
      ! Write output: one section per family.
      ! --------------------------------------------------------------------
      open(unit=10, file='makoh_reference.txt', status='replace', action='write')

      write(10,'(a)') '# makoh reference data'
      write(10,'(a)') '# Sections marked with ''%'' header lines.'

      write(10,'(/a,i0)') '% cubic_inputs  (rows: ncub; cols: p0 p1 p2). ncub=', ncub
      do i = 1, ncub
         write(10,'(3es24.16)') cub_p0(i), cub_p1(i), cub_p2(i)
      end do

      write(10,'(/a)') '% cubic_roots  (rows: ncub*3; cols: real imag; root-major)'
      do i = 1, ncub
         write(10,'(2es24.16)') real(cx3(1, i), r8), aimag(cx3(1, i))
         write(10,'(2es24.16)') real(cx3(2, i), r8), aimag(cx3(2, i))
         write(10,'(2es24.16)') real(cx3(3, i), r8), aimag(cx3(3, i))
      end do

      write(10,'(/a,i0)') '% quartic_inputs  (rows: nqua; cols: p0 p1 p2 p3). nqua=', nqua
      do i = 1, nqua
         write(10,'(4es24.16)') qua_p0(i), qua_p1(i), qua_p2(i), qua_p3(i)
      end do

      write(10,'(/a)') '% quartic_roots  (rows: nqua*4; cols: real imag; root-major)'
      do i = 1, nqua
         write(10,'(2es24.16)') real(cx4(1, i), r8), aimag(cx4(1, i))
         write(10,'(2es24.16)') real(cx4(2, i), r8), aimag(cx4(2, i))
         write(10,'(2es24.16)') real(cx4(3, i), r8), aimag(cx4(3, i))
         write(10,'(2es24.16)') real(cx4(4, i), r8), aimag(cx4(4, i))
      end do

      close(10)

      write(*,'(a)') 'makoh_driver: wrote makoh_reference.txt'

      end program makoh_driver
