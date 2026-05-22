! coag_coefficients_driver.F90 — reference-data harness for the leaf
! coagulation-coefficient subroutines in modal_aero_coag:
!   * getcoags    — closed-form Whitby coagulation coefficients
!                   (M3.6 PR-G1, this driver's primary target)
!   * getcoags_wrapper_f — the wrapper that preps inputs and calls
!                   getcoags (M3.6 PR-G2's target; captured here so
!                   the same fixture serves both PRs)
!
! Sweeps (T, P, dgnumA, dgnumB) for fixed (sigmag, pdens) values
! matching MAM4-MOM defaults. For each grid point:
!   1) Compute the intermediates (lamda, knc, kfmat, kfmac, kfmatac)
!      using the prep code from getcoags_wrapper_f.
!   2) Call getcoags with those intermediates → 8 outputs.
!   3) Call getcoags_wrapper_f with the same physical inputs → 8
!      post-processed coefficients (for PR-G2 validation).
!
! Writes a text file ./coag_coefficients_reference.txt that
! scripts/capture_reference.py --mode coag-coefficients parses into
! tests/reference/coag_coefficients/reference.npz.

      program coag_coefficients_driver

      use shr_kind_mod,     only: r8 => shr_kind_r8
      use physconst,        only: pstd, tmelt, boltz
      use modal_aero_coag,  only: getcoags, getcoags_wrapper_f

      implicit none

      ! Grid sizes (kept modest; the full sweep below is 240 records).
      integer, parameter :: n_temp     = 4
      integer, parameter :: n_press    = 2
      integer, parameter :: n_dgnumA   = 5   ! Aitken-mode-like diameters
      integer, parameter :: n_dgnumB   = 6   ! accum/coarse-like diameters
      integer, parameter :: ntot       = n_temp * n_press * n_dgnumA * n_dgnumB

      ! Fixed MAM4 mode constants (sigmag and dry-density defaults).
      real(r8), parameter :: sg_atk    = 1.6_r8         ! Aitken sigmag
      real(r8), parameter :: sg_acc    = 1.8_r8         ! accum sigmag
      real(r8), parameter :: pdens_atk = 1770.0_r8      ! kg/m³
      real(r8), parameter :: pdens_acc = 1770.0_r8
      real(r8), parameter :: two3      = 2.0_r8 / 3.0_r8

      real(r8) :: temp_g(n_temp), press_g(n_press)
      real(r8) :: dgnumA_g(n_dgnumA), dgnumB_g(n_dgnumB)

      ! Output arrays.
      real(r8) :: temp_o(ntot),    press_o(ntot)
      real(r8) :: dgnumA_o(ntot),  dgnumB_o(ntot)
      ! Intermediates (getcoags inputs).
      real(r8) :: lamda_o(ntot),   knc_o(ntot)
      real(r8) :: kfmat_o(ntot),   kfmac_o(ntot), kfmatac_o(ntot)
      ! getcoags outputs.
      real(r8) :: qs11_o(ntot), qn11_o(ntot), qs22_o(ntot), qn22_o(ntot)
      real(r8) :: qs12_o(ntot), qs21_o(ntot), qn12_o(ntot), qv12_o(ntot)
      ! getcoags_wrapper_f outputs (PR-G2 validation).
      real(r8) :: bij0_o(ntot), bij2i_o(ntot), bij2j_o(ntot), bij3_o(ntot)
      real(r8) :: bii0_o(ntot), bii2_o(ntot),  bjj0_o(ntot),  bjj2_o(ntot)

      integer  :: it, ip, ia, ib, idx
      real(r8) :: T, P, dgnA, dgnB
      real(r8) :: t0_val, sqrt_t, amu_val
      real(r8) :: xxlsgA, xxlsgB
      real(r8) :: qs11, qn11, qs22, qn22, qs12, qs21, qn12, qv12
      real(r8) :: bij0, bij2i, bij2j, bij3, bii0, bii2, bjj0, bjj2

      ! T sweep: 240, 260, 280, 300 K.
      temp_g(1) = 240.0_r8 ; temp_g(2) = 260.0_r8
      temp_g(3) = 280.0_r8 ; temp_g(4) = 300.0_r8

      ! P sweep: 5e4, 1e5 Pa (stratosphere-ish + sea level).
      press_g(1) = 5.0e4_r8 ; press_g(2) = 1.0e5_r8

      ! dgnumA (Aitken-like): 1e-8..1e-7 m, log-spaced 5 points.
      dgnumA_g(1) = 1.0e-8_r8
      dgnumA_g(2) = 1.778e-8_r8
      dgnumA_g(3) = 3.162e-8_r8
      dgnumA_g(4) = 5.623e-8_r8
      dgnumA_g(5) = 1.0e-7_r8

      ! dgnumB (accum/coarse-like): 5e-8..5e-6 m, log-spaced 6 points.
      dgnumB_g(1) = 5.0e-8_r8
      dgnumB_g(2) = 1.581e-7_r8
      dgnumB_g(3) = 5.0e-7_r8
      dgnumB_g(4) = 1.581e-6_r8
      dgnumB_g(5) = 3.0e-6_r8
      dgnumB_g(6) = 5.0e-6_r8

      xxlsgA = log(sg_atk)
      xxlsgB = log(sg_acc)

      idx = 0
      do it = 1, n_temp
         T = temp_g(it)
         t0_val  = tmelt + 15.0_r8
         sqrt_t  = sqrt(T)
         amu_val = 1.458e-6_r8 * T * sqrt_t / (T + 110.4_r8)
         do ip = 1, n_press
            P = press_g(ip)
            ! Intermediates from getcoags_wrapper_f's prep code.
            ! lamda = 6.6328e-8 * pstd * T / (t0 * P)
            ! knc = (2/3) * boltz * T / amu
            ! kfmat = sqrt(3*boltz*T / pdensat)
            ! kfmac = sqrt(3*boltz*T / pdensac)
            ! kfmatac = sqrt(6*boltz*T / (pdensat + pdensac))
            do ia = 1, n_dgnumA
               dgnA = dgnumA_g(ia)
               do ib = 1, n_dgnumB
                  dgnB = dgnumB_g(ib)
                  idx = idx + 1

                  temp_o(idx)   = T
                  press_o(idx)  = P
                  dgnumA_o(idx) = dgnA
                  dgnumB_o(idx) = dgnB

                  lamda_o(idx)   = 6.6328e-8_r8 * pstd * T / (t0_val * P)
                  knc_o(idx)     = two3 * boltz * T / amu_val
                  kfmat_o(idx)   = sqrt(3.0_r8 * boltz * T / pdens_atk)
                  kfmac_o(idx)   = sqrt(3.0_r8 * boltz * T / pdens_acc)
                  kfmatac_o(idx) = sqrt(6.0_r8 * boltz * T / (pdens_atk + pdens_acc))

                  ! Direct getcoags call (PR-G1 target).
                  call getcoags( lamda_o(idx), kfmatac_o(idx), &
                                 kfmat_o(idx), kfmac_o(idx), knc_o(idx), &
                                 dgnA, dgnB, sg_atk, sg_acc, xxlsgA, xxlsgB, &
                                 qs11, qn11, qs22, qn22, &
                                 qs12, qs21, qn12, qv12 )
                  qs11_o(idx) = qs11 ; qn11_o(idx) = qn11
                  qs22_o(idx) = qs22 ; qn22_o(idx) = qn22
                  qs12_o(idx) = qs12 ; qs21_o(idx) = qs21
                  qn12_o(idx) = qn12 ; qv12_o(idx) = qv12

                  ! Wrapper call (PR-G2 target — same fixture serves both).
                  call getcoags_wrapper_f( T, P, dgnA, dgnB, &
                                           sg_atk, sg_acc, xxlsgA, xxlsgB, &
                                           pdens_atk, pdens_acc, &
                                           bij0, bij2i, bij2j, bij3, &
                                           bii0, bii2, bjj0, bjj2 )
                  bij0_o(idx)  = bij0  ; bij2i_o(idx) = bij2i
                  bij2j_o(idx) = bij2j ; bij3_o(idx)  = bij3
                  bii0_o(idx)  = bii0  ; bii2_o(idx)  = bii2
                  bjj0_o(idx)  = bjj0  ; bjj2_o(idx)  = bjj2
               end do
            end do
         end do
      end do

      open(unit=10, file='coag_coefficients_reference.txt', status='replace', action='write')
      write(10,'(a)') '# coag-coefficients reference data'
      write(10,'(a)') "# Sections marked with '%' header lines."
      write(10,'(/a,i0)') '% n_total ', ntot

      ! Physical inputs sweep.
      write(10,'(/a)') '% physical_inputs (rows: ntot; cols: temp press dgnumA dgnumB)'
      do idx = 1, ntot
         write(10,'(4es27.16e3)') temp_o(idx), press_o(idx), &
                                  dgnumA_o(idx), dgnumB_o(idx)
      end do

      ! getcoags intermediates (these are the inputs to getcoags after
      ! the wrapper's prep — also used by PR-G2 for cross-validation).
      write(10,'(/a)') '% getcoags_inputs (rows: ntot; cols: lamda knc kfmat kfmac kfmatac)'
      do idx = 1, ntot
         write(10,'(5es27.16e3)') lamda_o(idx), knc_o(idx), &
                                  kfmat_o(idx), kfmac_o(idx), kfmatac_o(idx)
      end do

      ! getcoags outputs (PR-G1 validation target).
      write(10,'(/a)') '% getcoags_outputs (rows: ntot; cols: qs11 qn11 qs22 qn22 qs12 qs21 qn12 qv12)'
      do idx = 1, ntot
         write(10,'(8es27.16e3)') qs11_o(idx), qn11_o(idx), &
                                  qs22_o(idx), qn22_o(idx), &
                                  qs12_o(idx), qs21_o(idx), &
                                  qn12_o(idx), qv12_o(idx)
      end do

      ! getcoags_wrapper_f outputs (PR-G2 validation target).
      write(10,'(/a)') '% wrapper_outputs (rows: ntot; cols: betaij0 betaij2i betaij2j betaij3 betaii0 betaii2 betajj0 betajj2)'
      do idx = 1, ntot
         write(10,'(8es27.16e3)') bij0_o(idx), bij2i_o(idx), &
                                  bij2j_o(idx), bij3_o(idx), &
                                  bii0_o(idx), bii2_o(idx), &
                                  bjj0_o(idx), bjj2_o(idx)
      end do

      close(10)
      write(*,'(a,i0,a)') 'coag_coefficients_driver: wrote ', ntot, ' records to coag_coefficients_reference.txt'

      end program coag_coefficients_driver
