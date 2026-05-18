! cloudchem_simple.F90

   module cloudchem_simple

! !USES:
  use shr_kind_mod,    only:  r8 => shr_kind_r8
  use chem_mods,       only:  gas_pcnst

  implicit none
  private
  save

! !PUBLIC MEMBER FUNCTIONS:
  public cloudchem_simple_sub

! !PUBLIC DATA MEMBERS:
  integer, parameter :: pcnstxx = gas_pcnst


  contains


!----------------------------------------------------------------------
subroutine cloudchem_simple_sub(            &
   lchnk,    ncol,     nstep,               &
   loffset,  deltat,                        &
   q,        qqcw,     cldn                 )

! !USES:
!use modal_aero_data

use cam_logfile,       only:  iulog
use constituents,      only:  pcnst, cnst_name, cnst_get_ind
use ppgrid,            only:  pcols, pver
                                                                                                                                            
use abortutils,        only : endrun

use modal_aero_data


implicit none

! !PARAMETERS:
   integer,  intent(in)    :: lchnk                ! chunk identifier
   integer,  intent(in)    :: ncol                 ! number of atmospheric column
   integer,  intent(in)    :: nstep                ! model time-step number
   integer,  intent(in)    :: loffset              ! offset applied to modal aero "ptrs"
   real(r8), intent(in)    :: deltat               ! time step (s)

   real(r8), intent(inout) :: q(ncol,pver,pcnstxx) ! tracer mixing ratio (TMR) array
                                                   ! *** MUST BE  #/kmol-air for number
                                                   ! *** MUST BE mol/mol-air for mass
                                                   ! *** NOTE ncol dimension
   real(r8), intent(inout) :: qqcw(ncol,pver,pcnstxx) ! like q but for cloud-borner tracers
   real(r8), intent(in)    :: cldn(ncol,pver)      ! cloud fraction

! !DESCRIPTION: 
! computes TMR (tracer mixing ratio) tendencies for gas condensation
!    onto aerosol particles
!
! !REVISION HISTORY:
!   RCE 07.04.13:  Adapted from MIRAGE2 code
!
!EOP
!----------------------------------------------------------------------
!BOC

! local variables
   integer :: i
   integer :: k
   integer :: l_h2so4g, l_nh3g, l_so2g
   integer :: l_num_c1, l_num_c2, l_nh4_c1, l_nh4_c2, l_so4_c1, l_so4_c2

   real (r8) :: tau_cloudchem_simple
   real (r8) :: tmpa, tmpb, tmpc, tmpd, tmpe, tmpf


! set gas species indices
   call cnst_get_ind( 'H2SO4', l_h2so4g, .false. )
   call cnst_get_ind( 'SO2',   l_so2g,   .false. )
   call cnst_get_ind( 'NH3',   l_nh3g,   .false. )
   l_h2so4g = l_h2so4g - loffset
   l_so2g   = l_so2g   - loffset
   l_nh3g   = l_nh3g   - loffset
   if ((l_h2so4g <= 0) .or. (l_so2g <= 0)) then
      write( iulog, '(/a/a,3i7)' )   &
         '*** cloudchem_simple_sub -- cannot find gas species',   &
         '    l_h2so4g, l_so2g, loffset =', l_h2so4g, l_so2g, loffset
      call endrun( 'cloudchem_simple_sub error' )
   else
      write( *, '(/a,4i7)' )   &
         '*** cloudchem_simple_sub -- l_h2so4g, so2, nh3 =', l_h2so4g, l_so2g, l_nh3g
   end if

   l_num_c1 = numptrcw_amode(modeptr_accum) - loffset
   l_num_c2 = numptrcw_amode(modeptr_aitken) - loffset
   l_nh4_c1 = lptr_nh4_cw_amode(modeptr_accum) - loffset
   l_nh4_c2 = lptr_nh4_cw_amode(modeptr_aitken) - loffset
   l_so4_c1 = lptr_so4_cw_amode(modeptr_accum) - loffset
   l_so4_c2 = lptr_so4_cw_amode(modeptr_aitken) - loffset

   tau_cloudchem_simple = 1800.0_r8

   do k = 1, pver
   do i = 1, ncol

      if (cldn(i,k) <= 0.009_r8) cycle
      tmpf = min( 1.0_r8, cldn(i,k) )

      tmpd = max( qqcw(i,k,l_num_c1), 1.0_r8 )
      tmpe = max( qqcw(i,k,l_num_c2), 0.0_r8 )
      tmpd = tmpd/(tmpd + tmpe)
      tmpe = max( 0.0_r8, 1.0_r8 - tmpd )

      tmpa = tmpf * q(i,k,l_so2g)*exp( -deltat/tau_cloudchem_simple )
      tmpb = tmpf * q(i,k,l_h2so4g)

      q(i,k,l_so2g) = q(i,k,l_so2g) - tmpa
      q(i,k,l_h2so4g) = q(i,k,l_h2so4g) - tmpb
      qqcw(i,k,l_so4_c1) = qqcw(i,k,l_so4_c1) + tmpd*(tmpa + tmpb)
      qqcw(i,k,l_so4_c2) = qqcw(i,k,l_so4_c2) + tmpe*(tmpa + tmpb)

      if (l_nh3g > 0 .and. l_nh4_c1 > 0 .and. l_nh4_c2 > 0) then
         tmpc = min( tmpa+tmpb, tmpf*q(i,k,l_nh3g) )
         q(i,k,l_nh3g) = q(i,k,l_nh3g) - tmpc
         qqcw(i,k,l_nh4_c1) = qqcw(i,k,l_nh4_c1) + tmpd*tmpc
         qqcw(i,k,l_nh4_c2) = qqcw(i,k,l_nh4_c2) + tmpe*tmpc
      end if
   end do
   end do

   return
   end subroutine cloudchem_simple_sub


!----------------------------------------------------------------------
   end module cloudchem_simple
