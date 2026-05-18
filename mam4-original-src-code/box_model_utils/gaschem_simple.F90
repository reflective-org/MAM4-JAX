! gaschem_simple.F90

   module gaschem_simple

! !USES:
  use shr_kind_mod,    only:  r8 => shr_kind_r8
  use chem_mods,       only:  gas_pcnst

  implicit none
  private
  save

! !PUBLIC MEMBER FUNCTIONS:
  public gaschem_simple_sub

! !PUBLIC DATA MEMBERS:
  integer, parameter :: pcnstxx = gas_pcnst


  contains


!----------------------------------------------------------------------
subroutine gaschem_simple_sub(                     &
   lchnk,    ncol,     nstep,               &
   loffset,  deltat,                        &
   q,                  tau_gaschem_simple      )

! !USES:
!use modal_aero_data

use cam_logfile,       only:  iulog
use constituents,      only:  pcnst, cnst_name, cnst_get_ind
use ppgrid,            only:  pcols, pver
                                                                                                                                            
use abortutils,        only : endrun


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
   real(r8), intent(inout) :: tau_gaschem_simple(ncol,pver)
                                                   ! like q but for cloud-borner tracers

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
   integer, parameter :: jsrflx_gaexch = 1
   integer, parameter :: jsrflx_rename = 2
   integer, parameter :: ldiag1=-1, ldiag2=-1, ldiag3=-1, ldiag4=-1
   integer, parameter :: method_soa = 2
!     method_soa=0 is no uptake
!     method_soa=1 is irreversible uptake done like h2so4 uptake
!     method_soa=2 is reversible uptake using subr modal_aero_soaexch

   integer :: i
   integer :: k
   integer :: l_h2so4g, l_so2g

   real (r8) :: tmpa, tmpb


! set gas species indices
   call cnst_get_ind( 'H2SO4', l_h2so4g, .false. )
   call cnst_get_ind( 'SO2',   l_so2g, .false. )
   l_h2so4g = l_h2so4g - loffset
   l_so2g = l_so2g - loffset
   if ((l_h2so4g <= 0) .or. (l_h2so4g > pcnstxx) .or. &
       (l_so2g <= 0) .or. (l_so2g > pcnstxx)) then
      write( iulog, '(/a/a,2i7)' )   &
         '*** gaschem_simple_sub -- cannot find H2SO4 species',   &
         '    l_h2so4g, loffset =', l_h2so4g, loffset
      call endrun( 'gaschem_simple_sub error' )
   else
      write( *, '(/a,2i7)' )   &
         '*** gaschem_simple_sub -- l_so2g, l_h2so4g =', l_so2g, l_h2so4g
   end if

   do k = 1, pver
   do i = 1, ncol
      tmpa = q(i,k,l_so2g)*exp( -deltat/tau_gaschem_simple(i,k) )
      tmpb = q(i,k,l_so2g) - tmpa
      q(i,k,l_so2g) = tmpa
      q(i,k,l_h2so4g) = tmpb
   end do
   end do

   return
   end subroutine gaschem_simple_sub


!----------------------------------------------------------------------
   end module gaschem_simple
