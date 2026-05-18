! mo_chem_utls.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

module mo_chem_utls

  private
  public :: get_spc_ndx

  save

contains

  integer function get_spc_ndx( spc_name )
    !-----------------------------------------------------------------------
    !     ... return overall species index associated with spc_name
    !-----------------------------------------------------------------------

    use chem_mods,     only : gas_pcnst
    use mo_tracname,   only : tracnam => solsym

    implicit none

    !-----------------------------------------------------------------------
    !     ... dummy arguments
    !-----------------------------------------------------------------------
    character(len=*), intent(in) :: spc_name

    !-----------------------------------------------------------------------
    !     ... local variables
    !-----------------------------------------------------------------------
    integer :: m

    get_spc_ndx = -1
    do m = 1,gas_pcnst
       if( trim( spc_name ) == trim( tracnam(m) ) ) then
          get_spc_ndx = m
          exit
       end if
    end do

  end function get_spc_ndx

end module mo_chem_utls
