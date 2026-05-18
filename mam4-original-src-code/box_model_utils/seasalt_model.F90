! seasalt_model.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

       module seasalt_model

       implicit none

       integer, parameter, public :: n_ocean_data = 4
#if (defined MODAL_AERO_9MODE || defined MODAL_AERO_4MODE_MOM)
       logical, parameter, public :: has_mam_mom = .true.
#else
       logical, parameter, public :: has_mam_mom = .false.
#endif

       contains

!-------------------------------------------------------------------------------
       subroutine seasalt_init()
       return
       end subroutine seasalt_init

!-------------------------------------------------------------------------------
       end module seasalt_model

