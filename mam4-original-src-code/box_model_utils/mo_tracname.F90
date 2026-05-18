! mo_tracname.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

      module mo_tracname

      use shr_kind_mod, only: r8 => shr_kind_r8
      use chem_mods, only:  gas_pcnst

      implicit none

      public

      character(len=16) :: solsym(gas_pcnst) = '????????????????'

      end module mo_tracname
