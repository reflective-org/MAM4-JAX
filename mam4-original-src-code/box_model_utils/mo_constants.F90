! mo_constants.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

      module mo_constants

      use shr_kind_mod, only:  r8 => shr_kind_r8

      use physconst, only:  pi, &
                            avogadro_kmol => avogad, &
                            rgas_kmol => r_universal

      implicit none

      public

!     integer, parameter :: qakola_mo_constants = 0

      real(r8), parameter ::  avogadro = avogadro_kmol*1.e-3_r8 ! Avogadro numb - molecules/mole

      real(r8), parameter ::  rgas = rgas_kmol*1.e-3_r8         ! Gas constant (J/K/mol)

      end module mo_constants
