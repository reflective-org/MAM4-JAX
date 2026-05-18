! infnan.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

module infnan
!-------------------------------------------------------------------------
!
! Purpose:
!
! Set parameters for the floating point flags "inf" Infinity
! and "nan" not-a-number. As well as "bigint" the point
! at which integers start to overflow. These values are used
! to initialize arrays with as a way to detect if arrays
! are being used before being set.
!
! Author: CCM Core group
!
! $Id$
!
!-------------------------------------------------------------------------
  use shr_kind_mod, only: r8 => shr_kind_r8

  implicit none

#if ( ( defined __PGI ) || ( defined CPRPGI ) )
! quiet nan for portland group compilers
  real(r8), parameter :: inf = O'0777600000000000000000'
  real(r8), parameter :: nan = O'0777700000000000000000'

#elif ( defined CPRNAG )
! *** THIS NEEDS MORE WORK                                                ***
! *** the 2 lines in the "else" block produce errors with nagfor compiler ***
! *** the 1.0e300 lines here are a temporary bandaid                      ***
! *** NOTE ALSO that infnan is only used in the amicphys-mosaic interface ***
! ***      and mosaic is not incorporated in E3SM or cmdv-se verification ***
  real(r8), parameter :: inf =  1.0e300_r8
  real(r8), parameter :: nan = -1.0e300_r8

#else
! signaling nan otherwise
  real(r8), parameter :: inf = O'0777600000000000000000'
  real(r8), parameter :: nan = O'0777610000000000000000'
#endif

  integer,  parameter :: bigint = O'17777777777'           ! largest possible 32-bit integer
  real(r8), parameter :: uninit_r8 = inf                   ! uninitialized floating point number

end module infnan
