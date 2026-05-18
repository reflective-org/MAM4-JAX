! phys_control.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

      module phys_control

      use shr_kind_mod, only: r8 => shr_kind_r8

     implicit none

      public

      integer :: mam_amicphys_optaa = 1

      logical :: convproc_do_aer = .true.
      logical :: convproc_do_gas = .false.
      logical :: history_aerosol = .true.
      logical :: history_aerocom = .true.
      logical :: history_verbose = .true.
      logical :: pergro_mods     = .true.

      real(r8) :: n_so4_monolayers_pcage = 3.0_r8

      contains

!==============================================================================

  subroutine phys_getopts( &
    mam_amicphys_optaa_out, &
    convproc_do_aer_out, convproc_do_gas_out, &
    history_aerosol_out, history_aerocom_out, &
    history_verbose_out, pergro_mods_out, &
    n_so4_monolayers_pcage_out )

    integer, optional, intent(out) :: mam_amicphys_optaa_out

    logical, optional, intent(out) :: convproc_do_aer_out
    logical, optional, intent(out) :: convproc_do_gas_out
    logical, optional, intent(out) :: history_aerosol_out
    logical, optional, intent(out) :: history_aerocom_out
    logical, optional, intent(out) :: history_verbose_out
    logical, optional, intent(out) :: pergro_mods_out

    real(r8), optional, intent(out) :: n_so4_monolayers_pcage_out

    if ( present( mam_amicphys_optaa_out ) ) then
       mam_amicphys_optaa_out = mam_amicphys_optaa
    end if
    if ( present( convproc_do_aer_out ) ) then
       convproc_do_aer_out = convproc_do_aer
    end if
    if ( present( convproc_do_gas_out ) ) then
       convproc_do_gas_out = convproc_do_gas
    end if
    if ( present( history_aerosol_out ) ) then
       history_aerosol_out = history_aerosol
    end if
    if ( present( history_aerocom_out ) ) then
       history_aerocom_out = history_aerocom
    end if
    if ( present( history_verbose_out ) ) then
       history_verbose_out = history_verbose
    end if
    if ( present( pergro_mods_out ) ) then
       pergro_mods_out = pergro_mods
    end if
    if ( present( n_so4_monolayers_pcage_out ) ) then
       n_so4_monolayers_pcage_out = n_so4_monolayers_pcage
    end if
!   if ( present( x_out ) ) then
!      x_out = x
!   end if

    return
  end subroutine phys_getopts

!==============================================================================================

      end module phys_control
