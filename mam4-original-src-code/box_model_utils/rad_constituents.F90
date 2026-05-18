! rad_constituents.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

      module rad_constituents
!
! provides limited functionality of the CAM rad_constituents module
! that is needed by the following modules
!    modal_aero_calcsize.F90
!    modal_aero_initialize_data.F90
!    modal_aero_wateruptake.F90
!

      use shr_kind_mod,   only: r8 => shr_kind_r8

      use abortutils,     only: endrun
      use cam_logfile,    only: iulog
      use radconstants,   only: nlwbands, nswbands
      use physics_types,  only: physics_state, physics_ptend
      use physics_buffer, only: physics_buffer_desc

      use modal_aero_data, only: &
          lspectype_amode, nspec_amode, ntot_amode, specname_amode

      implicit none

      public

      public :: rad_cnst_get_mode_props
      public :: rad_cnst_get_info
      public :: rad_cnst_get_aer_props
      public :: rad_cnst_get_mode_num
      public :: rad_cnst_get_aer_mmr

      interface rad_cnst_get_info
         module procedure rad_cnst_get_info_v1
         module procedure rad_cnst_get_info_v2
         module procedure rad_cnst_get_info_v3
      end interface

# if ( defined MOSAIC_SPECIES ) 
      integer, parameter :: nspecs=11

      integer :: mode_idx_rc(nspecs) = &
                 (/ 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1 /)
      integer :: spec_idx_rc(nspecs) = &
                 (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 /)

      character(len=16) :: specname_amode_rc(nspecs) = &
                           (/ 'sulfate         ', &
                              'ammonium        ', &
                              'nitrate         ', &
                              'p-organic       ', &
                              's-organic       ', &
                              'black-c         ', &
                              'seasalt         ', &
                              'dust            ', &
                              'chloride        ', &
                              'calcium         ', &
                              'carbonate       ' /)

      real(r8) :: specdens_amode_rc(nspecs) = &
                  (/ 1770.0_r8, 1770.0_r8, 1770.0_r8, &
                     1000.0_r8, 1000.0_r8, 1700.0_r8, &
                     1900.0_r8, 2600.0_r8, 1900.0_r8, & 
                     2600.0_r8, 2600.0_r8             /)
      real(r8) :: spechygro_rc(nspecs) = &
                  (/ 0.507_r8, 0.507_r8, 0.507_r8,   &
                     0.010_r8, 0.140_r8, 1.0e-10_r8, &
                     1.160_r8, 0.068_r8, 1.160_r8,   & 
                     0.068_r8, 0.068_r8              /)

#elif ( defined MODAL_AERO_4MODE_MOM )
      integer, parameter :: nspecs=9

      integer :: mode_idx_rc(nspecs) = &
                 (/ 1, 1, 1, 1, 1, 1, 1, 1, 1 /)
      integer :: spec_idx_rc(nspecs) = &
                 (/ 1, 2, 3, 4, 5, 6, 7, 8, 9 /)

      character(len=16) :: specname_amode_rc(nspecs) = &
                           (/ 'sulfate         ', &
                              'ammonium        ', &
                              'nitrate         ', &
                              'p-organic       ', &
                              's-organic       ', &
                              'black-c         ', &
                              'seasalt         ', &
                              'dust            ', &
                              'm-organic       ' /)

      real(r8) :: specdens_amode_rc(nspecs) = &
                  (/ 1770.0_r8, 1770.0_r8, 1770.0_r8, &
                     1000.0_r8, 1000.0_r8, 1700.0_r8, &
                     1900.0_r8, 2600.0_r8, 1601.0_r8  /)
      real(r8) :: spechygro_rc(nspecs) = &
                  (/ 0.507_r8, 0.507_r8, 0.507_r8,   &
                     0.010_r8, 0.140_r8, 1.0e-10_r8, &
                     1.160_r8, 0.068_r8, 0.100_r8    /)

#else
      integer, parameter :: nspecs=8

      integer :: mode_idx_rc(nspecs) = &
                 (/ 1, 1, 1, 1, 1, 1, 1, 1 /)
      integer :: spec_idx_rc(nspecs) = &
                 (/ 1, 2, 3, 4, 5, 6, 7, 8 /)

      character(len=16) :: specname_amode_rc(nspecs) = &
                           (/ 'sulfate         ', &
                              'ammonium        ', &
                              'nitrate         ', &
                              'p-organic       ', &
                              's-organic       ', &
                              'black-c         ', &
                              'seasalt         ', &
                              'dust            ' /)

      real(r8) :: specdens_amode_rc(nspecs) = &
                  (/ 1770.0_r8, 1770.0_r8, 1770.0_r8, &
                     1000.0_r8, 1000.0_r8, 1700.0_r8, &
                     1900.0_r8, 2600.0_r8             /)
      real(r8) :: spechygro_rc(nspecs) = &
                  (/ 0.507_r8, 0.507_r8, 0.507_r8,   &
                     0.010_r8, 0.140_r8, 1.0e-10_r8, &
                     1.160_r8, 0.068                 /)

#endif

!     if      (cnst_name(qArrIndex)(1:4) == 'so4_') then
!        specdens_amode(l,m) = 1770.0_r8
!        spechygro(l,m)      = 0.507_r8
!     else if (cnst_name(qArrIndex)(1:4) == 'nh4_') then
!        specdens_amode(l,m) = 1770.0_r8
!        spechygro(l,m)      = 0.507_r8
!     else if (cnst_name(qArrIndex)(1:4) == 'no3_') then
!        specdens_amode(l,m) = 1770.0_r8
!        spechygro(l,m)      = 0.507_r8
!     else if (cnst_name(qArrIndex)(1:4) == 'pom_') then
!        specdens_amode(l,m) = 1000.0_r8
!        spechygro(l,m)      = 0.010_r8
!     else if (cnst_name(qArrIndex)(1:4) == 'soa_') then
!        specdens_amode(l,m) = 1000.0_r8
!        spechygro(l,m)      = 0.140_r8
!     else if (cnst_name(qArrIndex)(1:3) == 'bc_') then
!        specdens_amode(l,m) = 1700.0_r8
!        spechygro(l,m)      = 1.0e-10_r8
!     else if (cnst_name(qArrIndex)(1:4) == 'ncl_') then
!        specdens_amode(l,m) = 1900.0_r8
!        spechygro(l,m)      = 1.160_r8
!     else if (cnst_name(qArrIndex)(1:4) == 'dst_') then
!        specdens_amode(l,m) = 2600.0_r8
!        spechygro(l,m)      = 0.068_r8

#if ( defined MODAL_AERO_7MODE )
      integer, parameter :: n_modes=7
      real(r8) :: dgnum_amode_rc(n_modes)   = (/ 0.1100e-6, 0.0260e-6, 0.050e-6, 0.200e-6, 0.100e-6, 2.000e-6, 1.000e-6 /)
      real(r8) :: dgnumlo_amode_rc(n_modes) = (/ 0.0535e-6, 0.0087e-6, 0.010e-6, 0.050e-6, 0.050e-6, 1.000e-6, 0.500e-6 /)
      real(r8) :: dgnumhi_amode_rc(n_modes) = (/ 0.4400e-6, 0.0520e-6, 0.100e-6, 1.000e-6, 0.500e-6, 4.000e-6, 2.000e-6 /)
      real(r8) :: sigmag_amode_rc(n_modes)  = (/ 1.800, 1.600, 1.600, 2.000, 1.800, 2.000, 1.800 /)
#elif ( defined MODAL_AERO_4MODE || defined MODAL_AERO_4MODE_MOM || defined MODAL_AERO_4MODE_VBS )
      integer, parameter :: n_modes=4
      real(r8) :: dgnum_amode_rc(n_modes)   = (/ 0.1100e-6_r8, 0.0260e-6_r8, 2.000e-6_r8, 0.050e-6_r8 /)
      real(r8) :: dgnumlo_amode_rc(n_modes) = (/ 0.0535e-6_r8, 0.0087e-6_r8, 1.000e-6_r8, 0.010e-6_r8 /)
      real(r8) :: dgnumhi_amode_rc(n_modes) = (/ 0.4400e-6_r8, 0.0520e-6_r8, 4.000e-6_r8, 0.100e-6_r8 /)
      real(r8) :: sigmag_amode_rc(n_modes)  = (/ 1.800_r8, 1.600_r8, 1.800_r8, 1.600_r8 /)
#elif ( defined MODAL_AERO_3MODE )
      integer, parameter :: n_modes=3
      real(r8) :: dgnum_amode_rc(n_modes)   = (/ 0.1100e-6, 0.0260e-6, 2.000e-6 /)
      real(r8) :: dgnumlo_amode_rc(n_modes) = (/ 0.0535e-6, 0.0087e-6, 1.000e-6 /)
      real(r8) :: dgnumhi_amode_rc(n_modes) = (/ 0.4400e-6, 0.0520e-6, 4.000e-6 /)
      real(r8) :: sigmag_amode_rc(n_modes)  = (/ 1.800, 1.600, 1.800 /)
#endif

      real(r8), parameter :: rhcrystal_amode_rc(n_modes)   = 0.350_r8
      real(r8), parameter :: rhdeliques_amode_rc(n_modes)  = 0.800_r8

      complex(r8), target :: tmp_refindex_aer_sw(nswbands)
      complex(r8), target :: tmp_refindex_aer_lw(nlwbands)


      contains

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!     call rad_cnst_get_mode_props(0, m, &
!        sigmag=sigmag_amode(m), dgnum=dgnum_amode(m), dgnumlo=dgnumlo_amode(m), &
!        dgnumhi=dgnumhi_amode(m), rhcrystal=rhcrystal_amode(m), rhdeliques=rhdeliques_amode(m))

      subroutine rad_cnst_get_mode_props( itmpa, m, &
         sigmag, dgnum, dgnumlo, &
         dgnumhi, rhcrystal, rhdeliques )

      integer :: itmpa
      integer :: m
      real(r8), optional :: sigmag
      real(r8), optional :: dgnum
      real(r8), optional :: dgnumlo
      real(r8), optional :: dgnumhi
      real(r8), optional :: rhcrystal
      real(r8), optional :: rhdeliques

      character(len=120) :: errmsg

      real(r8) :: dgnum1, dgnum2, dgnum3, dgnum4, &
         sigmag1, sigmag2, sigmag3, sigmag4
      namelist /size_parameters/ dgnum1, dgnum2, dgnum3, dgnum4, &
         sigmag1, sigmag2, sigmag3, sigmag4

      open (UNIT = 101, FILE = 'namelist', STATUS = 'OLD')
            read (101, size_parameters)
      close (101)

! Overwrite default size parameters
      dgnum_amode_rc = (/ dgnum1,dgnum2,dgnum3,dgnum4 /)
      sigmag_amode_rc = (/ sigmag1,sigmag2,sigmag3,sigmag4 /)

      if (m < 1 .or. m > n_modes) then
         write(errmsg,'(a,i12)') 'rad_cnst_get_mode_props - bad m = ', m
         call endrun( errmsg )
      end if

      if ( present( sigmag     ) ) sigmag = sigmag_amode_rc(m)
      if ( present( dgnum      ) ) dgnum = dgnum_amode_rc(m)
      if ( present( dgnumlo    ) ) dgnumlo = dgnumlo_amode_rc(m)
      if ( present( dgnumhi    ) ) dgnumhi = dgnumhi_amode_rc(m)
      if ( present( rhcrystal  ) ) rhcrystal = rhcrystal_amode_rc(m)
      if ( present( rhdeliques ) ) rhdeliques = rhdeliques_amode_rc(m)

      return
      end subroutine rad_cnst_get_mode_props


!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!     call rad_cnst_get_info(0, specname_amode(l), mode_idx=m_idx, spec_idx=s_idx)

      subroutine rad_cnst_get_info_v1( itmpa, specname, mode_idx, spec_idx )

      integer :: itmpa
      character(len=*) :: specname
      integer :: mode_idx
      integer :: spec_idx

      integer :: itmpm, itmps, l1, l2, l3, m
      character(len=120) :: errmsg

      itmpm = -999888777
      itmps = -999888777
m_loop: do m = 1, ntot_amode
         do l1 = 1, nspec_amode(m)
            l2 = lspectype_amode(l1,m)
            do l3 = 1, nspecs
               if ( (specname == specname_amode(l2))    .and. &
                    (specname == specname_amode_rc(l3)) ) then
                  itmpm = m
                  itmps = l1
                  exit m_loop
               end if
            end do
         end do
      end do m_loop
!     do l = 1, nspecs
!        if (specname == specname_amode_rc(l)) then
!           itmpm = mode_idx_rc(l)
!           itmps = spec_idx_rc(l)
!        end if
!     end do

      if (itmpm <= 0 .or. itmps <= 0) then
         write(iulog,'(3a)') 'rad_cnst_get_info_v1 warning - specname = |', specname, '|'
      end if

      mode_idx = itmpm
      spec_idx = itmps

      return
      end subroutine rad_cnst_get_info_v1


!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
      subroutine rad_cnst_get_info_v2( itmpa, nmodes )

      integer :: itmpa
      integer :: nmodes
      !integer, optional :: nmodes

      nmodes   = n_modes  
      !if ( present( nmodes   ) ) nmodes   = n_modes  

      return
      end subroutine rad_cnst_get_info_v2


!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
      subroutine rad_cnst_get_info_v3( itmpa, n, nspec )

      use modal_aero_data, only:  nspec_amode

      integer :: itmpa
      integer :: n
      integer :: nspec
      !integer, optional :: nspec

      character(len=120) :: errmsg

      if (n <= 0 .or. n > n_modes) then
         write(errmsg,'(a,i12)') 'rad_cnst_get_info_v3 - bad n = ', n
         call endrun( errmsg )
      end if

! nspec provided here is used by routines that are never active in the cambox test driver
!    so this is just needed for compiling and linking
      !if ( present( nspec   ) ) nspec = nspec_amode(n)
      nspec = nspec_amode(n)

      return
      end subroutine rad_cnst_get_info_v3


!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!     call rad_cnst_get_aer_props(0, m_idx, s_idx, &
!        refindex_aer_sw=refindex_aer_sw, &
!        refindex_aer_lw=refindex_aer_lw, &
!        density_aer=specdens_amode(l), &
!        hygro_aer=spechygro(l))

      subroutine rad_cnst_get_aer_props( itmpa, m_idx, s_idx, &
         refindex_aer_sw, &
         refindex_aer_lw, &
         density_aer, &
         hygro_aer )

      integer :: itmpa
      integer :: m_idx
      integer :: s_idx
!     complex(r8), dimension(nswbands), optional :: refindex_aer_sw
!     complex(r8), dimension(nlwbands), optional :: refindex_aer_lw
      complex(r8), optional, pointer :: refindex_aer_sw(:)
      complex(r8), optional, pointer :: refindex_aer_lw(:)
      real(r8), optional :: density_aer
      real(r8), optional :: hygro_aer

      integer :: itmpl, l2, l3
      character(len=120) :: errmsg

      itmpl = 0
      l2 = lspectype_amode(s_idx,m_idx)
      do l3 = 1, nspecs
         if (specname_amode(l2) == specname_amode_rc(l3)) then
            itmpl = l3
            exit
         end if
      end do

      if (itmpl <= 0) then
         write(errmsg,'(a,2(1x,i12))') &
            'rad_cnst_get_aer_props - bad m_idx, s_idx = ', m_idx, s_idx
         call endrun( errmsg )
      end if

!     specrefndxsw(i,l,m) = ( 1.1_r8, 1.0e-6_r8 )
!     specrefndxlw(i,l,m) = ( 1.1_r8, 1.0e-6_r8 )
      if ( present ( refindex_aer_sw ) ) then
         tmp_refindex_aer_sw(:nswbands) = ( 1.1_r8, 1.0e-6_r8 )
         refindex_aer_sw => tmp_refindex_aer_sw
      end if

      if ( present ( refindex_aer_lw ) ) then
         tmp_refindex_aer_lw(:nlwbands) = ( 1.1_r8, 1.0e-6_r8 )
         refindex_aer_lw => tmp_refindex_aer_lw
      end if

      if ( present ( density_aer ) ) &
         density_aer = specdens_amode_rc(itmpl)

      if ( present ( hygro_aer ) ) &
         hygro_aer = spechygro_rc(itmpl)

      return
      end subroutine rad_cnst_get_aer_props


!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
      subroutine rad_cnst_get_mode_num( list_idx, mode_idx, phase, state, pbuf, num )
! this is just a stub
! it is called from modal_aero_calcsize_diag which is not used in the cambox test driver

! Return pointer to number mixing ratio for the aerosol mode from the specified
! climate or diagnostic list.

      use constituents, only: pcnst
      use modal_aero_data, only:  numptr_amode

! Arguments
      integer,                     intent(in) :: list_idx    ! index of the climate or a diagnostic list
      integer,                     intent(in) :: mode_idx    ! mode index
      character(len=1),            intent(in) :: phase       ! 'a' for interstitial, 'c' for cloud borne
      type(physics_state), target, intent(in) :: state
      type(physics_buffer_desc),   pointer    :: pbuf(:)
      real(r8),                    pointer    :: num(:,:)

      integer :: idx
      character(len=120) :: errmsg

      if (mode_idx < 1 .or. mode_idx > n_modes) then
         write(errmsg,'(a,i12)') &
            'rad_cnst_get_mode_num - bad mode_idx = ', mode_idx
         call endrun( errmsg )
      end if
      if (phase /= 'a') then
         write(errmsg,'(a,2x,a)') &
            'rad_cnst_get_mode_num - bad phase = ', phase
         call endrun( errmsg )
      end if

      idx = numptr_amode(mode_idx)
      if (idx < 1 .or. idx > pcnst) then
         write(errmsg,'(a,i12)') &
            'rad_cnst_get_mode_num - bad numptr = ', idx
         call endrun( errmsg )
      end if
      num => state%q(:,:,idx)

      return
      end subroutine rad_cnst_get_mode_num


!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
      subroutine rad_cnst_get_aer_mmr( list_idx, mode_idx, spec_idx, phase, state, pbuf, mmr )
! this is just a stub
! it is called from modal_aero_calcsize_diag which is not used in the cambox test driver

! Return pointer to mass mixing ratio for the modal aerosol specie from the specified
! climate or diagnostic list.

      use ppgrid, only: pver
      use constituents, only: pcnst
      use modal_aero_data, only:  lmassptr_amode, nspec_amode

! Arguments
      integer,                     intent(in) :: list_idx    ! index of the climate or a diagnostic list
      integer,                     intent(in) :: mode_idx    ! mode index
      integer,                     intent(in) :: spec_idx    ! index of specie in the mode
      character(len=1),            intent(in) :: phase       ! 'a' for interstitial, 'c' for cloud borne
      type(physics_state), target, intent(in) :: state
      type(physics_buffer_desc),   pointer    :: pbuf(:)
      real(r8),                    pointer    :: mmr(:,:)

      integer :: idx
      character(len=120) :: errmsg

      if (mode_idx < 1 .or. mode_idx > n_modes) then
         write(errmsg,'(a,i12)') &
            'rad_cnst_get_aer_mmr - bad mode_idx = ', mode_idx
         call endrun( errmsg )
      end if
      if (spec_idx < 1 .or. spec_idx > nspec_amode(mode_idx)) then
         write(errmsg,'(a,i12)') &
            'rad_cnst_get_aer_mmr - bad spec_idx = ', spec_idx
         call endrun( errmsg )
      end if
      if (phase /= 'a') then
         write(errmsg,'(a,2x,a)') &
            'rad_cnst_get_aer_mmr - bad phase = ', phase
         call endrun( errmsg )
      end if

      idx = lmassptr_amode(spec_idx,mode_idx)
      if (idx < 1 .or. idx > pcnst) then
         write(errmsg,'(a,i12)') &
            'rad_cnst_get_aer_mmr - bad lmassptr = ', idx
         call endrun( errmsg )
      end if
      mmr => state%q(:,:,idx)

      return
      end subroutine rad_cnst_get_aer_mmr


!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

      end module rad_constituents
