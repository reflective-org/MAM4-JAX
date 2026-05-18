! physics_types.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

!-------------------------------------------------------------------------------
!physics data types module
!-------------------------------------------------------------------------------
module physics_types

  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid,       only: pcols, pver
  use constituents, only: pcnst, cnst_name

  implicit none
  private          ! Make default type private to the module

! Public types:

  public physics_state
! public physics_tend
  public physics_ptend
  

!-------------------------------------------------------------------------------
  type physics_state
     integer                                     :: &
          lchnk,                &! chunk index
          ngrdcol,              &! -- Grid        -- number of active columns (on the grid)
          psetcols=0,           &! --             -- max number of columns set - if subcols = pcols*psubcols, else = pcols
          ncol=0                 ! --             -- sum of nsubcol for all ngrdcols - number of active columns
!     real(r8), dimension(:), allocatable         :: &
!          lat,     &! latitude (radians)
!          lon,     &! longitude (radians)
!          ps,      &! surface pressure
!          psdry,   &! dry surface pressure
!          phis,    &! surface geopotential
!          ulat,    &! unique latitudes  (radians)
!          ulon      ! unique longitudes (radians)
     real(r8), dimension(pcols,pver)        :: &
          t,       &! temperature (K)
          pmid,    &! midpoint pressure (Pa) 
          pdel      ! layer thickness (Pa)
!     real(r8), dimension(:,:),allocatable        :: &
!          t,       &! temperature (K)
!          u,       &! zonal wind (m/s)
!          v,       &! meridional wind (m/s)
!          s,       &! dry static energy
!          omega,   &! vertical pressure velocity (Pa/s) 
!          pmid,    &! midpoint pressure (Pa) 
!          pmiddry, &! midpoint pressure dry (Pa) 
!          pdel,    &! layer thickness (Pa)
!          pdeldry, &! layer thickness dry (Pa)
!          rpdel,   &! reciprocal of layer thickness (Pa)
!          rpdeldry,&! recipricol layer thickness dry (Pa)
!          lnpmid,  &! ln(pmid)
!          lnpmiddry,&! log midpoint pressure dry (Pa) 
!          exner,   &! inverse exner function w.r.t. surface pressure (ps/p)^(R/cp)
!          zm        ! geopotential height above surface at midpoints (m)

!     real(r8), dimension(:,:,:),allocatable      :: &
     real(r8), dimension(pcols,pver,pcnst)      :: &
          q         ! constituent mixing ratio (kg/kg moist or dry air depending on type)

!     real(r8), dimension(:,:),allocatable        :: &
!          pint,    &! interface pressure (Pa)
!          pintdry, &! interface pressure dry (Pa) 
!          lnpint,  &! ln(pint)
!          lnpintdry,&! log interface pressure dry (Pa) 
!          zi        ! geopotential height above surface at interfaces (m)

!     real(r8), dimension(:),allocatable          :: &
!          te_ini,  &! vertically integrated total (kinetic + static) energy of initial state
!          te_cur,  &! vertically integrated total (kinetic + static) energy of current state
!          tw_ini,  &! vertically integrated total water of initial state
!          tw_cur    ! vertically integrated total water of new state
!     integer :: count ! count of values with significant energy or water imbalances
!     integer, dimension(:),allocatable           :: &
!          latmapback, &! map from column to unique lat for that column
!          lonmapback, &! map from column to unique lon for that column
!          cid        ! unique column id
!     integer :: ulatcnt, &! number of unique lats in chunk
!                uloncnt   ! number of unique lons in chunk

  end type physics_state

!!-------------------------------------------------------------------------------
!  type physics_state
!     integer                                     :: &
!          lchnk,                &! chunk index
!          ngrdcol,              &! -- Grid        -- number of active columns (on the grid)
!          psetcols=0,           &! --             -- max number of columns set - if subcols = pcols*psubcols, else = pcols
!          ncol=0                 ! --             -- sum of nsubcol for all ngrdcols - number of active columns
!     real(r8), dimension(:), allocatable         :: &
!          lat,     &! latitude (radians)
!          lon,     &! longitude (radians)
!          ps,      &! surface pressure
!          psdry,   &! dry surface pressure
!          phis,    &! surface geopotential
!          ulat,    &! unique latitudes  (radians)
!          ulon      ! unique longitudes (radians)
!     real(r8), dimension(:,:),allocatable        :: &
!          t,       &! temperature (K)
!          u,       &! zonal wind (m/s)
!          v,       &! meridional wind (m/s)
!          s,       &! dry static energy
!          omega,   &! vertical pressure velocity (Pa/s) 
!          pmid,    &! midpoint pressure (Pa) 
!          pmiddry, &! midpoint pressure dry (Pa) 
!          pdel,    &! layer thickness (Pa)
!          pdeldry, &! layer thickness dry (Pa)
!          rpdel,   &! reciprocal of layer thickness (Pa)
!          rpdeldry,&! recipricol layer thickness dry (Pa)
!          lnpmid,  &! ln(pmid)
!          lnpmiddry,&! log midpoint pressure dry (Pa) 
!          exner,   &! inverse exner function w.r.t. surface pressure (ps/p)^(R/cp)
!          zm        ! geopotential height above surface at midpoints (m)

!     real(r8), dimension(:,:,:),allocatable      :: &
!          q         ! constituent mixing ratio (kg/kg moist or dry air depending on type)

!     real(r8), dimension(:,:),allocatable        :: &
!          pint,    &! interface pressure (Pa)
!          pintdry, &! interface pressure dry (Pa) 
!          lnpint,  &! ln(pint)
!          lnpintdry,&! log interface pressure dry (Pa) 
!          zi        ! geopotential height above surface at interfaces (m)

!     real(r8), dimension(:),allocatable          :: &
!          te_ini,  &! vertically integrated total (kinetic + static) energy of initial state
!          te_cur,  &! vertically integrated total (kinetic + static) energy of current state
!          tw_ini,  &! vertically integrated total water of initial state
!          tw_cur    ! vertically integrated total water of new state
!     integer :: count ! count of values with significant energy or water imbalances
!     integer, dimension(:),allocatable           :: &
!          latmapback, &! map from column to unique lat for that column
!          lonmapback, &! map from column to unique lon for that column
!          cid        ! unique column id
!     integer :: ulatcnt, &! number of unique lats in chunk
!                uloncnt   ! number of unique lons in chunk

!  end type physics_state

!!-------------------------------------------------------------------------------
!  type physics_tend

!     integer   ::   psetcols=0 ! max number of columns set- if subcols = pcols*psubcols, else = pcols

!     real(r8), dimension(:,:),allocatable        :: dtdt, dudt, dvdt
!     real(r8), dimension(:),  allocatable        :: flx_net
!     real(r8), dimension(:),  allocatable        :: &
!          te_tnd,  &! cumulative boundary flux of total energy
!          tw_tnd    ! cumulative boundary flux of total water
!  end type physics_tend

!-------------------------------------------------------------------------------
! This is for tendencies returned from individual parameterizations
  type physics_ptend

     integer   ::   psetcols=0 ! max number of columns set- if subcols = pcols*psubcols, else = pcols

     character*24 :: name    ! name of parameterization which produced tendencies.

     logical ::             &
          ls = .false.,               &! true if dsdt is returned
          lu = .false.,               &! true if dudt is returned
          lv = .false.                 ! true if dvdt is returned

     logical,dimension(pcnst) ::  lq = .false.  ! true if dqdt() is returned

     integer ::             &
          top_level,        &! top level index for which nonzero tendencies have been set
          bot_level          ! bottom level index for which nonzero tendencies have been set

!    real(r8), dimension(:,:),allocatable   :: &
     real(r8), dimension(pcols,pver)   :: &
          s,                &! heating rate (J/kg/s)
          u,                &! u momentum tendency (m/s/s)
          v                  ! v momentum tendency (m/s/s)
!    real(r8), dimension(:,:,:),allocatable :: &
     real(r8), dimension(pcols,pver,pcnst) :: &
          q                  ! consituent tendencies (kg/kg/s)

!! boundary fluxes
!     real(r8), dimension(:),allocatable     ::&
!          hflux_srf,     &! net heat flux at surface (W/m2)
!          hflux_top,     &! net heat flux at top of model (W/m2)
!          taux_srf,      &! net zonal stress at surface (Pa)
!          taux_top,      &! net zonal stress at top of model (Pa)
!          tauy_srf,      &! net meridional stress at surface (Pa)
!          tauy_top        ! net meridional stress at top of model (Pa)
!     real(r8), dimension(:,:),allocatable   ::&
!          cflx_srf,      &! constituent flux at surface (kg/m2/s)
!          cflx_top        ! constituent flux top of model (kg/m2/s)

  end type physics_ptend


!!-------------------------------------------------------------------------------
!! This is for tendencies returned from individual parameterizations
!  type physics_ptend

!     integer   ::   psetcols=0 ! max number of columns set- if subcols = pcols*psubcols, else = pcols

!     character*24 :: name    ! name of parameterization which produced tendencies.

!     logical ::             &
!          ls = .false.,               &! true if dsdt is returned
!          lu = .false.,               &! true if dudt is returned
!          lv = .false.                 ! true if dvdt is returned

!     logical,dimension(pcnst) ::  lq = .false.  ! true if dqdt() is returned

!     integer ::             &
!          top_level,        &! top level index for which nonzero tendencies have been set
!          bot_level          ! bottom level index for which nonzero tendencies have been set

!     real(r8), dimension(:,:),allocatable   :: &
!          s,                &! heating rate (J/kg/s)
!          u,                &! u momentum tendency (m/s/s)
!          v                  ! v momentum tendency (m/s/s)
!     real(r8), dimension(:,:,:),allocatable :: &
!          q                  ! consituent tendencies (kg/kg/s)

!! boundary fluxes
!     real(r8), dimension(:),allocatable     ::&
!          hflux_srf,     &! net heat flux at surface (W/m2)
!          hflux_top,     &! net heat flux at top of model (W/m2)
!          taux_srf,      &! net zonal stress at surface (Pa)
!          taux_top,      &! net zonal stress at top of model (Pa)
!          tauy_srf,      &! net meridional stress at surface (Pa)
!          tauy_top        ! net meridional stress at top of model (Pa)
!     real(r8), dimension(:,:),allocatable   ::&
!          cflx_srf,      &! constituent flux at surface (kg/m2/s)
!          cflx_top        ! constituent flux top of model (kg/m2/s)

!  end type physics_ptend


!===============================================================================
!contains
!===============================================================================

end module physics_types
