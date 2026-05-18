! test310_driver.F90
! changed on 09/02/19 for namelist input
!
! this does a test of mam4 (with marine organics and coarse-mode
!    carbonaceous species) that exercisess modal_aero_calcsize,
!    modal_aero_wateruptake, and modal_aero_amicphys (all the amicphys processes),
!    with cloud fraction = 0.
!
! there is no benchmark solution for this test case.
!
! this test driver takes namelist input for initial condition
!
!-------------------------------------------------------------------------------

      module driver

      use shr_kind_mod, only: r8 => shr_kind_r8
      use abortutils, only: endrun
      use cam_logfile, only: iulog
      use constituents, only: pcnst, cnst_name, cnst_get_ind
      use modal_aero_data, only: ntot_amode
      use ppgrid, only: pcols, pver, begchunk, endchunk
      use netcdf

      implicit none

      public

      integer, parameter :: lun_outfld = 90

      integer :: mdo_gaschem, mdo_cloudchem
      integer :: mdo_gasaerexch, mdo_rename, mdo_newnuc, mdo_coag
      integer :: mopt_aero_comp, mopt_aero_load, mopt_ait_size
      integer :: mopt_h2so4_uptake
      integer :: i_cldy_sameas_clear
      integer :: iwrite3x_species_flagaa, iwrite3x_units_flagaa
      integer :: iwrite4x_heading_flagbb
      real(r8) :: xopt_cloudf

      ! in the multiple nbc/npoa code, the following are in modal_aero_data
      integer :: lptr_bca_a_amode(ntot_amode) = -999888777 
      integer :: lptr_poma_a_amode(ntot_amode) = -999888777 

      integer :: species_class(pcnst) = -1

      contains


!-------------------------------------------------------------------------------
      subroutine cambox_main

      use cam_history,             only: ncol_for_outfld
      use wv_saturation,           only: ncol_for_qsat
      use modal_aero_data,         only: ntot_amode
      use physics_buffer, only: physics_buffer_desc

      use modal_aero_data, only: &
         lmassptrcw_amode, nspec_amode, numptrcw_amode, &
         qqcw_get_field


      integer, parameter :: ncolxx = min( pcols, 10 )

      integer  :: ncol
      integer  :: nstop

      real(r8) :: deltat
      real(r8) :: t(pcols,pver)      ! Temperature in Kelvin
      real(r8) :: pmid(pcols,pver)   ! pressure at model levels (Pa)
      real(r8) :: pdel(pcols,pver)   ! pressure thickness of levels
      real(r8) :: zm(pcols,pver)     ! midpoint height above surface (m)
      real(r8) :: pblh(pcols)        ! pbl height (m)
      real(r8) :: relhum(pcols,pver) ! layer relative humidity
      real(r8) :: qv(pcols,pver)     ! layer specific humidity
      real(r8) :: cld(pcols,pver)    ! stratiform cloud fraction

      real(r8) :: q(pcols,pver,pcnst)     ! Tracer MR array
      real(r8) :: qqcw(pcols,pver,pcnst)  ! Cloudborne aerosol MR array
      real(r8) :: dgncur_a(pcols,pver,ntot_amode)
      real(r8) :: dgncur_awet(pcols,pver,ntot_amode)
      real(r8) :: qaerwat(pcols,pver,ntot_amode)
      real(r8) :: wetdens(pcols,pver,ntot_amode)

      type(physics_buffer_desc), pointer :: pbuf2d(:,:)


      ncol = ncolxx
      ncol_for_outfld = ncol
      ncol_for_qsat = ncol

      write(lun_outfld,'(/a,i5)') 'istep = ', -1

      iulog = 91
      write(*,'(/a)') '*** Hello from MAIN ***'

      write(*,'(/a)') '*** main calling cambox_init_basics'
      call cambox_init_basics( ncol, pbuf2d )

      iulog = 92
      write(*,'(/a)') '*** main calling cambox_init_run'
      call cambox_init_run( &
         ncol, nstop, deltat, t, pmid, pdel, zm, pblh, cld, relhum, qv, &
         q, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens        )

      iulog = 93
      write(*,'(/a)') '*** main calling cambox_do_run'
      call cambox_do_run( &
         ncol, nstop, deltat, t, pmid, pdel, zm, pblh, cld, relhum, qv, &
         q, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens, pbuf2d )

      end subroutine cambox_main


!-------------------------------------------------------------------------------
      subroutine cambox_init_basics( ncol, pbuf2d )

      use chem_mods, only: adv_mass, gas_pcnst, imozart
      use mo_tracname, only: solsym

      use physics_buffer, only: physics_buffer_desc

      use modal_aero_data, only: nbc, npoa, nsoa, nsoag
      use modal_aero_initialize_data
      use modal_aero_amicphys, only: mosaic
      use modal_aero_calcsize, only: modal_aero_calcsize_reg
      use modal_aero_wateruptake, only: modal_aero_wateruptake_reg, modal_aero_wateruptake_init

      integer, intent(in) :: ncol

      type(physics_buffer_desc), pointer :: pbuf2d(:,:)

      integer :: l, l2
      integer :: n


#if ( ( defined MODAL_AERO_7MODE ) && ( defined MOSAIC_SPECIES ) )
      n = 60
#elif ( defined MODAL_AERO_7MODE ) 
      n = 42
#elif ( ( defined MODAL_AERO_4MODE_MOM ) && ( defined RAIN_EVAP_TO_COARSE_AERO ) ) 
      n = 35
#elif ( defined MODAL_AERO_4MODE_MOM ) 
      n = 31
#elif ( defined MODAL_AERO_4MODE ) 
      n = 28
#elif ( defined MODAL_AERO_3MODE ) 
      n = 25
#else
      call endrun( 'MODAL_AERO_3/4/4MOM/7MODE are all undefined' )
#endif
      n = n + 2*(nbc-1) + 2*(npoa-1) + 2*(nsoa-1)
      l = n - (imozart-1)


      write(*,'(/a,3i5 )') 'pcols, pver               =', pcols, pver
      write(*,'( a,3i5/)') 'pcnst, gas_pcnst, imozart =', pcnst, gas_pcnst, imozart
      if (pcnst /= gas_pcnst+imozart-1) call endrun( '*** bad pcnst aa' )
      if (pcnst /= n                  ) call endrun( '*** bad pcnst bb' )


#if ( defined MODAL_AERO_7MODE )
      if (nbc==1 .and. npoa==1 .and. nsoa==1) then

#if ( defined MOSAIC_SPECIES )
      solsym(:l) = &
      (/ 'H2O2    ','H2SO4   ','SO2     ','DMS     ','NH3     ', &
         'SOAG    ','HNO3    ','HCL     ',                       &
         'so4_a1  ','nh4_a1  ','pom_a1  ','soa_a1  ','bc_a1   ', &
         'ncl_a1  ','no3_a1  ','cl_a1   ','num_a1  ',            &
         'so4_a2  ','nh4_a2  ','soa_a2  ','ncl_a2  ','no3_a2  ', &
         'cl_a2   ','num_a2  ',                                  &
         'pom_a3  ','bc_a3   ','num_a3  ',                       &
         'ncl_a4  ','so4_a4  ','nh4_a4  ','no3_a4  ','cl_a4   ', &
         'num_a4  ',                                             &
         'dst_a5  ','so4_a5  ','nh4_a5  ','no3_a5  ','cl_a5   ', &
         'ca_a5   ','co3_a5  ','num_a5  ',                       &
         'ncl_a6  ','so4_a6  ','nh4_a6  ','no3_a6  ','cl_a6   ', &
         'num_a6  ',                                             &
         'dst_a7  ','so4_a7  ','nh4_a7  ','no3_a7  ','cl_a7   ', &
         'ca_a7   ','co3_a7  ','num_a7  '                        /)
      adv_mass(:l) = &
      (/ 34.0135994_r8, 98.0783997_r8, 64.0647964_r8, 62.1324005_r8, 17.0289402_r8, &
         12.0109997_r8, 63.0123400_r8, 36.4601000_r8,                               &
         96.0635986_r8, 18.0363407_r8, 12.0109997_r8, 12.0109997_r8, 12.0109997_r8, &
         22.9897667_r8, 62.0049400_r8, 35.4527000_r8, 1.00740004_r8,                &
         96.0635986_r8, 18.0363407_r8, 12.0109997_r8, 22.9897667_r8, 62.0049400_r8, &
         35.4527000_r8, 1.00740004_r8,                                              &
         12.0109997_r8, 12.0109997_r8, 1.00740004_r8,                               &
         22.9897667_r8, 96.0635986_r8, 18.0363407_r8, 62.0049400_r8, 35.4527000_r8, &
         1.00740004_r8,                                                             &
         135.064041_r8, 96.0635986_r8, 18.0363407_r8, 62.0049400_r8, 35.4527000_r8, &
         40.0780000_r8, 60.0092000_r8, 1.00740004_r8,                               &
         22.9897667_r8, 96.0635986_r8, 18.0363407_r8, 62.0049400_r8, 35.4527000_r8, &
         1.00740004_r8,                                                             &
         135.064041_r8, 96.0635986_r8, 18.0363407_r8, 62.0049400_r8, 35.4527000_r8, &
         40.0780000_r8, 60.0092000_r8, 1.00740004_r8                                /)
! nacl  58.4424667
! cl    35.4527000
! na    22.9897667
! hcl   36.4601000
! hno3  63.0123400
! no3   62.0049400
! ca    40.0780000
! co3   60.0092000


#else
      solsym(:l) = &
      (/ 'H2O2    ','H2SO4   ','SO2     ','DMS     ','NH3     ', &
         'SOAG    ','so4_a1  ','nh4_a1  ','pom_a1  ','soa_a1  ', &
         'bc_a1   ','ncl_a1  ','num_a1  ','so4_a2  ','nh4_a2  ', &
         'soa_a2  ','ncl_a2  ','num_a2  ','pom_a3  ','bc_a3   ', &
         'num_a3  ','ncl_a4  ','so4_a4  ','nh4_a4  ','num_a4  ', &
         'dst_a5  ','so4_a5  ','nh4_a5  ','num_a5  ','ncl_a6  ', &
         'so4_a6  ','nh4_a6  ','num_a6  ','dst_a7  ','so4_a7  ', &
         'nh4_a7  ','num_a7  ' /)
      adv_mass(:l) = &
      (/ 34.0135994_r8, 98.0783997_r8, 64.0647964_r8, 62.1324005_r8, 17.0289402_r8, &
         12.0109997_r8, 96.0635986_r8, 18.0363407_r8, 12.0109997_r8, 12.0109997_r8, &
         12.0109997_r8, 58.4424667_r8, 1.00740004_r8, 96.0635986_r8, 18.0363407_r8, &
         12.0109997_r8, 58.4424667_r8, 1.00740004_r8, 12.0109997_r8, 12.0109997_r8, &
         1.00740004_r8, 58.4424667_r8, 96.0635986_r8, 18.0363407_r8, 1.00740004_r8, &
         135.064041_r8, 96.0635986_r8, 18.0363407_r8, 1.00740004_r8, 58.4424667_r8, &
         96.0635986_r8, 18.0363407_r8, 1.00740004_r8, 135.064041_r8, 96.0635986_r8, &
         18.0363407_r8, 1.00740004_r8 /)

#endif

      else if (nbc==2 .and. npoa==2 .and. nsoa==1) then
      ! nbc=npoa=2 not fully implemented yet
      call endrun( '*** bad nbc and/or npoa and/or nsoa' )

      solsym(:l) = &
      (/ 'H2O2    ','H2SO4   ','SO2     ','DMS     ','NH3     ', &
         'SOAG    ','so4_a1  ','nh4_a1  ','poma_a1 ', &
                                          'pomb_a1 ','soa_a1  ', &
         'bca_a1  ', &
         'bcb_a1  ','ncl_a1  ','num_a1  ','so4_a2  ','nh4_a2  ', &
         'soa_a2  ','ncl_a2  ','num_a2  ','poma_a3 ','pomb_a3 ', &
                                          'bca_a3  ','bcb_a3  ', &
         'num_a3  ','ncl_a4  ','so4_a4  ','nh4_a4  ','num_a4  ', &
         'dst_a5  ','so4_a5  ','nh4_a5  ','num_a5  ','ncl_a6  ', &
         'so4_a6  ','nh4_a6  ','num_a6  ','dst_a7  ','so4_a7  ', &
         'nh4_a7  ','num_a7  ' /)
      adv_mass(:l) = &
      (/ 34.0135994_r8, 98.0783997_r8, 64.0647964_r8, 62.1324005_r8, 17.0289402_r8, &
         12.0109997_r8, 96.0635986_r8, 18.0363407_r8, 12.0109997_r8, 12.0109997_r8, 12.0109997_r8, &
         12.0109997_r8, 12.0109997_r8, 58.4424667_r8, 1.00740004_r8, 96.0635986_r8, 18.0363407_r8, &
         12.0109997_r8, 58.4424667_r8, 1.00740004_r8, 12.0109997_r8,12.0109997_r8,  12.0109997_r8, 12.0109997_r8, &
         1.00740004_r8, 58.4424667_r8, 96.0635986_r8, 18.0363407_r8, 1.00740004_r8, &
         135.064041_r8, 96.0635986_r8, 18.0363407_r8, 1.00740004_r8, 58.4424667_r8, &
         96.0635986_r8, 18.0363407_r8, 1.00740004_r8, 135.064041_r8, 96.0635986_r8, &
         18.0363407_r8, 1.00740004_r8 /)

      else
         call endrun( '*** bad nbc and/or npoa and/or nsoa' )
      end if

#elif ( defined MODAL_AERO_4MODE_MOM )
      if (nbc==1 .and. npoa==1 .and. nsoa==1 .and. nsoag==1) then
#if ( defined RAIN_EVAP_TO_COARSE_AERO )
      solsym(:l) = &
      (/ 'H2O2          ', 'H2SO4         ', 'SO2           ', 'DMS           ', 'SOAG          ', &
         'so4_a1        ', 'pom_a1        ', 'soa_a1        ', 'bc_a1         ', 'dst_a1        ', &
         'ncl_a1        ', 'mom_a1        ', 'num_a1        ', 'so4_a2        ', 'soa_a2        ', &
         'ncl_a2        ', 'mom_a2        ', 'num_a2        ', 'dst_a3        ', 'ncl_a3        ', &
         'so4_a3        ', 'bc_a3         ', 'pom_a3        ', 'soa_a3        ', 'mom_a3        ', &
         'num_a3        ', 'pom_a4        ', 'bc_a4         ', 'mom_a4        ', 'num_a4        ' /)
      adv_mass(:l) = &
      (/     34.013600_r8,     98.078400_r8,     64.064800_r8,     62.132400_r8,     12.011000_r8, &
            115.107340_r8,     12.011000_r8,     12.011000_r8,     12.011000_r8,    135.064039_r8, &
             58.442468_r8, 250092.672000_r8,      1.007400_r8,    115.107340_r8,     12.011000_r8, &
             58.442468_r8, 250092.672000_r8,      1.007400_r8,    135.064039_r8,     58.442468_r8, &
            115.107340_r8,     12.011000_r8,     12.011000_r8,     12.011000_r8, 250092.672000_r8, &
              1.007400_r8,     12.011000_r8,     12.011000_r8, 250092.672000_r8,      1.007400_r8 /)
#else
      solsym(:l) = &
      (/ 'H2O2    ', 'H2SO4   ', 'SO2     ', 'DMS     ',             &
         'SOAG    ', 'so4_a1  ',             'pom_a1  ', 'soa_a1  ', &
         'bc_a1   ', 'ncl_a1  ', 'dst_a1  ', 'mom_a1  ', 'num_a1  ', &
         'so4_a2  ', 'soa_a2  ', 'ncl_a2  ', 'mom_a2  ', 'num_a2  ', &
         'dst_a3  ', 'ncl_a3  ', 'so4_a3  ', 'num_a3  ',             &
         'pom_a4  ', 'bc_a4   ', 'mom_a4  ', 'num_a4  ' /)
      adv_mass(:l) = &
      (/ 34.0135994_r8, 98.0783997_r8, 64.0647964_r8, 62.1324005_r8,                &
         12.0109997_r8, 115.107340_r8,                12.0109997_r8, 12.0109997_r8, &
         12.0109997_r8, 58.4424667_r8, 135.064041_r8, 250092.672_r8, 1.00740004_r8, &
         115.107340_r8, 12.0109997_r8, 58.4424667_r8, 250092.672_r8, 1.00740004_r8, &
         135.064041_r8, 58.4424667_r8, 115.107340_r8, 1.00740004_r8,                &
         12.0109997_r8, 12.0109997_r8, 250092.672_r8, 1.00740004_r8 /)
#endif
      else
         call endrun( '*** bad nbc and/or npoa and/or nsoa' )
      end if


#elif ( defined MODAL_AERO_4MODE )
      if (nbc==1 .and. npoa==1 .and. nsoa==1 .and. nsoag==1) then

      solsym(:l) = &
      (/ 'H2O2    ', 'H2SO4   ', 'SO2     ', 'DMS     ',             &
         'SOAG    ', 'so4_a1  ',             'pom_a1  ', 'soa_a1  ', &
         'bc_a1   ', 'ncl_a1  ', 'dst_a1  ', 'num_a1  ', 'so4_a2  ', &
         'soa_a2  ', 'ncl_a2  ', 'num_a2  ',                         &
         'dst_a3  ', 'ncl_a3  ', 'so4_a3  ', 'num_a3  ',             &
         'pom_a4  ', 'bc_a4   ', 'num_a4  ' /)
      adv_mass(:l) = &
      (/ 34.0135994_r8, 98.0783997_r8, 64.0647964_r8, 62.1324005_r8,                &
         12.0109997_r8, 115.107340_r8,                12.0109997_r8, 12.0109997_r8, &
         12.0109997_r8, 58.4424667_r8, 135.064041_r8, 1.00740004_r8, 115.107340_r8, &
         12.0109997_r8, 58.4424667_r8, 1.00740004_r8,                               &
         135.064041_r8, 58.4424667_r8, 115.107340_r8, 1.00740004_r8,                &
         12.0109997_r8, 12.0109997_r8, 1.00740004_r8 /)
      else
         call endrun( '*** bad nbc and/or npoa and/or nsoa' )
      end if

#else
!if ( defined MODAL_AERO_3MODE )
      if (nbc==1 .and. npoa==1 .and. nsoa==1 .and. nsoag==1) then

      solsym(:l) = &
      (/ 'H2O2    ', 'H2SO4   ', 'SO2     ', 'DMS     ',             &
         'SOAG    ', 'so4_a1  ',             'pom_a1  ', 'soa_a1  ', &
         'bc_a1   ', 'ncl_a1  ', 'dst_a1  ', 'num_a1  ', 'so4_a2  ', &
         'soa_a2  ', 'ncl_a2  ', 'num_a2  ',                         &
         'dst_a3  ', 'ncl_a3  ', 'so4_a3  ', 'num_a3  ' /)
      adv_mass(:l) = &
      (/ 34.0135994_r8, 98.0783997_r8, 64.0647964_r8, 62.1324005_r8,                &
         12.0109997_r8, 115.107340_r8,                12.0109997_r8, 12.0109997_r8, &
         12.0109997_r8, 58.4424667_r8, 135.064041_r8, 1.00740004_r8, 115.107340_r8, &
         12.0109997_r8, 58.4424667_r8, 1.00740004_r8,                               &
         135.064041_r8, 58.4424667_r8, 115.107340_r8, 1.00740004_r8 /)

      else
         call endrun( '*** bad nbc and/or npoa and/or nsoa' )
      end if

#endif

      cnst_name(1) = 'QVAPOR'
      cnst_name(2) = 'CLDLIQ'
      cnst_name(3) = 'CLDICE'
      cnst_name(4) = 'NUMLIQ'
      cnst_name(5) = 'NUMICE'
      cnst_name(imozart:pcnst) = solsym(1:gas_pcnst)

      mosaic = .false.

      write(iulog,'(/a)') &
         'l, l2, cnst_name(l), solsym(l2), adv_mass(l2)'
      do l = 1, pcnst
         if (l < imozart) then
            write(iulog,'(i4,6x,a)') l, cnst_name(l)
         else
            l2 = l - imozart + 1
            if (adv_mass(l2) < 1.0e5_r8) then
               write(iulog,'(2i4,2x,2a,f9.3)') l, l2, cnst_name(l), solsym(l2), adv_mass(l2)
            else
               write(iulog,'(2i4,2x,2a,1pe16.8)') l, l2, cnst_name(l), solsym(l2), adv_mass(l2)
            end if
         end if
      end do


! should be done later
!     write(*,'(/a)') 'cambox_init_basics calling pbuf_initialize'
!     call pbuf_initialize( pbuf2d )

      write(*,'(/a)') 'cambox_init_basics calling modal_aero_register'
      call modal_aero_register( species_class )

      write(*,'(/a)') &
         'cambox_init_basics calling modal_aero_calcsize_reg'
      call modal_aero_calcsize_reg( )

      write(*,'(/a)') &
         'cambox_init_basics calling modal_aero_wateruptake_reg'
      call modal_aero_wateruptake_reg( )

      write(*,'(/a)') &
         'cambox_init_basics calling cambox_pbuf_init pbuf_init_time'
      call cambox_init_pbuf( ncol, pbuf2d )

      write(*,'(/a)') 'cambox_init_basics calling modal_aero_initialize'
      call modal_aero_initialize( pbuf2d, imozart, species_class )

      write(*,'(/a)') &
         'cambox_init_basics calling modal_aero_wateruptake_init'
      call modal_aero_wateruptake_init( pbuf2d )


      write(*,'(/a)') 'cambox_init_basics all done'

      return
      end subroutine cambox_init_basics


!-------------------------------------------------------------------------------
      subroutine cambox_init_pbuf( ncol, pbuf2d )

      use buffer, only: dtype_r8
      use physics_buffer, only: physics_buffer_desc, pbuf_initialize, &
         pbuf_init_time, pbuf_add_field

      integer, intent(in) :: ncol

      type(physics_buffer_desc), pointer :: pbuf2d(:,:)

      integer :: idx


! initialize pbuf time ???
      call pbuf_init_time()

! add pbuf fields that needed but not added by other modal_aero_... routines
      call pbuf_add_field( 'CLD',  'global', dtype_r8, (/pcols, pver/), idx )

! allocate memory for pbuf2d
      call pbuf_initialize( pbuf2d )


      return
      end subroutine cambox_init_pbuf


!-------------------------------------------------------------------------------
      subroutine cambox_init_run( &
         ncol, nstop, deltat, t, pmid, pdel, zm, pblh, cld, relhum, qv, &
         q, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens        )

      use chem_mods, only: adv_mass, gas_pcnst, imozart
      use physconst, only: pi, epsilo, latvap, latice, &
                           rh2o, cpair, tmelt, mwdry, r_universal
      use wv_saturation, only: qsat, gestbl

      use modal_aero_data
      use modal_aero_amicphys, only: &
           gaexch_h2so4_uptake_optaa, newnuc_h2so4_conc_optaa, mosaic, &
           dens_aer, iaer_bc, iaer_pom, iaer_so4, iaer_soa, iaer_ncl, &
           iaer_mom, iaer_dst


      integer,  intent(in   ) :: ncol
      integer,  intent(out  ) :: nstop

      real(r8), intent(out  ) :: deltat
      real(r8), intent(out  ) :: t(pcols,pver)      ! Temperature in Kelvin
      real(r8), intent(out  ) :: pmid(pcols,pver)   ! pressure at model levels (Pa)
      real(r8), intent(out  ) :: pdel(pcols,pver)   ! pressure thickness of levels
      real(r8), intent(out  ) :: zm(pcols,pver)     ! midpoint height above surface (m)
      real(r8), intent(out  ) :: pblh(pcols)        ! pbl height (m)
      real(r8), intent(out  ) :: cld(pcols,pver)    ! stratiform cloud fraction
      real(r8), intent(out  ) :: relhum(pcols,pver) ! layer relative humidity
      real(r8), intent(out  ) :: qv(pcols,pver)     ! layer specific humidity

      real(r8), intent(out  ) :: q(pcols,pver,pcnst)     ! Tracer MR array
      real(r8), intent(out  ) :: qqcw(pcols,pver,pcnst)  ! Cloudborne aerosol MR array
      real(r8), intent(out  ) :: dgncur_a(pcols,pver,ntot_amode)
      real(r8), intent(out  ) :: dgncur_awet(pcols,pver,ntot_amode)
      real(r8), intent(out  ) :: qaerwat(pcols,pver,ntot_amode)
      real(r8), intent(out  ) :: wetdens(pcols,pver,ntot_amode)

      integer :: i
      integer :: k
      integer :: l, ll, loffset, lun
      integer :: l_nh3g, l_so2g, l_soag, l_hno3g, l_hclg, l_h2so4g
      integer :: l_num_a1, l_num_a2, l_nh4_a1, l_nh4_a2, &
                 l_so4_a1, l_so4_a2, l_soa_a1, l_soa_a2
      integer :: l_numa, l_so4a, l_nh4a, l_soaa, l_poma, l_bcxa, l_ncla, &
                 l_dsta, l_no3a, l_clxa, l_caxa, l_co3a, l_moma
      integer :: mode123_empty
      integer :: mopt_aero_loadaa, mopt_aero_loadbb
      integer :: n, nacc, nait

      logical :: ip

      character(len=80) :: tmpch80

      real(r8) :: ev_sat(pcols,pver)
      real(r8) :: qv_sat(pcols,pver)
      real(r8) :: relhum_clea(ncol,pver)
      real(r8) :: zdel(ncol,pver)     ! thickness of levels (m)
      real(r8) :: tmn, tmx, trice
      real(r8) :: tmpa, tmpq
      real(r8) :: tmpfso4, tmpfnh4, tmpfsoa, tmpfpom, &
                  tmpfbcx, tmpfncl, tmpfdst, tmpfmom
      real(r8) :: tmpfno3, tmpfclx, tmpfcax, tmpfco3
      real(r8) :: tmpfmact, tmpfnact 
      real(r8) :: tmpdens, tmpvol, tmpmass, sx
      real(r8) :: aircon(pcols,pver) ! air concentration, kmol/m3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! JS - 06-03-2019: introduce namelist variables for flex control !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! namelist variable
!
      integer  :: mam_dt, mam_nstep
      real(r8) :: temp, press, RH_CLEA
      real(r8) :: numc1, numc2, numc3, numc4,                     &
                  mfso41, mfpom1, mfsoa1, mfbc1, mfdst1, mfncl1,  &
                  mfso42, mfsoa2, mfncl2,                         &
                  mfdst3, mfncl3, mfso43, mfbc3, mfpom3,  mfsoa3, &
                  mfpom4, mfbc4,                                  &
                  qso2, qh2so4, qsoag

      namelist /time_input/ mam_dt, mam_nstep
      namelist /cntl_input/ mdo_gaschem, mdo_gasaerexch, &
                            mdo_rename, mdo_newnuc, mdo_coag
      namelist /met_input/ temp, press, RH_CLEA
      namelist /chem_input/ numc1, numc2, numc3, numc4,          &
                  mfso41, mfpom1, mfsoa1, mfbc1, mfdst1, mfncl1, &
                  mfso42, mfsoa2, mfncl2, &
                  mfdst3, mfncl3, mfso43, mfbc3, mfpom3, mfsoa3, &
                  mfpom4, mfbc4, &
                  qso2, qh2so4, qsoag

      open (UNIT = 101, FILE = 'namelist', STATUS = 'OLD')
          read (101, time_input)
          read (101, cntl_input)
          read (101, met_input)
          read (101, chem_input)
      close (101)

      ! check if mass fraction is larger than one
      if (mfso41+mfpom1+mfsoa1+mfbc1+mfdst1+mfncl1 .gt. 1._r8) then
          print *, "The summed mass fraction is > 1 in mode 1"
          stop
      end if
      if (mfso42+mfsoa2+mfncl2 .gt. 1._r8) then
          print *, "The summed mass fraction is > 1 in mode 2"
          stop
      end if
      if (mfdst3+mfncl3+mfso43+mfbc3+mfpom3+mfsoa3 .gt. 1._r8) then
          print *, "The summed mass fraction is > 1 in mode 3"
          stop
      end if
      if (mfpom4+mfbc4 .gt. 1._r8) then
          print *, "The summed mass fraction is > 1 in mode 4"
          stop
      end if

      iwrite3x_species_flagaa = 1
      iwrite3x_units_flagaa   = 10
      iwrite4x_heading_flagbb = 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! JS - 06-03-2019: do not consider cloud chem at this moment !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!      mdo_gaschem    = 1
      mdo_cloudchem  = 0
!      mdo_gasaerexch = 1
!      mdo_rename     = 1
!      mdo_newnuc     = 1
!      mdo_coag       = 1

      gaexch_h2so4_uptake_optaa =  2  ! 1=sequential prod then loss,  2=prod+loss together
      newnuc_h2so4_conc_optaa   =  2  ! controls treatment of h2so4 concentrationin mam_newnuc_1subcol
                                      !  1 = use avg. value calculated in standard cam5.2.10 and earlier
                                      !  2 = use avg. value calculated in mam_gasaerexch_1subcol
                                      ! 11 = use avg. of initial and final values from mam_gasaerexch_1subcol
                                      ! 12 = use final value from mam_gasaerexch_1subcol
      mopt_h2so4_uptake         = 1   ! *** no longer used

      mopt_ait_size       = 2
      xopt_cloudf         = 0.6_r8
      i_cldy_sameas_clear = 0
      mosaic              = .false.

      !! time step 
      deltat              = mam_dt * 1._r8 
      nstop               = mam_nstep

      pmid(:,:)           = press
      t(:,:)              = temp
      relhum(:,:)         = RH_CLEA
      pblh(:)             = 1.1e3_r8
      zm(:,:)             = 3.0e3_r8
      aircon(:,:)         = pmid(:,:)/(r_universal*t(:,:))

      q                   = 0.0_r8
      qqcw                = 0.0_r8
      dgncur_a            = 0.0_r8
      dgncur_awet         = 0.0_r8
      qaerwat             = 0.0_r8
      wetdens             = 0.0_r8
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! JS - 06-03-2019: set cloud fraction to zero for simplicity !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      cld         = 0.0_r8

! call gestbl to build saturation vapor pressure table.
      tmn   = 173.16_r8
      tmx   = 375.16_r8
      trice =  20.00_r8
      ip    = .true.
      call gestbl(tmn     ,tmx     ,trice   ,ip      ,epsilo  , &
                  latvap  ,latice  ,rh2o    ,cpair   ,tmelt )

!     call aqsat( t, pmid, ev_sat, qv_sat, pcols, ncol, pver, 1, pver )
      call  qsat( t(1:ncol,1:pver), pmid(1:ncol,1:pver), &
                  ev_sat(1:ncol,1:pver), qv_sat(1:ncol,1:pver) )

      qv(1:ncol,1:pver) = relhum(1:ncol,1:pver)*qv_sat(1:ncol,1:pver)
      q(1:ncol,1:pver,1) = qv(1:ncol,1:pver)

! set trace gases
      call cnst_get_ind( 'SOAG',  l_soag,   .false. )
      call cnst_get_ind( 'SO2',   l_so2g,   .false. )
      call cnst_get_ind( 'NH3',   l_nh3g,   .false. )
      call cnst_get_ind( 'HNO3',  l_hno3g,  .false. )
      call cnst_get_ind( 'HCL',   l_hclg,   .false. )
      call cnst_get_ind( 'H2SO4', l_h2so4g, .false. )
      loffset = imozart-1

! initialize the aerosol/number mixing ratio
      do k = 1, pver
         do i = 1, ncol
            do  n = 1, ntot_amode

                sx = log( sigmag_amode(n) )

                if      (n == 1) then
                   dgncur_a(i,k,n) = dgnum_amode(n)  ! 0.20e-6_r8 ! m
                   tmpfsoa      = mfsoa1
                   tmpfso4      = mfso41
                   tmpfncl      = mfncl1
                   tmpfdst      = mfdst1
                   tmpfpom      = mfpom1
                   tmpfbcx      = mfbc1
                   tmpfmom      = 1._r8 - tmpfsoa - tmpfso4 - &
                                  tmpfncl - tmpfdst - tmpfpom - tmpfbcx
                else if (n == 2) then
                   dgncur_a(i,k,n) = dgnum_amode(n)  ! 0.04e-6_r8
                   tmpfsoa      = mfsoa2
                   tmpfso4      = mfso42
                   tmpfncl      = mfncl2
                   tmpfdst      = 0._r8
                   tmpfpom      = 0._r8
                   tmpfbcx      = 0._r8
                   tmpfmom      = 1._r8 - tmpfsoa - tmpfso4 - &
                                  tmpfncl - tmpfdst - tmpfpom - tmpfbcx
                else if (n == 3) then
                   dgncur_a(i,k,n) = dgnum_amode(n)  ! 2.00e-6_r8
                   tmpfsoa      = mfsoa3
                   tmpfso4      = mfso43
                   tmpfncl      = mfncl3
                   tmpfdst      = mfdst3
                   tmpfpom      = mfpom3
                   tmpfbcx      = mfbc3
                   tmpfmom      = 1._r8 - tmpfsoa - tmpfso4 - &
                                  tmpfncl - tmpfdst - tmpfpom - tmpfbcx
                else if (n == 4) then
                   dgncur_a(i,k,n) = dgnum_amode(n)  ! 0.08e-6_r8
                   tmpfsoa      = 0._r8
                   tmpfso4      = 0._r8
                   tmpfncl      = 0._r8
                   tmpfdst      = 0._r8
                   tmpfpom      = mfpom4
                   tmpfbcx      = mfbc4
                   tmpfmom      = 1._r8 - tmpfsoa - tmpfso4 - &
                                  tmpfncl - tmpfdst - tmpfpom - tmpfbcx
                end if
                ! q(i,k,numptr_amode(n)) = #/kg-air
                if (n == modeptr_aitken) then
                   q(i,k,numptr_amode(n)) = numc2 / aircon(i,k) / mwdry
                   l_num_a2 = numptr_amode(n)
                   l_so4_a2 = lptr_so4_a_amode(n)
                else if (n == modeptr_accum) then
                   q(i,k,numptr_amode(n)) = numc1 / aircon(i,k) / mwdry
                   l_num_a1 = numptr_amode(n)
                   l_so4_a1 = lptr_so4_a_amode(n)
                else if (n == modeptr_pcarbon) then
                   q(i,k,numptr_amode(n)) = numc4 / aircon(i,k) / mwdry
                else
                   q(i,k,numptr_amode(n)) = numc3 / aircon(i,k) / mwdry
                end if
      
                ! tmpvol: m3-dry-aerosol/kg-air
                tmpvol  = q(i,k,numptr_amode(n)) * &
                          (dgncur_a(i,k,n)**3) * &
                          (pi/6.0_r8) * exp(4.5_r8*sx*sx)
                tmpdens = 1.0_r8 /                           &
                          ( (tmpfsoa / dens_aer(iaer_soa)) + &
                            (tmpfso4 / dens_aer(iaer_so4)) + &
                            (tmpfbcx / dens_aer(iaer_bc )) + &
                            (tmpfpom / dens_aer(iaer_pom)) + &
                            (tmpfncl / dens_aer(iaer_ncl)) + &
                            (tmpfdst / dens_aer(iaer_dst)) + &
                            (tmpfmom / dens_aer(iaer_mom))   )
                tmpmass = tmpvol*tmpdens   ! kg-dry-aerosol/kg-air

                l_so4a = lptr_so4_a_amode(n)
                l_nh4a = -1
                l_soaa = lptr_soa_a_amode(n)
                l_poma = lptr_pom_a_amode(n)
                if (npoa == 2) l_poma = lptr_poma_a_amode(n)
                l_bcxa = lptr_bc_a_amode(n)
                if (nbc  == 2) l_bcxa = lptr_bca_a_amode(n)
                l_ncla = lptr_nacl_a_amode(n)
                l_dsta = lptr_dust_a_amode(n)
                l_moma = lptr_mom_a_amode(n)
#if ( defined MOSAIC_SPECIES )
                l_no3a = lptr_no3_a_amode(n)
                l_clxa = lptr_cl_a_amode(n)
                l_caxa = lptr_ca_a_amode(n)
                l_co3a = lptr_co3_a_amode(n)
#else
                l_no3a = -1
                l_clxa = -1
                l_caxa = -1
                l_co3a = -1
#endif
                ! q array return kg-aer/kg-air
                if (l_so4a > 0) q(i,k,l_so4a) = tmpmass*tmpfso4
                if (l_nh4a > 0) q(i,k,l_nh4a) = tmpmass*tmpfnh4
                if (l_soaa > 0) q(i,k,l_soaa) = tmpmass*tmpfsoa
                if (l_poma > 0) q(i,k,l_poma) = tmpmass*tmpfpom
                if (l_bcxa > 0) q(i,k,l_bcxa) = tmpmass*tmpfbcx
                if (l_dsta > 0) q(i,k,l_dsta) = tmpmass*tmpfdst
                if (l_ncla > 0) q(i,k,l_ncla) = tmpmass*tmpfncl
                if (l_moma > 0) q(i,k,l_moma) = tmpmass*tmpfmom
                if (l_no3a > 0) q(i,k,l_no3a) = tmpmass*tmpfno3
                if (l_clxa > 0) q(i,k,l_clxa) = tmpmass*tmpfclx
                if (l_caxa > 0) q(i,k,l_caxa) = tmpmass*tmpfcax
                if (l_co3a > 0) q(i,k,l_co3a) = tmpmass*tmpfco3

            end do ! n
         end do ! i
      end do ! k             

! initialize the gas mixing ratio
      q(:,:,l_so2g)   = qso2
      q(:,:,l_soag)   = qsoag
      q(:,:,l_h2so4g) = qh2so4

      write(*,'(/a)') 'cambox_init_run all done'

      return
      end subroutine cambox_init_run


!-------------------------------------------------------------------------------
      subroutine cambox_do_run( &
         ncol, nstop, deltat, t, pmid, pdel, zm, pblh, cld, relhum, qv, &
         q, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens, pbuf2d )

      use chem_mods, only: adv_mass, gas_pcnst, imozart
      use physconst, only: mwdry
      use physics_types, only: physics_state, physics_ptend
      use physics_buffer, only: physics_buffer_desc, pbuf_get_chunk

      use modal_aero_data
      use modal_aero_calcsize, only: modal_aero_calcsize_sub
      use modal_aero_amicphys, only: modal_aero_amicphys_intr, &
          gaexch_h2so4_uptake_optaa, newnuc_h2so4_conc_optaa, mosaic
      use modal_aero_wateruptake, only: modal_aero_wateruptake_dr
      use gaschem_simple, only: gaschem_simple_sub
      use cloudchem_simple, only: cloudchem_simple_sub

      integer,  intent(in   ) :: ncol
      integer,  intent(in   ) :: nstop

      real(r8), intent(in   ) :: deltat
      real(r8), intent(in   ) :: t(pcols,pver)      ! Temperature in Kelvin
      real(r8), intent(in   ) :: pmid(pcols,pver)   ! pressure at model levels (Pa)
      real(r8), intent(in   ) :: pdel(pcols,pver)   ! pressure thickness of levels
      real(r8), intent(in   ) :: zm(pcols,pver)     ! midpoint height above surface (m)
      real(r8), intent(in   ) :: pblh(pcols)        ! pbl height (m)
      real(r8), intent(in   ) :: cld(pcols,pver)    ! stratiform cloud fraction
      real(r8), intent(in   ) :: relhum(pcols,pver) ! layer relative humidity
      real(r8), intent(in   ) :: qv(pcols,pver)     ! layer specific humidity

      real(r8), intent(inout) :: q(pcols,pver,pcnst)     ! Tracer MR array
      real(r8), intent(inout) :: qqcw(pcols,pver,pcnst)  ! Cloudborne aerosol MR array
      real(r8), intent(inout) :: dgncur_a(pcols,pver,ntot_amode)
      real(r8), intent(inout) :: dgncur_awet(pcols,pver,ntot_amode)
      real(r8), intent(inout) :: qaerwat(pcols,pver,ntot_amode)
      real(r8), intent(inout) :: wetdens(pcols,pver,ntot_amode)

      type(physics_buffer_desc), pointer :: pbuf2d(:,:)  ! full physics buffer

      integer, parameter :: nqtendbb = 4
      integer, parameter :: iqtend_cond = 1
      integer, parameter :: iqtend_rnam = 2
      integer, parameter :: iqtend_nnuc = 3
      integer, parameter :: iqtend_coag = 4
      integer, parameter :: nqqcwtendbb = 1
      integer, parameter :: iqqcwtend_rnam = 1

      integer :: i, icalcaer_flag, iwaterup_flag
      integer :: istep
      integer :: itmpa, itmpb
      integer :: k
      integer :: l, l2, ll
      integer :: l_h2so4g, l_nh3g, l_so2g, l_hno3g, l_hclg, l_soag
      integer :: l_num_a1, l_nh4_a1, l_so4_a1
      integer :: l_num_a2, l_nh4_a2, l_so4_a2
      integer :: lmz_h2so4g, lmz_nh3g, lmz_so2g, &
                 lmz_hno3g, lmz_hclg, lmz_soag
      integer :: lmz_num_a1, lmz_nh4_a1, lmz_so4_a1
      integer :: lmz_num_a2, lmz_nh4_a2, lmz_so4_a2
      integer :: lchnk, loffset, lun
      integer :: latndx(pcols), lonndx(pcols)
      integer :: n, nacc, nait, nstep

      logical :: aero_mmr_flag
      logical :: h2o_mmr_flag
      logical :: dotend(pcnst)

      character(len=80) :: tmpch80

      real(r8) :: cld_ncol(ncol,pver)
      real(r8) :: del_h2so4_aeruptk(ncol,pver)
      real(r8) :: del_h2so4_gasprod(ncol,pver)
      real(r8) :: dqdt(pcols,pver,pcnst)        ! Tracer MR tendency array
      real(r8) :: dvmrdt_bb(ncol,pver,gas_pcnst,nqtendbb)   ! mixing ratio changes
      real(r8) :: dvmrcwdt_bb(ncol,pver,gas_pcnst,nqqcwtendbb) ! mixing ratio changes
      real(r8) :: dvmrdt_cond(ncol,pver,gas_pcnst)   ! mixing ratio changes from renaming 
      real(r8) :: dvmrcwdt_cond(ncol,pver,gas_pcnst) ! mixing ratio changes from renaming 
      real(r8) :: dvmrdt_nnuc(ncol,pver,gas_pcnst)   ! mixing ratio changes from renaming 
      real(r8) :: dvmrcwdt_nnuc(ncol,pver,gas_pcnst) ! mixing ratio changes from renaming 
      real(r8) :: dvmrdt_coag(ncol,pver,gas_pcnst)   ! mixing ratio changes from renaming 
      real(r8) :: dvmrcwdt_coag(ncol,pver,gas_pcnst) ! mixing ratio changes from renaming 
      real(r8) :: dvmrdt_rnam(ncol,pver,gas_pcnst)   ! mixing ratio changes from renaming 
      real(r8) :: dvmrcwdt_rnam(ncol,pver,gas_pcnst) ! mixing ratio changes from renaming 
      real(r8) :: h2so4_pre_gaschem(ncol,pver) ! grid-avg h2so4(g) mix ratio before gas chem (mol/mol)
      real(r8) :: h2so4_aft_gaschem(ncol,pver) ! grid-avg h2so4(g) mix ratio after  gas chem (mol/mol)
      real(r8) :: h2so4_clear_avg(  ncol,pver) ! average clear sub-area h2so4(g) mix ratio (mol/mol)
      real(r8) :: h2so4_clear_fin(  ncol,pver) ! final   clear sub-area h2so4(g) mix ratio (mol/mol)
      real(r8) :: mmr(ncol,pver,gas_pcnst)     ! gas & aerosol mass   mixing ratios
      real(r8) :: mmrcw(ncol,pver,gas_pcnst)   ! gas & aerosol mass   mixing ratios
      real(r8) :: tau_gaschem_simple(ncol,pver)
      real(r8) :: tmpa, tmpb, tmpc
      real(r8) :: tmpveca(999)
      real(r8) :: told, tnew
      real(r8) :: uptkrate_h2so4(   ncol,pver) ! h2so4(g) uptake (by aerosols) rate (1/s)
      real(r8) :: vmr(ncol,pver,gas_pcnst)     ! gas & aerosol volume mixing ratios
      real(r8) :: vmr_svaa(ncol,pver,gas_pcnst)
      real(r8) :: vmr_svbb(ncol,pver,gas_pcnst)
      real(r8) :: vmr_svcc(ncol,pver,gas_pcnst)
      real(r8) :: vmr_svdd(ncol,pver,gas_pcnst)
      real(r8) :: vmr_svee(ncol,pver,gas_pcnst)
      real(r8) :: vmrcw(ncol,pver,gas_pcnst)   ! gas & aerosol volume mixing ratios
      real(r8) :: vmrcw_svaa(ncol,pver,gas_pcnst)
      real(r8) :: vmrcw_svbb(ncol,pver,gas_pcnst)
      real(r8) :: vmrcw_svcc(ncol,pver,gas_pcnst)
      real(r8) :: vmrcw_svdd(ncol,pver,gas_pcnst)
      real(r8) :: vmrcw_svee(ncol,pver,gas_pcnst)

!     type(physics_state), target, intent(in)    :: state       ! Physics state variables
!     type(physics_ptend), target, intent(inout) :: ptend       ! indivdual parameterization tendencies
      type(physics_state)                        :: state       ! Physics state variables
      type(physics_ptend)                        :: ptend       ! indivdual parameterization tendencies
      type(physics_buffer_desc), pointer         :: pbuf(:)     ! physics buffer for a chunk

!
! for netcdf file
!
      character (len = *), parameter :: FILE_NAME = "mam_output.nc"
      integer :: ncid, nstep_dimid, mode_dimid
      integer :: error
      integer :: dimids(2), varid(23)
      character (8) :: date
      real(r8), dimension(nstop,ntot_amode) :: tmp_dgn_a, &
                               tmp_dgn_awet, tmp_num_aer, &
                               tmp_so4_aer, tmp_soa_aer
      real(r8), dimension(nstop)            :: tmp_h2so4, &
                                               tmp_soag
      real(r8), dimension(nstop,ntot_amode) :: qtend_cond_aging_so4, &
                                               qtend_cond_aging_soa, &
                                               qtend_rename_so4, &
                                               qtend_rename_soa, &
                                               qtend_newnuc_so4, &
                                               qtend_newnuc_soa, &
                                               qtend_coag_so4, &
                                               qtend_coag_soa
      real(r8), dimension(nstop)            :: qtend_cond_aging_h2so4, &
                                               qtend_cond_aging_soag,  &
                                               qtend_rename_h2so4,     &
                                               qtend_rename_soag,      &
                                               qtend_newnuc_h2so4,     &
                                               qtend_newnuc_soag,      &
                                               qtend_coag_h2so4,       &
                                               qtend_coag_soag

!
! output comparison results
!
      ! Create the netCDF file. The nf90_clobber parameter tells 
      ! netCDF to overwrite this file, if it already exists.
      call check( nf90_create(FILE_NAME, NF90_CLOBBER, ncid) )

      ! Define the dimensions. NetCDF will hand back an ID for each. 
      call check( nf90_def_dim(ncid, "nsteps", nstop, nstep_dimid) )
      call check( nf90_def_dim(ncid, "mode", ntot_amode, mode_dimid) )

      ! The dimids array is used to pass the IDs of the dimensions of
      ! the variables.
      dimids  = (/nstep_dimid, mode_dimid/)

      ! Define the variable.
      call check( nf90_def_var(ncid, "num_aer", &
                  NF90_DOUBLE, dimids, varid(1)) )
      call check( nf90_def_var(ncid, "so4_aer", &
                  NF90_DOUBLE, dimids, varid(2)) )
      call check( nf90_def_var(ncid, "soa_aer", &
                  NF90_DOUBLE, dimids, varid(3)) )
      call check( nf90_def_var(ncid, "h2so4_gas", &
                  NF90_DOUBLE, nstep_dimid, varid(4)) )
      call check( nf90_def_var(ncid, "soag_gas", &
                  NF90_DOUBLE, nstep_dimid, varid(5)) )
      call check( nf90_def_var(ncid, "dgn_a", &
                  NF90_DOUBLE, dimids, varid(6)) )
      call check( nf90_def_var(ncid, "dgn_awet", &
                  NF90_DOUBLE, dimids, varid(7)) )
      call check( nf90_def_var(ncid, "qtend_cond_aging_so4", &
                  NF90_DOUBLE, dimids, varid(8)) )
      call check( nf90_def_var(ncid, "qtend_rename_so4", &
                  NF90_DOUBLE, dimids, varid(9)) )
      call check( nf90_def_var(ncid, "qtend_newnuc_so4", &
                  NF90_DOUBLE, dimids, varid(10)) )
      call check( nf90_def_var(ncid, "qtend_coag_so4", &
                  NF90_DOUBLE, dimids, varid(11)) )
      call check( nf90_def_var(ncid, "qtend_cond_aging_soa", &
                  NF90_DOUBLE, dimids, varid(12)) )
      call check( nf90_def_var(ncid, "qtend_rename_soa", &
                  NF90_DOUBLE, dimids, varid(13)) )
      call check( nf90_def_var(ncid, "qtend_newnuc_soa", &
                  NF90_DOUBLE, dimids, varid(14)) )
      call check( nf90_def_var(ncid, "qtend_coag_soa", &
                  NF90_DOUBLE, dimids, varid(15)) )
      call check( nf90_def_var(ncid, "qtend_cond_aging_h2so4", &
                  NF90_DOUBLE, nstep_dimid, varid(16)) )
      call check( nf90_def_var(ncid, "qtend_rename_h2so4", &
                  NF90_DOUBLE, nstep_dimid, varid(17)) )
      call check( nf90_def_var(ncid, "qtend_newnuc_h2so4", &
                  NF90_DOUBLE, nstep_dimid, varid(18)) )
      call check( nf90_def_var(ncid, "qtend_coag_h2so4", &
                  NF90_DOUBLE, nstep_dimid, varid(19)) )
      call check( nf90_def_var(ncid, "qtend_cond_aging_soag", &
                  NF90_DOUBLE, nstep_dimid, varid(20)) )
      call check( nf90_def_var(ncid, "qtend_rename_soag", &
                  NF90_DOUBLE, nstep_dimid, varid(21)) )
      call check( nf90_def_var(ncid, "qtend_newnuc_soag", &
                  NF90_DOUBLE, nstep_dimid, varid(22)) )
      call check( nf90_def_var(ncid, "qtend_coag_soag", &
                  NF90_DOUBLE, nstep_dimid, varid(23)) )

      ! Assign units attributes to coordinate var data. 
      ! This attaches a text attribute to each of the 
      ! coordinate variables, containing the units.
      call check( nf90_put_att(ncid, varid(1), "units", "#/kg-air") )
      call check( nf90_put_att(ncid, varid(2), "units", "kg-aer/kg-air") )
      call check( nf90_put_att(ncid, varid(3), "units", "kg-aer/kg-air") )
      call check( nf90_put_att(ncid, varid(4), "units", "kg-gas/kg-air") )
      call check( nf90_put_att(ncid, varid(5), "units", "kg-gas/kg-air") )
      call check( nf90_put_att(ncid, varid(6), "units", "meter") )
      call check( nf90_put_att(ncid, varid(7), "units", "meter") )
      call check( nf90_put_att(ncid, varid(8), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(8), "descr", &
                                          "condensation-aging tendency") )
      call check( nf90_put_att(ncid, varid(9), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(9), "descr", &
                                          "rename tendency") )
      call check( nf90_put_att(ncid, varid(10), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(10), "descr", &
                                          "nucleation tendency") )
      call check( nf90_put_att(ncid, varid(11), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(11), "descr", &
                                          "coagulation tendency") )
      call check( nf90_put_att(ncid, varid(12), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(12), "descr", &
                                          "condensation-aging tendency") )
      call check( nf90_put_att(ncid, varid(13), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(13), "descr", &
                                          "rename tendency") )
      call check( nf90_put_att(ncid, varid(14), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(14), "descr", &
                                          "nucleation tendency") )
      call check( nf90_put_att(ncid, varid(15), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(15), "descr", &
                                          "coagulation tendency") )
      call check( nf90_put_att(ncid, varid(16), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(16), "descr", &
                                          "condensation-aging tendency") )
      call check( nf90_put_att(ncid, varid(17), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(17), "descr", &
                                          "rename tendency") )
      call check( nf90_put_att(ncid, varid(18), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(18), "descr", &
                                          "nucleation tendency") )
      call check( nf90_put_att(ncid, varid(19), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(19), "descr", &
                                          "coagulation tendency") )
      call check( nf90_put_att(ncid, varid(20), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(20), "descr", &
                                          "condensation-aging tendency") )
      call check( nf90_put_att(ncid, varid(21), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(21), "descr", &
                                          "rename tendency") )
      call check( nf90_put_att(ncid, varid(22), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(22), "descr", &
                                          "nucleation tendency") )
      call check( nf90_put_att(ncid, varid(23), "units", "mol mol-1 s-1") )
      call check( nf90_put_att(ncid, varid(23), "descr", &
                                          "coagulation tendency") )

      ! Add global attribute
      call check( nf90_put_att(ncid, NF90_GLOBAL, &
                               "Created_by", "PNNL") )
      call date_and_time(date)
      call check( nf90_put_att(ncid, NF90_GLOBAL, &
                               "Created_date", date) )

      ! End define mode. This tells netCDF we are done defining
      ! metadata.
      call check( nf90_enddef(ncid) )

      tmp_dgn_a              = 0._r8 ; tmp_dgn_awet          = 0._r8
      tmp_num_aer            = 0._r8 ; tmp_so4_aer           = 0._r8
      tmp_soa_aer            = 0._r8 ; tmp_h2so4             = 0._r8
      tmp_soag               = 0._r8
      qtend_cond_aging_so4   = 0._r8 ; qtend_cond_aging_soa  = 0._r8
      qtend_rename_so4       = 0._r8 ; qtend_rename_soa      = 0._r8
      qtend_newnuc_so4       = 0._r8 ; qtend_newnuc_soa      = 0._r8
      qtend_coag_so4         = 0._r8 ; qtend_coag_soa        = 0._r8
      qtend_cond_aging_h2so4 = 0._r8 ; qtend_cond_aging_soag = 0._r8
      qtend_rename_h2so4     = 0._r8 ; qtend_rename_soag     = 0._r8
      qtend_newnuc_h2so4     = 0._r8 ; qtend_newnuc_soag     = 0._r8
      qtend_coag_h2so4       = 0._r8 ; qtend_coag_soag       = 0._r8

      lchnk = begchunk
      pbuf => pbuf_get_chunk( pbuf2d, lchnk)

      cld_ncol(1:ncol,:) = cld(1:ncol,:)

      latndx = -1
      lonndx = -1

      call cnst_get_ind( 'H2SO4', l_h2so4g, .false. )
      call cnst_get_ind( 'SO2',   l_so2g,   .false. )
      call cnst_get_ind( 'NH3',   l_nh3g,   .false. )
      call cnst_get_ind( 'HNO3',  l_hno3g,  .false. )
      call cnst_get_ind( 'HCL',   l_hclg,   .false. )
      call cnst_get_ind( 'SOAG',  l_soag,   .false. )

      nacc = modeptr_accum
      l_num_a1 = numptr_amode(nacc)
      l_so4_a1 = lptr_so4_a_amode(nacc)
      l_nh4_a1 = lptr_nh4_a_amode(nacc)

      nait = modeptr_aitken
      l_num_a2 = numptr_amode(nait)
      l_so4_a2 = lptr_so4_a_amode(nait)
      l_nh4_a2 = lptr_nh4_a_amode(nait)

      lmz_h2so4g = l_h2so4g - (imozart-1)
      lmz_so2g   = l_so2g   - (imozart-1)
      lmz_nh3g   = l_nh3g   - (imozart-1)
      lmz_hno3g  = l_hno3g  - (imozart-1)
      lmz_hclg   = l_hclg   - (imozart-1)
      lmz_soag   = l_soag   - (imozart-1)

      lmz_num_a1 = l_num_a1 - (imozart-1)
      lmz_so4_a1 = l_so4_a1 - (imozart-1)
      lmz_nh4_a1 = l_nh4_a1 - (imozart-1)

      lmz_num_a2 = l_num_a2 - (imozart-1)
      lmz_so4_a2 = l_so4_a2 - (imozart-1)
      lmz_nh4_a2 = l_nh4_a2 - (imozart-1)

      write(*,'(/a,3i5)') 'l_h2so4g, l_so2g,   l_nh3g  ', l_h2so4g, l_so2g,   max(l_nh3g,-999)
      write(*,'( a,3i5)') 'l_num_a1, l_so4_a1, l_nh4_a1', l_num_a1, l_so4_a1, max(l_nh4_a1,-999)
      write(*,'( a,3i5)') 'l_num_a2, l_so4_a2, l_nh4_a2', l_num_a2, l_so4_a2, max(l_nh4_a2,-999)


main_time_loop: &
      do nstep = 1, nstop
      istep = nstep
      if (nstep == 1) tnew = 0.0_r8
      told = tnew
      tnew = told + deltat

      write(lun_outfld,'(/a,i5,2f10.3)') 'istep, told, tnew (h) = ', &
         istep, told/3600.0_r8, tnew/3600.0_r8


!
! calcsize
!
      lun = 6
      write(lun,'(/a,i8)') 'cambox_do_run doing calcsize, istep=', istep
      loffset = 0
      icalcaer_flag = 1
      aero_mmr_flag = .true.

      dotend = .false.
      dqdt = 0.0_r8

! *** new calcsize interface ***
! load state
      state%lchnk = lchnk
      state%ncol  = ncol
      state%t     = t
      state%pmid  = pmid
      state%pdel  = pdel
      state%q     = q
! load ptend
      ptend%lq    = dotend
      ptend%q     = dqdt
! load pbuf
      call load_pbuf( pbuf, lchnk, ncol, &
         cld, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens )

! call calcsize
      call modal_aero_calcsize_sub( state, ptend, deltat, pbuf, &
         do_adjust_in=.true., do_aitacc_transfer_in=.true. )

! unload ptend
      dotend = ptend%lq
      dqdt   = ptend%q
! unload pbuf
      call unload_pbuf( pbuf, lchnk, ncol, &
         cld, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens )
      
! apply tendencies
      itmpb = 0
      do l = 1, pcnst
         itmpa = 0
         if ( .not. dotend(l) ) cycle
         do k = 1, pver
         do i = 1, ncol
            if (abs(dqdt(i,k,l)) > 1.0e-30_r8) then
!              write(lun,'(2a,2i4,1p,2e10.2)') &
!                 'calcsize tend > 0   ', cnst_name(l), i, k, &
!                 q(i,k,l), dqdt(i,k,l)*deltat
               itmpa = itmpa + 1
            end if
            q(i,k,l) = q(i,k,l) + dqdt(i,k,l)*deltat
            q(i,k,l) = max( q(i,k,l), 0.0_r8 )
         end do
         end do
         if (itmpa > 0) then
            write(lun,'(2a,i7)') &
               'calcsize tend > 0   ', cnst_name(l), itmpa
            itmpb = itmpb + 1
         end if
      end do
      if (itmpb > 0) then
         write(lun,'(a,i7)') 'calcsize tend > 0 for nspecies =', itmpb
      else
         write(lun,'(a,i7)') 'calcsize tend = 0 for all species'
      end if

      do i = 1, ncol
      lun = 29 + i
      write(lun,'(/a,i8)') 'cambox_do_run doing calcsize, istep=', istep
      if (itmpb > 0) then
         write(lun,'(a,i7)') 'calcsize tend > 0 for nspecies =', itmpb
      else
         write(lun,'(a,i7)') 'calcsize tend = 0 for all species'
      end if
      if (iwrite3x_units_flagaa >= 10) then
         tmpch80 = '  (#/mg,  nmol/mol,  nm)'
         tmpa = 1.0e9*mwdry/adv_mass(lmz_so4_a1)
      else
         tmpch80 = '  (#/mg,  ug/kg,  nm)'
         tmpa = 1.0e9
      end if
      write(lun,'( 2a)') &
         'k, accum num, so4, dgncur_a, same for aitken', trim(tmpch80)
      do k = 1, pver
      write(lun,'( i4,1p,4(2x,3e12.4))') k, &
         q(i,k,l_num_a1)*1.0e-6, q(i,k,l_so4_a1)*tmpa, dgncur_a(i,k,nacc)*1.0e9, &
         q(i,k,l_num_a2)*1.0e-6, q(i,k,l_so4_a2)*tmpa, dgncur_a(i,k,nait)*1.0e9
      end do
      end do ! i


!
! watruptake
!
      lun = 6
      write(lun,'(/a,i8)') 'cambox_do_run doing wateruptake, istep=', istep
      loffset = 0
      iwaterup_flag = 1
      aero_mmr_flag = .true.
      h2o_mmr_flag = .true.

      dotend = .false.
      dqdt = 0.0_r8

! *** new wateruptake interface ***
! load state
      state%lchnk = lchnk
      state%ncol  = ncol
      state%t     = t
      state%pmid  = pmid
      state%pdel  = pdel
      state%q     = q
! load pbuf
      call load_pbuf( pbuf, lchnk, ncol, &
         cld, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens )

! call wateruptake
      call modal_aero_wateruptake_dr( state, pbuf )

! unload pbuf
      call unload_pbuf( pbuf, lchnk, ncol, &
         cld, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens )
      
!
! switch from q & qqcw to vmr and vmrcw
!
      loffset = imozart - 1
      mmr = 0.0_r8
      mmrcw = 0.0_r8
      vmr = 0.0_r8
      vmrcw = 0.0_r8
      do l = imozart, pcnst
         l2 = l - loffset
         mmr(  1:ncol,1:pver,l2) = q(  1:ncol,1:pver,l)
         mmrcw(1:ncol,1:pver,l2) = qqcw(1:ncol,1:pver,l)
         vmr(  1:ncol,1:pver,l2) = mmr(  1:ncol,1:pver,l2)*mwdry/adv_mass(l2)
         vmrcw(1:ncol,1:pver,l2) = mmrcw(1:ncol,1:pver,l2)*mwdry/adv_mass(l2)
      end do

!
! gaschem_simple
!
      lun = 6
      write(lun,'(/a,i8)') 'cambox_do_run doing gaschem simple, istep=', istep
      vmr_svaa   = vmr
      vmrcw_svaa = vmrcw
      h2so4_pre_gaschem(1:ncol,:) = vmr(1:ncol,:,lmz_h2so4g)

! global avg ~= 13 d = 1.12e6 s, daytime avg ~= 5.6e5, noontime peak ~= 3.7e5
      tau_gaschem_simple = 3.0e5  ! so2 gas-rxn timescale (s)

      if (mdo_gaschem > 0) then
         call gaschem_simple_sub(                       &
            lchnk,    ncol,     nstep,               &
            loffset,  deltat,                        &
            vmr,                tau_gaschem_simple      )
      else
         ! assumed constant gas chemistry production rate (mol/mol)
         vmr(1:ncol,:,lmz_h2so4g) = vmr(1:ncol,:,lmz_h2so4g) + 1.e-16_r8*deltat
      end if

      h2so4_aft_gaschem(1:ncol,:) = vmr(1:ncol,:,lmz_h2so4g)

!
! cloudchem_simple
!
      lun = 6
      write(lun,'(/a,i8)') &
         'cambox_do_run doing cloudchem simple, istep=', istep
      vmr_svbb = vmr
      vmrcw_svbb = vmrcw

      if (mdo_cloudchem > 0 .and. maxval( cld_ncol(:,:) ) > 1.0e-6_r8) then

      call cloudchem_simple_sub(                  &
         lchnk,    ncol,     nstep,               &
         loffset,  deltat,                        &
         vmr,      vmrcw,    cld_ncol             )

      end if ! (mdo_cloudchem > 0 .and. maxval( cld_ncol(:,:) ) > 1.0e-6_r8) then


!
! gasaerexch
!
      lun = 6
      write(lun,'(/a,i8)') 'cambox_do_run doing gasaerexch, istep=', istep
      vmr_svcc = vmr
      vmrcw_svcc = vmrcw

      dvmrdt_bb = 0.0_r8 ; dvmrcwdt_bb = 0.0_r8

      call modal_aero_amicphys_intr(              &
         mdo_gasaerexch,     mdo_rename,          &
         mdo_newnuc,         mdo_coag,            &
         lchnk,    ncol,     nstep,               &
         loffset,  deltat,                        &
         latndx,   lonndx,                        &
         t,        pmid,     pdel,                &
         zm,       pblh,                          &
         qv,       cld_ncol,                      &
         vmr,                vmrcw,               &   ! after  cloud chem
         vmr_svaa,                                &   ! before gas chem
         vmr_svbb,           vmrcw_svbb,          &   ! before cloud chem
         nqtendbb,           nqqcwtendbb,         &
         dvmrdt_bb,          dvmrcwdt_bb,         &
         dgncur_a,           dgncur_awet,         &
         wetdens,            qaerwat              )

      dvmrdt_cond(  :,:,:) = dvmrdt_bb(  :,:,:,iqtend_cond)
      dvmrdt_rnam(  :,:,:) = dvmrdt_bb(  :,:,:,iqtend_rnam)
      dvmrdt_nnuc(  :,:,:) = dvmrdt_bb(  :,:,:,iqtend_nnuc)
      dvmrdt_coag(  :,:,:) = dvmrdt_bb(  :,:,:,iqtend_coag)
      dvmrcwdt_cond(:,:,:) = 0.0_r8
      dvmrcwdt_rnam(:,:,:) = dvmrcwdt_bb(:,:,:,iqqcwtend_rnam)
      dvmrcwdt_nnuc(:,:,:) = 0.0_r8
      dvmrcwdt_coag(:,:,:) = 0.0_r8

!
! done
!
      lun = 6
      write(lun,'(/a,i8)') 'cambox_do_run step done, istep=', istep

!
! switch from vmr & vmrcw to q & qqcw
!
      loffset = imozart - 1
      do l = imozart, pcnst
         l2 = l - loffset
         mmr(  1:ncol,1:pver,l2) = vmr(  1:ncol,1:pver,l2) * adv_mass(l2)/mwdry
         mmrcw(1:ncol,1:pver,l2) = vmrcw(1:ncol,1:pver,l2) * adv_mass(l2)/mwdry
         q(    1:ncol,1:pver,l)  = mmr(  1:ncol,1:pver,l2)
         qqcw( 1:ncol,1:pver,l)  = mmrcw(1:ncol,1:pver,l2)
      end do

!
! store the data of each time step for netcdf output
!
      tmp_dgn_a(nstep,1:ntot_amode)        = dgncur_a(1,1,1:ntot_amode)
      tmp_dgn_awet(nstep,1:ntot_amode)     = dgncur_awet(1,1,1:ntot_amode)
      do i = 1, ntot_amode
         tmp_num_aer(nstep,i)              = q(1,1,numptr_amode(i))
         l                                 = lptr_so4_a_amode(i)
         if  (l .gt. 0) then 
             tmp_so4_aer(nstep,i)          = q(1,1,l)
             l2                            = l - loffset
             qtend_cond_aging_so4(nstep,i) = dvmrdt_cond(1,1,l2)
             qtend_rename_so4(nstep,i)     = dvmrdt_rnam(1,1,l2)
             qtend_newnuc_so4(nstep,i)     = dvmrdt_nnuc(1,1,l2)
             qtend_coag_so4(nstep,i)       = dvmrdt_coag(1,1,l2)
         end if
         l                                 = lptr_soa_a_amode(i)
         if  (l .gt. 0) then 
             tmp_soa_aer(nstep,i)          = q(1,1,l)
             l2                            = l - loffset
             qtend_cond_aging_soa(nstep,i) = dvmrdt_cond(1,1,l2)
             qtend_rename_soa(nstep,i)     = dvmrdt_rnam(1,1,l2)
             qtend_newnuc_soa(nstep,i)     = dvmrdt_nnuc(1,1,l2)
             qtend_coag_soa(nstep,i)       = dvmrdt_coag(1,1,l2)
         end if
      end do
      tmp_h2so4(nstep)                     = q(1,1,l_h2so4g)
      l2                                   = l_h2so4g - loffset
      qtend_cond_aging_h2so4(nstep)        = dvmrdt_cond(1,1,l2)
      qtend_rename_h2so4(nstep)            = dvmrdt_rnam(1,1,l2)
      qtend_newnuc_h2so4(nstep)            = dvmrdt_nnuc(1,1,l2)
      qtend_coag_h2so4(nstep)              = dvmrdt_coag(1,1,l2)

      tmp_soag(nstep)                      = q(1,1,l_soag)
      l2                                   = l_soag - loffset
      qtend_cond_aging_soag(nstep)         = dvmrdt_cond(1,1,l2)
      qtend_rename_soag(nstep)             = dvmrdt_rnam(1,1,l2)
      qtend_newnuc_soag(nstep)             = dvmrdt_nnuc(1,1,l2)
      qtend_coag_soag(nstep)               = dvmrdt_coag(1,1,l2)

      end do main_time_loop

!
! Write the data to the file.
!
      call check( nf90_put_var(ncid, varid(1), &
                  tmp_num_aer(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(2), &
                  tmp_so4_aer(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(3), &
                  tmp_soa_aer(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(4), &
                  tmp_h2so4(1:nstop)) )
      call check( nf90_put_var(ncid, varid(5), &
                  tmp_soag(1:nstop))  )
      call check( nf90_put_var(ncid, varid(6), &
                  tmp_dgn_a(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(7), &
                  tmp_dgn_awet(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(8), &
                  qtend_cond_aging_so4(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(9), &
                  qtend_rename_so4(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(10), &
                  qtend_newnuc_so4(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(11), &
                  qtend_coag_so4(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(12), &
                  qtend_cond_aging_soa(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(13), &
                  qtend_rename_soa(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(14), &
                  qtend_newnuc_soa(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(15), &
                  qtend_coag_soa(1:nstop,1:ntot_amode)) )
      call check( nf90_put_var(ncid, varid(16), &
                  qtend_cond_aging_h2so4(1:nstop)) )
      call check( nf90_put_var(ncid, varid(17), &
                  qtend_rename_h2so4(1:nstop)) )
      call check( nf90_put_var(ncid, varid(18), &
                  qtend_newnuc_h2so4(1:nstop)) )
      call check( nf90_put_var(ncid, varid(19), &
                  qtend_coag_h2so4(1:nstop)) )
      call check( nf90_put_var(ncid, varid(20), &
                  qtend_cond_aging_soag(1:nstop)) )
      call check( nf90_put_var(ncid, varid(21), &
                  qtend_rename_soag(1:nstop)) )
      call check( nf90_put_var(ncid, varid(22), &
                  qtend_newnuc_soag(1:nstop)) )
      call check( nf90_put_var(ncid, varid(23), &
                  qtend_coag_soag(1:nstop)) )

      ! Close the file. This frees up any internal netCDF resources
      ! associated with the file, and flushes any buffers.
      call check( nf90_close(ncid) )

      return
      end subroutine cambox_do_run


!-------------------------------------------------------------------------------
      subroutine load_pbuf( pbuf, lchnk, ncol, &
         cld, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens )

      use chem_mods, only: adv_mass, gas_pcnst, imozart
      use physconst, only: mwdry

      use modal_aero_data, only:  &
         lmassptrcw_amode, nspec_amode, numptrcw_amode, &
         qqcw_get_field

      use physics_buffer, only: physics_buffer_desc, &
         pbuf_get_index, pbuf_get_field

      type(physics_buffer_desc), pointer :: pbuf(:)  ! physics buffer for a chunk

      integer,  intent(in   ) :: lchnk, ncol

      real(r8), intent(in   ) :: cld(pcols,pver)    ! stratiform cloud fraction
      real(r8), intent(in   ) :: qqcw(pcols,pver,pcnst)  ! Cloudborne aerosol MR array
      real(r8), intent(in   ) :: dgncur_a(pcols,pver,ntot_amode)
      real(r8), intent(in   ) :: dgncur_awet(pcols,pver,ntot_amode)
      real(r8), intent(in   ) :: qaerwat(pcols,pver,ntot_amode)
      real(r8), intent(in   ) :: wetdens(pcols,pver,ntot_amode)

      integer :: idx, l, ll, n

      real(r8), pointer :: fldcw(:,:)
      real(r8), pointer :: ycld(:,:)
      real(r8), pointer :: ydgnum(:,:,:)
      real(r8), pointer :: ydgnumwet(:,:,:)
      real(r8), pointer :: yqaerwat(:,:,:)
      real(r8), pointer :: ywetdens(:,:,:)


      idx = pbuf_get_index( 'CLD' )
      call pbuf_get_field( pbuf, idx, ycld )
      ycld(:,:) = 0.0_r8
      ycld(1:ncol,:) = cld(1:ncol,:)

      idx = pbuf_get_index( 'DGNUM' )
      call pbuf_get_field( pbuf, idx, ydgnum )
      ydgnum(:,:,:) = 0.0_r8
      ydgnum(1:ncol,:,:) = dgncur_a(1:ncol,:,:)

      idx = pbuf_get_index( 'DGNUMWET' )
      call pbuf_get_field( pbuf, idx, ydgnumwet )
      ydgnumwet(:,:,:) = 0.0_r8
      ydgnumwet(1:ncol,:,:) = dgncur_awet(1:ncol,:,:)

      idx = pbuf_get_index( 'QAERWAT' )
      call pbuf_get_field( pbuf, idx, yqaerwat )
      yqaerwat(:,:,:) = 0.0_r8
      yqaerwat(1:ncol,:,:) = qaerwat(1:ncol,:,:)

      idx = pbuf_get_index( 'WETDENS_AP' )
      call pbuf_get_field( pbuf, idx, ywetdens )
      ywetdens(:,:,:) = 0.0_r8
      ywetdens(1:ncol,:,:) = wetdens(1:ncol,:,:)

      do n = 1, ntot_amode
      do ll = 0, nspec_amode(n)
         l = numptrcw_amode(n)
         if (ll > 0) l = lmassptrcw_amode(ll,n)
         fldcw => qqcw_get_field( pbuf, l, lchnk )
         fldcw(:,:) = 0.0_r8
         fldcw(1:ncol,:) = qqcw(1:ncol,:,l)
      end do
      end do


      return
      end subroutine load_pbuf


!-------------------------------------------------------------------------------
      subroutine unload_pbuf( pbuf, lchnk, ncol, &
         cld, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens )

      use chem_mods, only: adv_mass, gas_pcnst, imozart
      use physconst, only: mwdry

      use modal_aero_data, only:  &
         lmassptrcw_amode, nspec_amode, numptrcw_amode, &
         qqcw_get_field

      use physics_buffer, only: physics_buffer_desc, &
         pbuf_get_index, pbuf_get_field

      type(physics_buffer_desc), pointer :: pbuf(:)  ! physics buffer for a chunk

      integer,  intent(in   ) :: lchnk, ncol

      real(r8), intent(in   ) :: cld(pcols,pver)    ! stratiform cloud fraction

      real(r8), intent(inout) :: qqcw(pcols,pver,pcnst)  ! Cloudborne aerosol MR array
      real(r8), intent(inout) :: dgncur_a(pcols,pver,ntot_amode)
      real(r8), intent(inout) :: dgncur_awet(pcols,pver,ntot_amode)
      real(r8), intent(inout) :: qaerwat(pcols,pver,ntot_amode)
      real(r8), intent(inout) :: wetdens(pcols,pver,ntot_amode)

      integer :: i, idx, k, l, ll, n
      real(r8) :: tmpa

      real(r8), pointer :: fldcw(:,:)
      real(r8), pointer :: ycld(:,:)
      real(r8), pointer :: ydgnum(:,:,:)
      real(r8), pointer :: ydgnumwet(:,:,:)
      real(r8), pointer :: yqaerwat(:,:,:)
      real(r8), pointer :: ywetdens(:,:,:)


      idx = pbuf_get_index( 'CLD' )
      call pbuf_get_field( pbuf, idx, ycld )
! cld should not have changed, so check for changes rather than unloading it
!     cld(1:ncol,:) = ycld(1:ncol,:)
      tmpa = maxval( abs( cld(1:ncol,:) - ycld(1:ncol,:) ) )
      if (tmpa /= 0.0_r8) then
         write(*,*) '*** unload_pbuf cld change error - ', tmpa
         stop
      end if

      idx = pbuf_get_index( 'DGNUM' )
      call pbuf_get_field( pbuf, idx, ydgnum )
      dgncur_a(1:ncol,:,:) = ydgnum(1:ncol,:,:)

      idx = pbuf_get_index( 'DGNUMWET' )
      call pbuf_get_field( pbuf, idx, ydgnumwet )
      dgncur_awet(1:ncol,:,:) = ydgnumwet(1:ncol,:,:)

      idx = pbuf_get_index( 'QAERWAT' )
      call pbuf_get_field( pbuf, idx, yqaerwat )
      qaerwat(1:ncol,:,:) = yqaerwat(1:ncol,:,:)

      idx = pbuf_get_index( 'WETDENS_AP' )
      call pbuf_get_field( pbuf, idx, ywetdens )
      wetdens(1:ncol,:,:) = ywetdens(1:ncol,:,:)

      do n = 1, ntot_amode
      do ll = 0, nspec_amode(n)
         l = numptrcw_amode(n)
         if (ll > 0) l = lmassptrcw_amode(ll,n)
         fldcw => qqcw_get_field( pbuf, l, lchnk )
         qqcw(1:ncol,:,l) = fldcw(1:ncol,:)
      end do
      end do


      return
      end subroutine unload_pbuf


!-------------------------------------------------------------------------------


      subroutine check(status)
      integer, intent(in) :: status

      if(status /= nf90_noerr) then
         print *, trim(nf90_strerror(status))
         stop "Stopped"
      end if
      end subroutine check


!-------------------------------------------------------------------------------

      end module driver
