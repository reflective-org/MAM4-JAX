! physconst.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

module physconst

   ! Physical constants.  Use CCSM shared values whenever available.

   use shr_kind_mod, only: r8 => shr_kind_r8
   use shr_const_mod, only: shr_const_g,      shr_const_stebol, shr_const_tkfrz,  &
                            shr_const_mwdair, shr_const_rdair,  shr_const_mwwv,   &
                            shr_const_latice, shr_const_latvap, shr_const_cpdair, &
                            shr_const_rhofw,  shr_const_cpwv,   shr_const_rgas,   &
                            shr_const_karman, shr_const_pstd,   shr_const_rhodair,&
                            shr_const_avogad, shr_const_boltz,  shr_const_cpfw,   &
                            shr_const_rwv,    shr_const_zvir,   shr_const_pi,     &
                            shr_const_rearth, shr_const_sday,   shr_const_cday,   &
                            shr_const_spval         
   implicit none
   private
#if ( ! defined( CAMBOX_DEACTIVATE_THIS ) )
   public  :: physconst_readnl
#endif
   save

   ! Constantants for MAM spciesi classes
   integer, public, parameter :: spec_class_undefined  = 0
   integer, public, parameter :: spec_class_cldphysics = 1
   integer, public, parameter :: spec_class_aerosol    = 2
   integer, public, parameter :: spec_class_gas        = 3
   integer, public, parameter :: spec_class_other      = 4

   ! Constants based off share code or defined in physconst

   real(r8), public, parameter :: avogad      = shr_const_avogad     ! Avogadro's number (molecules/kmole)
   real(r8), public, parameter :: boltz       = shr_const_boltz      ! Boltzman's constant (J/K/molecule)
   real(r8), public, parameter :: cday        = shr_const_cday       ! sec in calendar day ~ sec
   real(r8), public, parameter :: cpair       = shr_const_cpdair     ! specific heat of dry air (J/K/kg)
   real(r8), public, parameter :: cpliq       = shr_const_cpfw       ! specific heat of fresh h2o (J/K/kg)
   real(r8), public, parameter :: karman      = shr_const_karman     ! Von Karman constant
   real(r8), public, parameter :: latice      = shr_const_latice     ! Latent heat of fusion (J/kg)
   real(r8), public, parameter :: latvap      = shr_const_latvap     ! Latent heat of vaporization (J/kg)
   real(r8), public, parameter :: pi          = shr_const_pi         ! 3.14...
   real(r8), public, parameter :: pstd        = shr_const_pstd       ! Standard pressure (Pascals)
   real(r8), public, parameter :: r_universal = shr_const_rgas       ! Universal gas constant (J/K/kmol)
   real(r8), public, parameter :: rhoh2o      = shr_const_rhofw      ! Density of liquid water (STP)
   real(r8), public, parameter :: spval       = shr_const_spval      !special value 
   real(r8), public, parameter :: stebol      = shr_const_stebol     ! Stefan-Boltzmann's constant (W/m^2/K^4)

   real(r8), public, parameter :: c0          = 2.99792458e8_r8      ! Speed of light in a vacuum (m/s)
   real(r8), public, parameter :: planck      = 6.6260755e-34_r8     ! Planck's constant (J.s)

   ! Molecular weights
   real(r8), public, parameter :: mwco2       =  44._r8             ! molecular weight co2
   real(r8), public, parameter :: mwn2o       =  44._r8             ! molecular weight n2o
   real(r8), public, parameter :: mwch4       =  16._r8             ! molecular weight ch4
   real(r8), public, parameter :: mwf11       = 136._r8             ! molecular weight cfc11
   real(r8), public, parameter :: mwf12       = 120._r8             ! molecular weight cfc12
   real(r8), public, parameter :: mwo3        =  48._r8             ! molecular weight O3
   real(r8), public, parameter :: mwso2       =  64._r8
   real(r8), public, parameter :: mwso4       =  96._r8
   real(r8), public, parameter :: mwh2o2      =  34._r8
   real(r8), public, parameter :: mwdms       =  62._r8
   real(r8), public, parameter :: mwnh4       =  18._r8


   ! modifiable physical constants for aquaplanet

   real(r8), public           :: gravit       = shr_const_g     ! gravitational acceleration (m/s**2)
   real(r8), public           :: sday         = shr_const_sday  ! sec in siderial day ~ sec
   real(r8), public           :: mwh2o        = shr_const_mwwv  ! molecular weight h2o
   real(r8), public           :: cpwv         = shr_const_cpwv  ! specific heat of water vapor (J/K/kg)
   real(r8), public           :: mwdry        = shr_const_mwdair! molecular weight dry air
   real(r8), public           :: rearth       = shr_const_rearth! radius of earth (m)
   real(r8), public           :: tmelt        = shr_const_tkfrz ! Freezing point of water (K)
   real(r8), public           :: vmdry        = 20.1_r8         ! molecular diffusion volume of dry air (unitless)

!---------------  Variables below here are derived from those above -----------------------

   real(r8), public           :: rga          = 1._r8/shr_const_g                 ! reciprocal of gravit
   real(r8), public           :: ra           = 1._r8/shr_const_rearth            ! reciprocal of earth radius
   real(r8), public           :: omega        = 2.0_R8*shr_const_pi/shr_const_sday! earth rot ~ rad/sec
   real(r8), public           :: rh2o         = shr_const_rgas/shr_const_mwwv     ! Water vapor gas constant ~ J/K/kg
   real(r8), public           :: rair         = shr_const_rdair   ! Dry air gas constant     ~ J/K/kg
   real(r8), public           :: epsilo       = shr_const_mwwv/shr_const_mwdair   ! ratio of h2o to dry air molecular weights 
   real(r8), public           :: zvir         = (shr_const_rwv/shr_const_rdair)-1.0_R8 ! (rh2o/rair) - 1
   real(r8), public           :: cpvir        = (shr_const_cpwv/shr_const_cpdair)-1.0_R8 ! CPWV/CPDAIR - 1.0
   real(r8), public           :: rhodair      = shr_const_pstd/(shr_const_rdair*shr_const_tkfrz)
   real(r8), public           :: cappa        = (shr_const_rgas/shr_const_mwdair)/shr_const_cpdair  ! R/Cp
   real(r8), public           :: ez           ! Coriolis expansion coeff -> omega/sqrt(0.375)   
   real(r8), public           :: Cpd_on_Cpv   = shr_const_cpdair/shr_const_cpwv

!!---------------  Variables below here are for Gauss-Hermite quadrature points  -----------------------                         

   integer, public             :: nghq        = 2
   real(r8), public, parameter :: xghq50(1:50) = (/  &
                -9.1824069581293166e+00_r8, -8.5227710309178040e+00_r8, -7.9756223682056371e+00_r8, &
                -7.4864094298641941e+00_r8, -7.0343235097706112e+00_r8, -6.6086479738553594e+00_r8, &
                -6.2029525192746719e+00_r8, -5.8129946754204056e+00_r8, -5.4357860872249484e+00_r8, &
                -5.0691175849172350e+00_r8, -4.7112936661690430e+00_r8, -4.3609731604545789e+00_r8, &
                -4.0170681728581341e+00_r8, -3.6786770625152694e+00_r8, -3.3450383139378910e+00_r8, &
                -3.0154977695745222e+00_r8, -2.6894847022677451e+00_r8, -2.3664939042986637e+00_r8, &
                -2.0460719686864093e+00_r8, -1.7278065475158986e+00_r8, -1.4113177548983000e+00_r8, &
                -1.0962511289576817e+00_r8, -7.8227172955460689e-01_r8, -4.6905905667823611e-01_r8, &
                -1.5630254688946871e-01_r8,  1.5630254688946871e-01_r8,  4.6905905667823611e-01_r8, &
                 7.8227172955460689e-01_r8,  1.0962511289576817e+00_r8,  1.4113177548983000e+00_r8, &
                 1.7278065475158986e+00_r8,  2.0460719686864093e+00_r8,  2.3664939042986637e+00_r8, &
                 2.6894847022677451e+00_r8,  3.0154977695745222e+00_r8,  3.3450383139378910e+00_r8, &
                 3.6786770625152694e+00_r8,  4.0170681728581341e+00_r8,  4.3609731604545789e+00_r8, &
                 4.7112936661690430e+00_r8,  5.0691175849172350e+00_r8,  5.4357860872249484e+00_r8, &
                 5.8129946754204056e+00_r8,  6.2029525192746719e+00_r8,  6.6086479738553594e+00_r8, &
                 7.0343235097706112e+00_r8,  7.4864094298641941e+00_r8,  7.9756223682056371e+00_r8, &
                 8.5227710309178040e+00_r8,  9.1824069581293166e+00_r8                             /)
   real(r8), public, parameter :: wghq50(1:50) = (/  &
                 1.8337940485734285e-37_r8,  1.6738016679078189e-32_r8,  1.2152441234044839e-28_r8, &
                 2.1376583083600919e-25_r8,  1.4170935995733830e-22_r8,  4.4709843654078295e-20_r8, &
                 7.7423829570433439e-18_r8,  8.0942618934651701e-16_r8,  5.4659440318155634e-14_r8, &
                 2.5066555238996854e-12_r8,  8.1118773649302035e-11_r8,  1.9090405438118900e-09_r8, &
                 3.3467934040214521e-08_r8,  4.4570299668178269e-07_r8,  4.5816827079555334e-06_r8, &
                 3.6840190537807259e-05_r8,  2.3426989210925602e-04_r8,  1.1890117817496436e-03_r8, &
                 4.8532638261719442e-03_r8,  1.6031941068412194e-02_r8,  4.3079159156765592e-02_r8, &
                 9.4548935477086191e-02_r8,  1.7003245567716388e-01_r8,  2.5113085633200255e-01_r8, &
                 3.0508512920439884e-01_r8,  3.0508512920439884e-01_r8,  2.5113085633200255e-01_r8, &
                 1.7003245567716388e-01_r8,  9.4548935477086191e-02_r8,  4.3079159156765592e-02_r8, &
                 1.6031941068412194e-02_r8,  4.8532638261719442e-03_r8,  1.1890117817496436e-03_r8, &
                 2.3426989210925602e-04_r8,  3.6840190537807259e-05_r8,  4.5816827079555334e-06_r8, &
                 4.4570299668178269e-07_r8,  3.3467934040214521e-08_r8,  1.9090405438118900e-09_r8, &
                 8.1118773649302035e-11_r8,  2.5066555238996854e-12_r8,  5.4659440318155634e-14_r8, &
                 8.0942618934651701e-16_r8,  7.7423829570433439e-18_r8,  4.4709843654078295e-20_r8, &
                 1.4170935995733830e-22_r8,  2.1376583083600919e-25_r8,  1.2152441234044839e-28_r8, &
                 1.6738016679078189e-32_r8,  1.8337940485734285e-37_r8                             /)

   real(r8), public, parameter :: xghq40(1:40) = (/  &
                -8.0987611392508505e+00_r8, -7.4115825314854691e+00_r8, -6.8402373052493557e+00_r8, &
                -6.3282553512200819e+00_r8, -5.8540950560303999e+00_r8, -5.4066542479701276e+00_r8, &
                -4.9792609785452555e+00_r8, -4.5675020728443947e+00_r8, -4.1682570668325001e+00_r8, &
                -3.7792067534352234e+00_r8, -3.3985582658596281e+00_r8, -3.0248798839012845e+00_r8, &
                -2.6569959984428957e+00_r8, -2.2939171418750837e+00_r8, -1.9347914722822959e+00_r8, &
                -1.5788698949316138e+00_r8, -1.2254801090462890e+00_r8, -8.7400661235708799e-01_r8, &
                -5.2387471383227724e-01_r8, -1.7453721459758237e-01_r8,  1.7453721459758237e-01_r8, &
                 5.2387471383227724e-01_r8,  8.7400661235708799e-01_r8,  1.2254801090462890e+00_r8, &
                 1.5788698949316138e+00_r8,  1.9347914722822959e+00_r8,  2.2939171418750837e+00_r8, &
                 2.6569959984428957e+00_r8,  3.0248798839012845e+00_r8,  3.3985582658596281e+00_r8, &
                 3.7792067534352234e+00_r8,  4.1682570668325001e+00_r8,  4.5675020728443947e+00_r8, &
                 4.9792609785452555e+00_r8,  5.4066542479701276e+00_r8,  5.8540950560303999e+00_r8, &
                 6.3282553512200819e+00_r8,  6.8402373052493557e+00_r8,  7.4115825314854691e+00_r8, &
                 8.0987611392508505e+00_r8                                                         /)
   real(r8), public, parameter :: wghq40(1:40) = (/  &
                 2.5910437138470341e-29_r8,  8.5440569637754309e-25_r8,  2.5675933654116484e-21_r8, &
                 1.9891810121165004e-18_r8,  6.0083587894908174e-16_r8,  8.8057076452161075e-14_r8, &
                 7.1565280526903606e-12_r8,  3.5256207913654217e-10_r8,  1.1212360832275837e-08_r8, &
                 2.4111441636705304e-07_r8,  3.6315761506930358e-06_r8,  3.9369339810924898e-05_r8, &
                 3.1385359454133164e-04_r8,  1.8714968295979507e-03_r8,  8.4608880082581318e-03_r8, &
                 2.9312565536172400e-02_r8,  7.8474605865404404e-02_r8,  1.6337873271327147e-01_r8, &
                 2.6572825187737725e-01_r8,  3.3864327742558897e-01_r8,  3.3864327742558897e-01_r8, &
                 2.6572825187737725e-01_r8,  1.6337873271327147e-01_r8,  7.8474605865404404e-02_r8, &
                 2.9312565536172400e-02_r8,  8.4608880082581318e-03_r8,  1.8714968295979507e-03_r8, &
                 3.1385359454133164e-04_r8,  3.9369339810924898e-05_r8,  3.6315761506930358e-06_r8, &
                 2.4111441636705304e-07_r8,  1.1212360832275837e-08_r8,  3.5256207913654217e-10_r8, &
                 7.1565280526903606e-12_r8,  8.8057076452161075e-14_r8,  6.0083587894908174e-16_r8, &
                 1.9891810121165004e-18_r8,  2.5675933654116484e-21_r8,  8.5440569637754309e-25_r8, &
                 2.5910437138470341e-29_r8                                                         /)

   real(r8), public, parameter :: xghq30(1:30) = (/  &
                -6.8633452935298918e+00_r8, -6.1382792201239349e+00_r8, -5.5331471515674959e+00_r8, &
                -4.9889189685899442e+00_r8, -4.4830553570925185e+00_r8, -4.0039086038612286e+00_r8, &
                -3.5444438731553500e+00_r8, -3.0999705295864417e+00_r8, -2.6671321245356174e+00_r8, &
                -2.2433914677615041e+00_r8, -1.8267411436036880e+00_r8, -1.4155278001981886e+00_r8, &
                -1.0083382710467235e+00_r8, -6.0392105862555234e-01_r8, -2.0112857654887151e-01_r8, &
                 2.0112857654887151e-01_r8,  6.0392105862555234e-01_r8,  1.0083382710467235e+00_r8, &
                 1.4155278001981886e+00_r8,  1.8267411436036880e+00_r8,  2.2433914677615041e+00_r8, &
                 2.6671321245356174e+00_r8,  3.0999705295864417e+00_r8,  3.5444438731553500e+00_r8, &
                 4.0039086038612286e+00_r8,  4.4830553570925185e+00_r8,  4.9889189685899442e+00_r8, &
                 5.5331471515674959e+00_r8,  6.1382792201239349e+00_r8,  6.8633452935298918e+00_r8 /)
   real(r8), public, parameter :: wghq30(1:30) = (/  &
                 2.9082547001312045e-21_r8,  2.8103336027508752e-17_r8,  2.8786070805487023e-14_r8, &
                 8.1061862974630327e-12_r8,  9.1785804243784853e-10_r8,  5.1085224507759580e-08_r8, &
                 1.5790948873247110e-06_r8,  2.9387252289229880e-05_r8,  3.4831012431868485e-04_r8, &
                 2.7379224730676565e-03_r8,  1.4703829704826678e-02_r8,  5.5144176870234186e-02_r8, &
                 1.4673584754089003e-01_r8,  2.8013093083921264e-01_r8,  3.8639488954181395e-01_r8, &
                 3.8639488954181395e-01_r8,  2.8013093083921264e-01_r8,  1.4673584754089003e-01_r8, &
                 5.5144176870234186e-02_r8,  1.4703829704826678e-02_r8,  2.7379224730676565e-03_r8, &
                 3.4831012431868485e-04_r8,  2.9387252289229880e-05_r8,  1.5790948873247110e-06_r8, &
                 5.1085224507759580e-08_r8,  9.1785804243784853e-10_r8,  8.1061862974630327e-12_r8, &
                 2.8786070805487023e-14_r8,  2.8103336027508752e-17_r8,  2.9082547001312045e-21_r8 /)
  
   real(r8), public, parameter :: xghq20(1:20) = (/  &
                -5.3874808900112328e+00_r8, -4.6036824495507442e+00_r8, -3.9447640401156252e+00_r8, &
                -3.3478545673832163e+00_r8, -2.7888060584281305e+00_r8, -2.2549740020892757e+00_r8, &
                -1.7385377121165861e+00_r8, -1.2340762153953231e+00_r8, -7.3747372854539439e-01_r8, &
                -2.4534070830090124e-01_r8,  2.4534070830090124e-01_r8,  7.3747372854539439e-01_r8, &
                 1.2340762153953231e+00_r8,  1.7385377121165861e+00_r8,  2.2549740020892757e+00_r8, &
                 2.7888060584281305e+00_r8,  3.3478545673832163e+00_r8,  3.9447640401156252e+00_r8, &
                 4.6036824495507442e+00_r8,  5.3874808900112328e+00_r8                             /)
   real(r8), public, parameter :: wghq20(1:20) = (/  &
                 2.2293936455341447e-13_r8,  4.3993409922731747e-10_r8,  1.0860693707692782e-07_r8, &
                 7.8025564785320599e-06_r8,  2.2833863601635365e-04_r8,  3.2437733422378567e-03_r8, &
                 2.4810520887463643e-02_r8,  1.0901720602002329e-01_r8,  2.8667550536283415e-01_r8, &
                 4.6224366960061009e-01_r8,  4.6224366960061009e-01_r8,  2.8667550536283415e-01_r8, &
                 1.0901720602002329e-01_r8,  2.4810520887463643e-02_r8,  3.2437733422378567e-03_r8, &
                 2.2833863601635365e-04_r8,  7.8025564785320599e-06_r8,  1.0860693707692782e-07_r8, &
                 4.3993409922731747e-10_r8,  2.2293936455341447e-13_r8                             /)

   real(r8), public, parameter :: xghq15(1:15) = (/ &
                -4.4999907073093919e+00_r8, -3.6699503734044527e+00_r8, -2.9671669279056032e+00_r8, &
                -2.3257324861738580e+00_r8, -1.7199925751864888e+00_r8, -1.1361155852109206e+00_r8, &
                -5.6506958325557577e-01_r8,  0.0000000000000000e+00_r8,  5.6506958325557577e-01_r8, &
                 1.1361155852109206e+00_r8,  1.7199925751864888e+00_r8,  2.3257324861738580e+00_r8, &
                 2.9671669279056032e+00_r8,  3.6699503734044527e+00_r8,  4.4999907073093919e+00_r8 /)
   real(r8), public, parameter :: wghq15(1:15) = (/  &
                 1.5224758042535209e-09_r8,  1.0591155477110625e-06_r8,  1.0000444123249982e-04_r8, &
                 2.7780688429127750e-03_r8,  3.0780033872546100e-02_r8,  1.5848891579593571e-01_r8, &
                 4.1202868749889870e-01_r8,  5.6410030872641737e-01_r8,  4.1202868749889870e-01_r8, &
                 1.5848891579593571e-01_r8,  3.0780033872546100e-02_r8,  2.7780688429127750e-03_r8, &
                 1.0000444123249982e-04_r8,  1.0591155477110625e-06_r8,  1.5224758042535209e-09_r8 /)

   real(r8), public, parameter :: xghq10(1:10) = (/ &
                -3.4361591188377374e+00_r8, -2.5327316742327897e+00_r8, -1.7566836492998816e+00_r8, &
                -1.0366108297895136e+00_r8, -3.4290132722370459e-01_r8,  3.4290132722370459e-01_r8, &
                 1.0366108297895136e+00_r8,  1.7566836492998816e+00_r8,  2.5327316742327897e+00_r8, &
                 3.4361591188377374e+00_r8                                                         /)
   real(r8), public, parameter :: wghq10(1:10) = (/  &
                 7.6404328552326410e-06_r8,  1.3436457467812324e-03_r8,  3.3874394455481106e-02_r8, &
                 2.4013861108231471e-01_r8,  6.1086263373532579e-01_r8,  6.1086263373532579e-01_r8, &
                 2.4013861108231471e-01_r8,  3.3874394455481106e-02_r8,  1.3436457467812324e-03_r8, &
                 7.6404328552326410e-06_r8                                                         /)

   real(r8), public, parameter :: xghq4(1:4) = (/ -1.6506801238857847e+00_r8, -5.2464762327529035e-01_r8, &
                                                   5.2464762327529035e-01_r8,  1.6506801238857847e+00_r8 /)
   real(r8), public, parameter :: wghq4(1:4) = (/  8.1312835447245185e-02_r8,  8.0491409000551273e-01_r8, &
                                                   8.0491409000551273e-01_r8,  8.1312835447245185e-02_r8 /)

   real(r8), public, parameter :: xghq2(1:2) = (/ -7.0710678118654746e-01_r8,  7.0710678118654746e-01_r8 /)
   real(r8), public, parameter :: wghq2(1:2) = (/  8.8622692545275794e-01_r8,  8.8622692545275794e-01_r8 /)

#if ( ! defined( CAMBOX_DEACTIVATE_THIS ) )
!================================================================================================
contains
!================================================================================================

   ! Read namelist variables.
   subroutine physconst_readnl(nlfile)

      use namelist_utils,  only: find_group_name
      use units,           only: getunit, freeunit
      use mpishorthand
      use spmd_utils,      only: masterproc
      use abortutils,      only: endrun
      use cam_logfile,     only: iulog

      character(len=*), intent(in) :: nlfile  ! filepath for file containing namelist input

      ! Local variables
      integer :: unitn, ierr
      character(len=*), parameter :: subname = 'physconst_readnl'
      logical       newg, newsday, newmwh2o, newcpwv, newmwdry, newrearth, newtmelt

      ! Physical constants needing to be reset (ie. for aqua planet experiments)
      namelist /physconst_nl/  cpwv, gravit, mwdry, mwh2o, rearth, sday, tmelt

      !-----------------------------------------------------------------------------

      if (masterproc) then
         unitn = getunit()
         open( unitn, file=trim(nlfile), status='old' )
         call find_group_name(unitn, 'physconst_nl', status=ierr)
         if (ierr == 0) then
            read(unitn, physconst_nl, iostat=ierr)
            if (ierr /= 0) then
               call endrun(subname // ':: ERROR reading namelist')
            end if
         end if
         close(unitn)
         call freeunit(unitn)
      end if

#ifdef SPMD
      ! Broadcast namelist variables
      call mpibcast(cpwv,      1,                   mpir8,   0, mpicom)
      call mpibcast(gravit,    1,                   mpir8,   0, mpicom)
      call mpibcast(mwdry,     1,                   mpir8,   0, mpicom)
      call mpibcast(mwh2o,     1,                   mpir8,   0, mpicom)
      call mpibcast(rearth,    1,                   mpir8,   0, mpicom)
      call mpibcast(sday,      1,                   mpir8,   0, mpicom)
      call mpibcast(tmelt,     1,                   mpir8,   0, mpicom)
#endif


      
      newg     =  gravit .ne. shr_const_g 
      newsday  =  sday   .ne. shr_const_sday
      newmwh2o =  mwh2o  .ne. shr_const_mwwv
      newcpwv  =  cpwv   .ne. shr_const_cpwv
      newmwdry =  mwdry  .ne. shr_const_mwdair
      newrearth=  rearth .ne. shr_const_rearth
      newtmelt =  tmelt  .ne. shr_const_tkfrz
      
      
      
      if (newg .or. newsday .or. newmwh2o .or. newcpwv .or. newmwdry .or. newrearth .or. newtmelt) then
         if (masterproc) then
            write(iulog,*)'****************************************************************************'
            write(iulog,*)'***    New Physical Constant Values set via namelist                     ***'
            write(iulog,*)'***                                                                      ***'
            write(iulog,*)'***    Physical Constant    Old Value                  New Value         ***'
            if (newg)       write(iulog,*)'***       GRAVITY   ',shr_const_g,gravit,'***'
            if (newsday)    write(iulog,*)'***       SDAY      ',shr_const_sday,sday,'***'
            if (newmwh2o)   write(iulog,*)'***       MWH20     ',shr_const_mwwv,mwh2o,'***'
            if (newcpwv)    write(iulog,*)'***       CPWV      ',shr_const_cpwv,cpwv,'***'
            if (newmwdry)   write(iulog,*)'***       MWDRY     ',shr_const_mwdair,mwdry,'***'
            if (newrearth)  write(iulog,*)'***       REARTH    ',shr_const_rearth,rearth,'***'
            if (newtmelt)   write(iulog,*)'***       TMELT     ',shr_const_tkfrz,tmelt,'***'
            write(iulog,*)'****************************************************************************'
         end if
         rga         = 1._r8/gravit 
         ra          = 1._r8/rearth
         omega       = 2.0_R8*pi/sday
         cpvir       = cpwv/cpair - 1._r8
         epsilo      = mwh2o/mwdry      
         
         !  rair and rh2o have to be defined before any of the variables that use them
         
         rair        = r_universal/mwdry
         rh2o        = r_universal/mwh2o  
         
         cappa       = rair/cpair       
         rhodair     = pstd/(rair*tmelt)
         zvir        =  (rh2o/rair)-1.0_R8
         ez          = omega / sqrt(0.375_r8)
         Cpd_on_Cpv  = cpair/cpwv
         
      else	
         ez          = omega / sqrt(0.375_r8)
      end if
      
    end subroutine physconst_readnl
#endif
  end module physconst
