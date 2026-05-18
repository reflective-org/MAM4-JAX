! cam_history.F90
!    This F90 module file is a special version of the equivalent ACME (and CAM5) module.
!    It provides the functionality needed by the cambox offline code
!    that is used for development and testing of the modal aerosol module (MAM),
!    but (in most cases) not all the functionality of the equivalent ACME module.
!    Also, it may have been taken from a version of CAM5 that was older
!    than ACME-V0 (i.e., pre 2014).

      module cam_history

      use shr_kind_mod, only: r8 => shr_kind_r8

      implicit none

      public

      integer, parameter :: fieldname_len = 64 
      integer, parameter :: phys_decomp = 1   ! *** this is a kludge to avoid the pio file

      integer, public :: ncol_for_outfld

      character(len=10), public :: horiz_only = "horiz_only"

      contains

!#######################################################################

   subroutine addfldv1 (fname, units, numlev, avgflag, long_name, &
                      decomp_type, flag_xyfill,sampling_seq, &
                      begdim1, enddim1, &
                      begdim2, enddim2,&
                      begdim3, enddim3, mdimnames, fill_value )

!
!----------------------------------------------------------------------- 
! 
! Purpose: Add a field to the master field list
! 
! Method: Put input arguments of field name, units, number of levels, averaging flag, and 
!         long name into a type entry in the global master field list (masterlist).
! 
! Author: CCM Core Group
! 
!-----------------------------------------------------------------------

!
! Arguments
!
      character(len=*), intent(in) :: fname      ! field name--should be "max_fieldname_len" characters long
                                                 ! or less
      character(len=*), intent(in) :: units      ! units of fname--should be 8 chars
      character(len=1), intent(in) :: avgflag    ! averaging flag
      character(len=*), intent(in) :: long_name  ! long name of field

      integer, intent(in) :: numlev              ! number of vertical levels (dimension and loop)
      integer, intent(in) :: decomp_type         ! decomposition type

      logical, intent(in), optional :: flag_xyfill ! non-applicable xy points flagged with fillvalue

      character(len=*), intent(in), optional :: sampling_seq ! sampling sequence - if not every timestep, 
                                                             ! how often field is sampled:  
                                                             ! every other; only during LW/SW radiation calcs, etc.
      integer, intent(in), optional :: begdim1, enddim1
      integer, intent(in), optional :: begdim2, enddim2
      integer, intent(in), optional :: begdim3, enddim3

      character(len=*), intent(in), optional :: mdimnames(:)
      real(r8), intent(in), optional :: fill_value

      character(len=16) :: txtaa
      txtaa = fname
      write(95,'(2a,i6)') 'addfld - ', txtaa, numlev

      return
   end subroutine addfldv1

!#######################################################################

   subroutine addfld (fname, numlev_txt, avgflag, units, long_name, &
                      flag_xyfill, sampling_seq, &
                      begdim1, enddim1, &
                      begdim2, enddim2,&
                      begdim3, enddim3, mdimnames, fill_value )

!
!----------------------------------------------------------------------- 
! 
! Purpose: Add a field to the master field list
! 
! Method: Put input arguments of field name, units, number of levels, averaging flag, and 
!         long name into a type entry in the global master field list (masterlist).
! 
! Author: CCM Core Group
! 
!-----------------------------------------------------------------------

      use ppgrid, only:  pver
!
! Arguments
!
      character(len=*), intent(in) :: fname      ! field name--should be "max_fieldname_len" characters long
                                                 ! or less
      character(len=*), intent(in) :: units      ! units of fname--should be 8 chars
      character(len=1), intent(in) :: avgflag    ! averaging flag
      character(len=*), intent(in) :: long_name  ! long name of field

      character(len=*), intent(in) :: numlev_txt(*)              ! number of vertical levels (dimension and loop)

      logical, intent(in), optional :: flag_xyfill ! non-applicable xy points flagged with fillvalue

      character(len=*), intent(in), optional :: sampling_seq ! sampling sequence - if not every timestep, 
                                                             ! how often field is sampled:  
                                                             ! every other; only during LW/SW radiation calcs, etc.
      integer, intent(in), optional :: begdim1, enddim1
      integer, intent(in), optional :: begdim2, enddim2
      integer, intent(in), optional :: begdim3, enddim3

      character(len=*), intent(in), optional :: mdimnames(:)
      real(r8), intent(in), optional :: fill_value

      character(len=16) :: txtaa
      txtaa = fname
      if (numlev_txt(1) == 'horiz_only') then
         write(95,'(2a,i6)') 'addfld - ', txtaa, 1
      else
         write(95,'(2a,i6)') 'addfld - ', txtaa, pver
      end if

      return
   end subroutine addfld

!#######################################################################

   subroutine add_default (name, tindex, flag)
!
!----------------------------------------------------------------------- 
! 
! Purpose: Add a field to the default "on" list for a given history file
! 
! Method: 
! 
!-----------------------------------------------------------------------
!
! Arguments
!
      character(len=*), intent(in) :: name  ! field name
      character(len=1), intent(in) :: flag  ! averaging flag

      integer, intent(in) :: tindex         ! history tape index

      character(len=16) :: txtaa
      txtaa = name
      write(95,'(2a,i6)') 'adddef - ', txtaa, tindex

      return
   end subroutine add_default

!#######################################################################

   subroutine outfld (fname, field, idim, c)
!
!----------------------------------------------------------------------- 
! 
! Purpose: Accumulate (or take min, max, etc. as appropriate) input field
!          into its history buffer for appropriate tapes
! 
! Method: Check 'masterlist' whether the requested field 'fname' is active
!         on one or more history tapes, and if so do the accumulation.
!         If not found, return silently.
! 
! Author: CCM Core Group
! 
!-----------------------------------------------------------------------
!
      use ppgrid, only: pcols, pver

! Arguments
!
      character(len=*), intent(in) :: fname ! Field name--should be 8 chars long

      integer, intent(in) :: idim           ! Longitude dimension of field array
      integer, intent(in) :: c              ! chunk (physics) or latitude (dynamics) index

      real(r8), intent(in) :: field(idim,*) ! Array containing field values

      integer :: k
      character(len=16) :: txtaa

      txtaa = fname
      write(90,'(/a,1p,20e11.3)') txtaa, field(1:ncol_for_outfld,1)

      if ( txtaa == 'SOAG_sfgaex3d   ' .or. &
           txtaa == 'num_a2_nuc1     ' .or. &
           txtaa == 'num_a2_nuc2     ' ) then
         do k = 2, pver
            write(90,'(a,i2,10x,1p,20e11.3)') '  k=', k, field(1:ncol_for_outfld,k)
         end do
      end if

      
   end subroutine outfld

!#######################################################################

      end module cam_history
