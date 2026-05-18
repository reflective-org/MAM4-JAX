# Makefile for pegasus.2.6.1
#
#----------------------------------------------------------------------
#
# includes
#
include ./cambox_config.make.in
include ./cambox_config.cpp.in

OUT = ../run

#
# Suffix rules...
#
.SUFFIXES : .F .f .F90 .f90 .o

.F.o:
	$(CPP) $(CPPFLAGS) $*.F $(OUT)/$*.f ;  $(FC) -c $(FCFLAGS) $(OUT)/$*.f

.F.f:
	$(CPP) $(CPPFLAGS) $*.F $(OUT)/$*.f

.f.o:
	$(CPP) $(CPPFLAGS) $(OUT)/$*.F $(OUT)/$*.f ;  $(FC) -c $(FCFLAGS) $(OUT)/$*.f

.F90.o:
	$(RM) $(OUT)/$*.mod ; $(CPP) $(CPPFLAGS) $*.F90 $(OUT)/$*.f90 ; cd $(OUT)/ ; $(FC) -c $(FCFLAGS) $(OUT)/$*.f90

.F90.f90:
	$(RM) $(OUT)/$*.mod ; $(CPP) $(CPPFLAGS) $*.F90 $(OUT)/$*.f90

.f90.o:
	$(RM) $(OUT)/$*.mod ; $(CPP) $(CPPFLAGS) $*.F90 $(OUT)/$*.f90 ; cd $(OUT)/ ; $(FC) -c $(FCFLAGS) $*.f90

#
# Object lists...
#
# OBJ1, OBJ2, and OBJ4 files 
#    These files are special versions of the equivalent CAM5 modules.
#    They provide the functionality needed by the cambox offline code
#    that is used for development and testing of the modal aerosol module (MAM),
#    but (in most cases) not all the functionality of the equivalent CAM5 module.
#    Also, they may have been taken from a version of CAM5 that was older
#    than ACME-V0 (i.e., pre 2014).
# OBJ3 and OBJ5 files
#    These files should be identical to the equivalent CAM5 module (or almost so).
# OBJ9 files 
#    These are specific to the offline test code and have no CAM equivalents.

CONFIGINFO = \
	./cambox_config.make.in  \
	./cambox_config.cpp.in

OBJ1 = \
	shr_kind_mod.o  \
	shr_const_mod.o

OBJ2 = \
	cam_logfile.o  \
	spmd_utils.o  \
	abortutils.o  \
	cam_abortutils.o  \
	ppgrid.o  \
	pmgrid.o  \
	constituents.o  \
	radconstants.o  \
	physconst.o  \
	cam_history.o  \
	units.o  \
	ref_pres.o  \
	dyn_grid.o  \
	buffer.o  \
	physics_buffer.o  \
	physics_types.o  \
	phys_control.o  \
	mo_constants.o  \
	chem_mods.o  \
	mo_tracname.o  \
	mo_chem_utls.o  \
	gffgch.o  \
	wv_saturation.o  \
	error_function.o  \
	shr_spfn_mod.o  \
	infnan.o \
	seasalt_model.o  \
	time_manager.o  \
	aerodep_flx.o  \
	modal_aero_convproc.o  \
	modal_aero_deposition.o 

OBJ3 = \
	modal_aero_data.o

OBJ4 = \
	rad_constituents.o

OBJ5 = \
	modal_aero_wateruptake.o \
	modal_aero_rename.o  \
	modal_aero_calcsize.o  \
	modal_aero_gasaerexch.o  \
	modal_aero_newnuc.o  \
	modal_aero_coag.o  \
	modal_aero_amicphys.o \
	modal_aero_initialize_data.o \

OBJ9 = \
	gaschem_simple.o  \
	cloudchem_simple.o  \
	driver.o  \
	main.o

#-----------------------------------------------------------------------------
# WARNING: Don't touch anything below this line unless you exactly know it !!!
#-----------------------------------------------------------------------------


#
# Dependancies...
#
$(EXE): $(OBJ1) $(OBJ2) $(OBJ3) $(OBJ4) $(OBJ5) $(OBJ9) $(CONFIGINFO)
	cd $(OUT)/ ; $(FC) -o $@ $(FCFLAGS) $(LDFLAGS) $(OBJ1) $(OBJ2) $(OBJ3) $(OBJ4) $(OBJ5) $(OBJ9)


$(OBJ1):  $(CONFIGINFO)

$(OBJ2):  $(OBJ1)

$(OBJ3):  $(OBJ2)

$(OBJ4):  $(OBJ3)

$(OBJ5):  $(OBJ4)

$(OBJ9):  $(OBJ5)

rad_constituents.o: modal_aero_data.o

clean:
	cd $(OUT) ; /bin/rm  *.x  *.o  *.f  *.f90  *.mod fort.* *.out* make_log *exe


cleanmod:
	cd $(OUT) ; /bin/rm  *.x  *.mod  module*.o  module*.f90  module*.f
