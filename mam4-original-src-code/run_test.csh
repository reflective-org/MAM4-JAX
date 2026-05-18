#!/bin/csh

#################################################
# step1: generate empty build and run directory #
#        set output path and copy source codes  #
#################################################
if (! -d build) then
   mkdir -p build
else
   rm -rf build
   mkdir -p build
endif

if (! -d run) then
   mkdir -p run
else
   rm -rf run
   mkdir -p run
endif

set outpath = /Users/sunj695/Downloads/mam_all/mam_src/restructed_codes/postprocess

cp box_model_utils/* build/
cp e3sm_src/* build/
cp e3sm_src_modified/* build/
cp test_drivers/* build/
cp Makefile build/

#######################################################
# step2: compile the source codes for executable file # 
#######################################################
cd build/
make

#####################################################
# step3: set up namelist variables and run tests    #
#        rename/move the output file to output path # 
#####################################################
cd ../run/

set nsteps     = (1 2 4 9 18 30 60 120 180 360 900 1800)

foreach i ($nsteps)
@ dt           = 1800 / $i
# set up namelist variable 
cat > namelist << EOF
&time_input
mam_dt         = $dt,
mam_nstep      = $i,
/
&cntl_input
mdo_gaschem    = 0,
mdo_gasaerexch = 1,
mdo_rename     = 1,
mdo_newnuc     = 1,
mdo_coag       = 1,
/
&met_input
temp           = 273.,
press          = 1.e5,
RH_CLEA        = 0.9,
/
&chem_input
numc1          = 1.e8,    ! unit: #/m3
numc2          = 1.e9,
numc3          = 1.e5,
numc4          = 2.e8,
!
! mfABCx: mass fraction of species ABC in mode x.
! 
! The mass fraction of mom is calculated by
! 1 - sum(mfABCx). If sum(mfABCx) > 1, an error
! is issued by the test driver. number of species
! ABC in each mode x comes from the MAM4 with mom.
! 
mfso41         = 0.3,
mfpom1         = 0.,
mfsoa1         = 0.3,
mfbc1          = 0.,
mfdst1         = 0.,
mfncl1         = 0.4,
mfso42         = 0.3,
mfsoa2         = 0.3,
mfncl2         = 0.4,
mfdst3         = 0.,
mfncl3         = 0.4,
mfso43         = 0.3,
mfbc3          = 0.,
mfpom3         = 0.,
mfsoa3         = 0.3,
mfpom4         = 0.,
mfbc4          = 1.,
qso2           = 1.e-4,
qh2so4         = 1.e-13,
qsoag          = 5.e-10,
/
EOF

# run 
./mam_box_test.exe
exit
# copy/rename the output
mv mam_output.nc $outpath/postprocess_input/mam_dt${dt}_ndt$i.nc

end

# clean up folders
cd ..
rm -rf build/ run/
