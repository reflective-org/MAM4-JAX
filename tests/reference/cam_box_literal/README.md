# Literal-CAM box trajectories with an aitken<->accum transfer

Generated 2026-11-02 from the **non-fixdumfac** `mam-box-fortran` builds
(`cam6_4_187` sources byte-identical, zero patches; mam-box-fortran
`d0cf723`), i.e. upstream CAM *including* the stale `dumfac` in
`modal_aero_calcsize_sub`'s aitken<->accum recompute
(`mam-box-fortran/docs/bugs/BUG-cam-calcsize-stale-dumfac.md`) and the
first surface-tension lookup's abscissa (`BUG-cam-wateruptake-surftens-interp.md`):

    build/cam-mam4/mam_box_cam.exe 0.9 nostrat 30 120 all mam4 no_acc_crs 1e-5
    build/cam-mam4/mam_box_cam.exe 0.9 strat   30 120 all mam4 no_acc_crs 1e-5
    build/cam-mam5/mam_box_cam.exe 0.9 nostrat 30 120 all mam5 no_acc_crs 1e-5
    build/cam-mam5/mam_box_cam.exe 0.9 strat   30 120 all mam5 no_acc_crs 1e-5

Executables: cam-mam4 sha256 `ed27538d…2323`, cam-mam5 sha256 `08f4a6e1…83dd`.
Namelist as `../cam_box/README.md` except the eighth argument, initial SO2
`qso2 = 1e-5` vmr (default 1e-7): at 1e-7 no transfer fires and the bug is
unreachable. With 1e-5 the transfer fires from step ~50 (MAM5) / ~60 (MAM4);
the same runs on the fixdumfac builds differ by up to 21 % (MAM4) / 34 %
(MAM5) in `dgn_a1`, so these fixtures discriminate the two behaviours.
Columns as `../cam_box/README.md` (seven significant digits).
