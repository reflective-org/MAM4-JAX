# CAM box-model reference trajectories

Generated from `mam-box-fortran` **fixdumfac** builds (the JAX calcsize
uses the correct per-mode `dumfac` by default; the non-fixdumfac builds
reproduce upstream CAM's stale-scalar bug — see
`mam-box-fortran/docs/bugs/BUG-cam-calcsize-stale-dumfac.md` and
`calcsize(bug_compat_stale_dumfac=True)`):

    cd mam-box-fortran
    ./build/build_cam.sh --fix-dumfac
    ./build/build_cam.sh --fix-dumfac --mam5
    build/cam-mam4-fixdumfac/mam_box_cam.exe 0.9 nostrat 30 120 all mam4 no_acc_crs
    build/cam-mam4-fixdumfac/mam_box_cam.exe 0.9 strat   30 120 all mam4 no_acc_crs
    build/cam-mam5-fixdumfac/mam_box_cam.exe 0.9 nostrat 30 120 all mam5 no_acc_crs
    build/cam-mam5-fixdumfac/mam_box_cam.exe 0.9 strat   30 120 all mam5 no_acc_crs

(each run writes `mam_box_cam.out` in its cwd; the four are committed
here as `<topo>_<kind>.out`). Namelist: all defaults (T=273 K, p=1e5 Pa,
RH=0.9, dt=30 s, 120 steps, numc = 1e8/1e9/1e5/1 (+1e4 mode 5), so4frac
1/1/1/0, qso2=1e-7, qh2so4=1e-11 vmr, SO2→H2SO4 1e-5 /s).

Columns are `es14.6` — SEVEN significant digits — except `totS_mol`
(`es24.16`, full precision). Per-tracer comparisons therefore bottom out
at ~5e-7 relative (the print floor); `totS` is the machine-precision
handle and agrees with the JAX driver at ~4.5e-15 on all four runs.
