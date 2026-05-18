# The MAM aerosol microphysics box model

Atmospheric aerosol particles play a crucial role in the climate system, influencing cloud formation, radiation balance, and atmospheric chemistry. The representation of aerosol microphysics in global aerosol models involves numerous physical and numerical assumptions as well as approximations, leading to significant discrepancies in the simulated aerosol physical properties and lifecycles among different models. Simplified models that isolate aerosol microphysics from resolved atmospheric dynamics and other parameterized processes (e.g., clouds and radiation) enable researchers to pinpoint the root causes of these discrepancies, which can be obscured in full global model comparisons.

The ```Modal Aerosol Module``` (MAM, Liu et al., 2012; Liu et al., 2016) in the Energy Exascale Earth System Model (E3SM) provides a simplified yet comprehensive treatment of aerosol processes. Supported by the ```Climate Model Development and Validation – Software Modernization``` (CMDV-SM) and ```E3SM Software Modernization``` projects, we have developed a new box model driver for the MAM aerosol microphysics calculations. This development also extends the functionality of the box model to facilitate comparisons of various aerosol microphysics treatments commonly used in global aerosol models, using either idealized or observation-constrained initial conditions and atmospheric states.

The MAM aerosol box model driver offers a convenient tool for new aerosol modelers to learn and test different aerosol microphysics parameterizations. It is also well-suited to support aerosol microphysics model intercomparison studies and machine learning applications (such as training data generation) aimed at enhancing global aerosol models.

## Description 

### Features  

- MAM aerosol code is the same as used in E3SMv1 (same module/data structure)
- Single-box or multi-box model configuration using data structure applied in the atmosphere component of E3SM/CESM
- Namelist control for meteorological state and initial condition of aerosol/gas concentrations 
- Namelist control for switching on/off individual microphysical processes  
- Improved I/O (now netCDF) and post-processing workflow 

### Upcoming features 

- Either idealized aerosol/gas initial condition or data from an E3SM simulation
- Example data for selected ARM sites 
- Convergence test suits and analysis scripts  
- Various drivers for testing individual processes in isolation or in combination

### Flowchart 

<img width="800" alt="image" src="https://github.com/kaizhangpnl/MAM_box_model/blob/main/figures/flowchart.png" />

Figure 1. Left: Flowchart showing the sequence of calculation (i.e., the time integration loop) in the E3SM atmosphere model. Colored boxes indicate aerosol parameterizations. Right: Diagram showing the process coupling (overall sequential splitting) in the MAM box aerosol model.


### Code  

```
e3sm_src:         files that are the same as MAM4 used in E3SMv1 
e3sm_modified:    modified MAM4 code used in E3SMv1
box_model_utils:  utility code for box model
test_drivers:     example test drivers 
```

## Developers: 

```
   Richcard C. Easter (v1.0, original version)  
   Hui Wan            (v1.1-v1.2) 
   Jian Sun           (v1.1-v1.2) 
   Kai Zhang          (v1.1-v1.2) 
```

## Versions

https://github.com/kaizhangpnl/MAM_box_model/blob/main/version.md 

## Documentation 

To be updated.  

## Test cases

We have developed offline drivers for aerosol microphysical processes including water uptake, nucleation, condensation, coagulation, aging, and redistribution of particles among modes of different size ranges. In addition to performing unit tests for individual aerosol processes, the box model is also used to test numerical convergence for time integration of multiple processes and the impact of operator splitting. 

Please contact us for more details. 

## Contact

Kai Zhang (kai.zhang@pnnl.gov) 

## Reference 

- Liu, X., Easter, R. C., Ghan, S. J., Zaveri, R., Rasch, P., Shi, X., Lamarque, J.-F., Gettelman, A., Morrison, H., Vitt, F., Conley, A., Park, S., Neale, R., Hannay, C., Ekman, A. M. L., Hess, P., Mahowald, N., Collins, W., Iacono, M. J., Bretherton, C. S., Flanner, M. G., and Mitchell, D.: Toward a minimal representation of aerosols in climate models: description and evaluation in the Community Atmosphere Model CAM5, Geosci. Model Dev., 5, 709–739, https://doi.org/10.5194/gmd-5-709-2012, 2012. 
- Liu, X., Ma, P.-L., Wang, H., Tilmes, S., Singh, B., Easter, R. C., Ghan, S. J., and Rasch, P. J.: Description and evaluation of a new four-mode version of the Modal Aerosol Module (MAM4) within version 5.3 of the Community Atmosphere Model, Geosci. Model Dev., 9, 505–522, https://doi.org/10.5194/gmd-9-505-2016, 2016.  
- Easter, R. C., S. J. Ghan, Y. Zhang, R. D. Saylor, E. G. Chapman, N. S. Laulainen, H. Abdul-Razzak, L. R. Leung, X. Bian, and R. A. Zaveri (2004), MIRAGE: Model description and evaluation of aerosols and trace gases, J. Geophys. Res., 109, D20210, https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2004JD004571, doi:10.1029/2004JD004571.

## Acknowlegment 


- Energy Exascale Earth System Model: Software and Algorithms for Exascale Subproject. https://e3sm.org/about/organization/phase-2/ngd-sub-projects/ 
- Climate Model Development and Validation – Software Modernization (CMDV-SM). https://climatemodeling.science.energy.gov/projects/climate-model-development-and-validation-software-modernization-cmdv-sm 


