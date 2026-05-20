"""Physical constants used across the JAX port.

Values transcribed from
``mam4-original-src-code/e3sm_src/shr_const_mod.F90`` so they are exactly
the constants the Fortran reference uses at run time. The Fortran sets
the module-level state of ``wv_saturation`` via ``gestbl(...)``:

    epsqs  = epsil    = mwwv  / mwdair          (shr_const_mwwv/shr_const_mwdair)
    hlatv  = latvap   = 2.501e6   J/kg          (shr_const_latvap)
    hlatf  = latice   = 3.337e5   J/kg          (shr_const_latice)
    rgasv  = rh2o     = RGAS / mwwv             (RGAS = avogadro * boltzmann)

Derived values (epsqs, rgasv) are computed exactly the same way here so
the JAX port matches Fortran to within float64 round-off.
"""
from __future__ import annotations

# shr_const_mod.F90:33-61
BOLTZ: float = 1.38065e-23          # J/K/molecule  (shr_const_boltz)
AVOGAD: float = 6.02214e26          # molecules/kmole  (shr_const_avogad)
RGAS: float = AVOGAD * BOLTZ        # J/K/kmole  (shr_const_rgas)
MWDAIR: float = 28.966              # kg/kmole  (shr_const_mwdair)
MWWV: float = 18.016                # kg/kmole  (shr_const_mwwv)

# Derived gas constants
RDAIR: float = RGAS / MWDAIR        # J/K/kg  (shr_const_rdair)
RH2O: float = RGAS / MWWV           # J/K/kg  (shr_const_rwv)

# Latent heats
LATICE: float = 3.337e5             # J/kg  (shr_const_latice)
LATVAP: float = 2.501e6             # J/kg  (shr_const_latvap)
LATSUB: float = LATICE + LATVAP     # J/kg  (latent heat of sublimation)

# Densities (shr_const_rhofw = 1000 kg/m³ in shr_const_mod.F90:51).
RHOH2O: float = 1.0e3               # kg/m³  (density of liquid water at STP)

# Aliases matching the wv_saturation module-level names used after
# gestbl() runs. Use the wv_saturation names in code that mirrors Fortran
# formulas; use the shr_const_* names elsewhere.
EPSQS: float = MWWV / MWDAIR        # ~0.6219 (h2o:dry-air mw ratio)
HLATV: float = LATVAP
HLATF: float = LATICE
RGASV: float = RH2O
