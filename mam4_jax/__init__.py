"""MAM4-JAX: JAX port of the MAM4 aerosol-microphysics box model.

Importing this package enables JAX's float64 mode globally by default
(ADR-002): the ``diffrax`` condensation backend (``atol=1e-20``) and a few
core formulas need it. A host that uses ONLY the float32-safe ``substep`` /
``astem`` backends can opt out by setting ``MAM4_JAX_ENABLE_X64=0`` in the
environment before importing — then the whole coupled model can run in
float32. (The coag ``qv12`` coefficient is the one spot that needed an
explicit float32-safe reformulation; with that in place the core is finite
and accurate in float32.)
"""
import os

import jax

if os.environ.get("MAM4_JAX_ENABLE_X64", "1") != "0":
    jax.config.update("jax_enable_x64", True)

x64_enabled: bool = bool(jax.config.read("jax_enable_x64"))

__version__ = "0.0.1"
