"""MAM4-JAX: JAX port of the MAM4 aerosol-microphysics box model.

Importing this package enables JAX's float64 mode globally (ADR-002).
"""
import jax

jax.config.update("jax_enable_x64", True)

x64_enabled: bool = bool(jax.config.read("jax_enable_x64"))

__version__ = "0.0.1"
