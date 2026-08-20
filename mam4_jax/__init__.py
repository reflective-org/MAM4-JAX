"""MAM4-JAX: JAX port of the MAM4 aerosol-microphysics box model.

Importing this package enables JAX's ``jax_enable_x64`` flag globally by
default (ADR-002, amended by ADR-018): the ``diffrax`` condensation backend
(``atol=1e-20``) and a few core formulas need it. A host that uses ONLY the
float32-safe ``substep`` / ``astem`` backends can opt out by setting the
standard JAX environment variable ``JAX_ENABLE_X64=0`` **before** importing
``mam4_jax`` — then the whole coupled model can run in float32. We honor
JAX's own truthy values (``1``/``true``/``yes``/``on``, case-insensitive);
anything else (including ``0``/``false``) leaves x64 off.

Caveats
-------
- The opt-out must be set before the first ``import mam4_jax``; toggling
  ``jax.config.update("jax_enable_x64", False)`` in-process AFTER import
  leaves cached float64 module-level tensors in modules like ``coag``,
  which then produce float32×float64 promotion warnings on first call.
- Several modules (``kohler``, ``processes.wateruptake``,
  ``processes.calcsize``, ``processes.amicphys``, ``processes.newnuc``)
  currently contain explicit ``dtype=jnp.float64`` casts. With x64 off they
  emit a JAX UserWarning at trace time and silently truncate to float32 —
  fine for the ``substep`` / ``astem`` backends but a precision hazard for
  ``diffrax``. We emit a single warning at import time when x64 is off.
"""
import os
import warnings

import jax

_JAX_X64_TRUTHY = ("1", "true", "yes", "on")
_x64_env = os.environ.get("JAX_ENABLE_X64")
if _x64_env is None or _x64_env.lower() in _JAX_X64_TRUTHY:
    jax.config.update("jax_enable_x64", True)
# else: JAX honors its own env var on its own; we leave the config alone.

if not jax.config.read("jax_enable_x64"):
    warnings.warn(
        "mam4_jax: imported with jax_enable_x64=False (JAX_ENABLE_X64 opt-out). "
        "Several modules contain explicit dtype=jnp.float64 casts that will be "
        "truncated to float32 and emit JAX UserWarnings on first call: kohler, "
        "processes.wateruptake, processes.calcsize, processes.amicphys, "
        "processes.newnuc. The default 'diffrax' condensation backend uses "
        "atol=1e-20 and is numerically unsafe in float32 — switch to the "
        "'substep' or 'astem' backend via configure_condensation() before use. "
        "See ADR-018 in docs/KEY_DECISIONS.md.",
        UserWarning,
        stacklevel=2,
    )


def __getattr__(name):
    # PEP 562: ``mam4_jax.x64_enabled`` reads JAX's *live* state on each access,
    # so callers that toggle ``jax.config.update("jax_enable_x64", ...)`` at
    # runtime always see the truth. A frozen module-level snapshot would lie.
    if name == "x64_enabled":
        return bool(jax.config.read("jax_enable_x64"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Public API.
#
# Everything a downstream model should need, re-exported here so that callers
# never import from a submodule path. Those paths are internal and DO move --
# the package was restructured in v0.3.0, which renamed
# ``mam4_jax.processes.*`` out of existence. Anything imported from this
# top-level namespace is covered by semantic versioning; anything reached by a
# dotted submodule path is not.
#
# Imported at the bottom of the module, after the x64 setup above has run,
# because the science modules read the x64 flag at import time.
# ---------------------------------------------------------------------------
from .driver import run_step, run_timesteps, cloud_chem_simple_sub  # noqa: E402
from .coupling.amicphys import configure_gas_netprod  # noqa: E402
from .core.topology import (  # noqa: E402
    Topology,
    available_topologies,
    get_topology,
    register_topology,
    set_topology,
    topology_jit,
    trace_policy,
)

__all__ = [
    # time integration
    "run_step",
    "run_timesteps",
    "cloud_chem_simple_sub",
    # gas-phase source terms. configure_gas_netprod(h2so4=..., soa=...) sets the
    # "other-process" production rate [mol/mol/s] that the aerosol system sees;
    # this is the hook a host model drives its own gas chemistry through.
    "configure_gas_netprod",
    # mode configuration
    "Topology",
    "get_topology",
    "set_topology",
    "register_topology",
    "available_topologies",
    "topology_jit",
    "trace_policy",
    # introspection
    "__version__",
    "x64_enabled",
]

__version__ = "0.3.1"
