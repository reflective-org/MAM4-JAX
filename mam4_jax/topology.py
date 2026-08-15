"""Aerosol mode topology as a parameter, not a hard-coded module.

WHY THIS EXISTS. ``data.py`` pins one configuration -- E3SM's MAM4-MOM -- as
module-level constants. Supporting CESM/CAM, and MAM5 in particular, means the
mode count, the species carried by each mode, the size distribution and the
tracer index tables all become configuration rather than constants.

TWO ORTHOGONAL AXES, deliberately not one:

  ``variant``  "e3sm" | "cesm"   -- which code line's parameter values
  ``nmodes``   4 | 5             -- whether the coarse_strat mode is present

E3SM has no 5-mode configuration, so ``("e3sm", 5)`` is rejected at
construction. Every other combination is legal. Stratospheric water uptake is a
THIRD, independent axis and is deliberately not represented here: it already
runs in the 4-mode Fortran configuration, so tying it to ``nmodes`` would encode
a coupling that does not exist.

WHY MAM4 IS NOT A FORK OF MAM5. Verified in the CAM sources: the only
``MODAL_AERO_5MODE`` reference in the seven microphysics kernels is
``modal_aero_coag.F90:27``, and it places MAM5 in the *same* branch as MAM4
(``pair_option_acoag = 3``). The kernels are one code path with a mode count.
That is what makes a parameter the right shape here rather than a second module.

SCOPE OF THIS MODULE (plan 024 revision 2, PR B). It introduces the type, the
validation and the E3SM instance built from ``data.py``'s existing literals --
so the E3SM path is provably bit-unchanged. The CAM instances need index tables
extracted from CAM's chemistry preprocessor, which lands in PRs C-E; inventing
them here would be worse than their absence.

⚠ HOW TO CONSUME THIS FROM A JITTED KERNEL (PR C, and the reason `Topology` is
frozen and hashable). Do NOT read ``get_topology()`` from inside a jitted
function body. jit caches are not keyed on module globals, so a kernel traced
under one topology keeps returning that topology's answer after
``set_topology`` -- silently wrong physics, no warning. Confirmed empirically
during the PR-B review. Pass the topology in as a hashable static argument
instead, and give the change a test asserting that switching topologies
actually changes a jitted result.
"""
from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass

import numpy as np

# Detecting "are we inside a jit trace" has no public JAX API. This uses a
# private one, defensively: if a JAX upgrade moves it, the guard degrades to a
# no-op rather than breaking the package. The guard is a safety net, so failing
# open is the right direction -- but see _TRACE_PROBE_AVAILABLE, which the test
# suite asserts on so a silent degradation is noticed.
try:                                                        # pragma: no cover
    from jax._src.core import trace_ctx as _trace_ctx

    def _inside_jit_trace() -> bool:
        return type(_trace_ctx.trace).__name__ != "EvalTrace"

    _TRACE_PROBE_AVAILABLE = True
except Exception:                                           # pragma: no cover
    def _inside_jit_trace() -> bool:
        return False

    _TRACE_PROBE_AVAILABLE = False

__all__ = [
    "Topology",
    "get_topology",
    "register_topology",
    "set_topology",
    "available_topologies",
    "topology_jit",
    "trace_policy",
]

_VALID_VARIANTS = ("e3sm", "cesm")


@dataclass(frozen=True)
class Topology:
    """One fully-specified aerosol mode configuration.

    Every field is a primitive the Fortran also carries; derived quantities
    (``alnsg``, ``voltonumb*``, ``dumfac``) are computed here rather than stored,
    so they cannot drift out of sync with the values they come from.
    """

    name: str
    variant: str
    nmodes: int
    pcnst: int

    mode_names: tuple[str, ...]
    specname_amode: tuple[str, ...]
    nspec_amode: tuple[int, ...]

    # 0-based pcnst indices. -1 marks an unused SLOT in the per-(mode, slot)
    # tables below; numptr_amode has one entry per mode and is never -1.
    numptr_amode: tuple[int, ...]
    numptrcw_amode: tuple[int, ...]
    lspectype_amode: tuple[tuple[int, ...], ...]
    lmassptr_amode: tuple[tuple[int, ...], ...]
    lmassptrcw_amode: tuple[tuple[int, ...], ...]

    # Indexed by species TYPE (into specname_amode).
    specdens_amode: tuple[float, ...]
    spechygro_amode: tuple[float, ...]

    # Per-mode size distribution.
    sigmag_amode: tuple[float, ...]
    dgnum_amode: tuple[float, ...]
    dgnumlo_amode: tuple[float, ...]
    dgnumhi_amode: tuple[float, ...]
    rhcrystal_amode: tuple[float, ...]
    rhdeliques_amode: tuple[float, ...]

    provenance: str = ""

    # -- validation ---------------------------------------------------------
    def __post_init__(self) -> None:
        if self.variant not in _VALID_VARIANTS:
            raise ValueError(
                f"{self.name}: variant must be one of {_VALID_VARIANTS}, "
                f"got {self.variant!r}"
            )
        if self.nmodes not in (4, 5):
            raise ValueError(f"{self.name}: nmodes must be 4 or 5, got {self.nmodes}")
        if self.variant == "e3sm" and self.nmodes != 4:
            # E3SMv1 has no coarse_strat mode. Catching this at construction
            # keeps an impossible configuration from reaching the kernels.
            raise ValueError(
                f"{self.name}: E3SM has no {self.nmodes}-mode configuration; "
                "coarse_strat is CAM-only"
            )

        per_mode = {
            "mode_names": self.mode_names,
            "nspec_amode": self.nspec_amode,
            "numptr_amode": self.numptr_amode,
            "numptrcw_amode": self.numptrcw_amode,
            "lspectype_amode": self.lspectype_amode,
            "lmassptr_amode": self.lmassptr_amode,
            "lmassptrcw_amode": self.lmassptrcw_amode,
            "sigmag_amode": self.sigmag_amode,
            "dgnum_amode": self.dgnum_amode,
            "dgnumlo_amode": self.dgnumlo_amode,
            "dgnumhi_amode": self.dgnumhi_amode,
            "rhcrystal_amode": self.rhcrystal_amode,
            "rhdeliques_amode": self.rhdeliques_amode,
        }
        for label, value in per_mode.items():
            if len(value) != self.nmodes:
                raise ValueError(
                    f"{self.name}: {label} has {len(value)} entries, "
                    f"expected nmodes = {self.nmodes}"
                )

        nspec_types = len(self.specname_amode)
        for label, value in (("specdens_amode", self.specdens_amode),
                             ("spechygro_amode", self.spechygro_amode)):
            if len(value) != nspec_types:
                raise ValueError(
                    f"{self.name}: {label} has {len(value)} entries, expected "
                    f"{nspec_types} to match specname_amode"
                )

        # Slot tables must actually carry nspec_amode[m] populated entries, and
        # every populated species index must be a real species. A silent -1 in a
        # populated slot is the failure mode that produces zero-valued physics
        # rather than an error, so it is checked rather than assumed.
        for m in range(self.nmodes):
            n = self.nspec_amode[m]
            if not 0 <= n <= len(self.specname_amode):
                # A negative count makes range(n) empty, which would skip this
                # mode's entire slot loop and leave every pointer in it
                # unvalidated -- silent, and exactly the wrong direction.
                raise ValueError(
                    f"{self.name}: nspec_amode[{m}] = {n} outside "
                    f"[0, {len(self.specname_amode)}]"
                )
            for table_name in ("lspectype_amode", "lmassptr_amode",
                               "lmassptrcw_amode"):
                row = getattr(self, table_name)[m]
                if n > len(row):
                    raise ValueError(
                        f"{self.name}: mode {m} declares {n} species but "
                        f"{table_name}[{m}] has only {len(row)} slots"
                    )
            for s in range(n):
                if self.lspectype_amode[m][s] < 0:
                    raise ValueError(
                        f"{self.name}: mode {m} slot {s} is within "
                        f"nspec_amode = {n} but lspectype_amode is -1"
                    )
                if self.lspectype_amode[m][s] >= nspec_types:
                    raise ValueError(
                        f"{self.name}: mode {m} slot {s} references species "
                        f"{self.lspectype_amode[m][s]}, only {nspec_types} exist"
                    )
                if self.lmassptr_amode[m][s] < 0:
                    raise ValueError(
                        f"{self.name}: mode {m} slot {s} is within "
                        f"nspec_amode = {n} but lmassptr_amode is -1"
                    )
                if self.lmassptr_amode[m][s] >= self.pcnst:
                    raise ValueError(
                        f"{self.name}: mode {m} slot {s} points at pcnst index "
                        f"{self.lmassptr_amode[m][s]}, pcnst = {self.pcnst}"
                    )
            if not 0 <= self.numptr_amode[m] < self.pcnst:
                raise ValueError(
                    f"{self.name}: numptr_amode[{m}] = {self.numptr_amode[m]} "
                    f"outside [0, pcnst = {self.pcnst})"
                )

        # Every interstitial tracer index must be claimed exactly once. Two
        # modes sharing a mass tracer, or a number pointer colliding with a
        # mass pointer, is accepted arithmetic and silently double-counts mass.
        # That is the same class of quiet-wrong-physics as the -1 rule above,
        # and it is the specific failure mode of hand-extracting CAM's tables.
        seen: dict[int, str] = {}
        for m in range(self.nmodes):
            claims = [(self.numptr_amode[m], f"numptr_amode[{m}]")]
            claims += [
                (self.lmassptr_amode[m][s], f"lmassptr_amode[{m}][{s}]")
                for s in range(self.nspec_amode[m])
            ]
            for idx, label in claims:
                if idx in seen:
                    raise ValueError(
                        f"{self.name}: pcnst index {idx} claimed by both "
                        f"{seen[idx]} and {label}"
                    )
                seen[idx] = label

        for m in range(self.nmodes):
            if not (self.dgnumlo_amode[m] < self.dgnum_amode[m]
                    < self.dgnumhi_amode[m]):
                raise ValueError(
                    f"{self.name}: mode {m} violates "
                    f"dgnumlo < dgnum < dgnumhi: "
                    f"{self.dgnumlo_amode[m]:.3e} / {self.dgnum_amode[m]:.3e} / "
                    f"{self.dgnumhi_amode[m]:.3e}"
                )
            if self.sigmag_amode[m] <= 1.0:
                raise ValueError(
                    f"{self.name}: mode {m} sigmag = {self.sigmag_amode[m]} "
                    "must exceed 1"
                )

    # -- derived quantities -------------------------------------------------
    # Definitions from modal_aero_initialize_data.F90:428-435:
    #     alnsg     = log(sigmag)
    #     voltonumb = 1 / ( (pi/6) * dgnum^3 * exp(4.5 * alnsg^2) )
    #     dumfac    =       (pi/6)           * exp(4.5 * alnsg^2)

    @property
    def alnsg_amode(self) -> np.ndarray:
        return np.log(np.asarray(self.sigmag_amode, dtype=np.float64))

    @property
    def dumfac_amode(self) -> np.ndarray:
        return (np.pi / 6.0) * np.exp(4.5 * self.alnsg_amode ** 2)

    def _voltonumb(self, dgn: tuple[float, ...]) -> np.ndarray:
        return 1.0 / (self.dumfac_amode * np.asarray(dgn, dtype=np.float64) ** 3)

    @property
    def voltonumb_amode(self) -> np.ndarray:
        return self._voltonumb(self.dgnum_amode)

    @property
    def voltonumblo_amode(self) -> np.ndarray:
        return self._voltonumb(self.dgnumlo_amode)

    @property
    def voltonumbhi_amode(self) -> np.ndarray:
        return self._voltonumb(self.dgnumhi_amode)

    # -- convenience --------------------------------------------------------
    def mode_index(self, name: str) -> int:
        """0-based index of a named mode. Raises if absent."""
        try:
            return self.mode_names.index(name)
        except ValueError:
            raise KeyError(
                f"{self.name} has no mode {name!r}; modes are {self.mode_names}"
            ) from None

    def has_mode(self, name: str) -> bool:
        return name in self.mode_names

    @property
    def is_mam5(self) -> bool:
        return self.has_mode("coarse_strat")

    def __str__(self) -> str:
        return f"{self.name} (variant={self.variant}, nmodes={self.nmodes})"


# ---------------------------------------------------------------------------
# Registry.
#
# E3SM_MAM4_MOM is constructed in data.py, from the literals that already live
# there, and registered on import. Building it from those literals rather than
# re-typing them is deliberate: it makes "the E3SM path is bit-unchanged" a
# property of the code rather than a claim about my transcription.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Topology] = {}
_ACTIVE: list[Topology] = []       # single-element; list so it stays mutable


def register_topology(topology: Topology, *, make_active: bool = False) -> Topology:
    """Add `topology` to the registry.

    Registration deliberately does NOT activate implicitly. An earlier version
    activated whenever nothing was active yet, which meant the first import of
    any module that registers could silently revert an explicit
    ``set_topology`` choice. Callers say what they mean.
    """
    _REGISTRY[topology.name] = topology
    if make_active:
        _ACTIVE[:] = [topology]        # single store; no window with 0 entries
    return topology


# What to do when get_topology() is called during a jit trace. "error" by
# default: reading the active topology from inside a traced function bakes it
# into that trace, and the cache is not keyed on it, so a later set_topology
# silently returns the old answer. See the module docstring.
_TRACE_POLICY = ["error"]


def trace_policy(policy: str | None = None) -> str:
    """Get, or set, what happens when get_topology() runs inside a jit trace.

    ``"error"`` (default), ``"warn"``, or ``"allow"``. The escape hatch exists
    for the deliberate case where a caller knows the kernel is retraced per
    topology; it should be rare, and reaching for it is a signal to pass the
    topology as a static argument instead.

    The policy is process-global, not thread-local, so ``"allow"`` disables the
    guard for every thread.

    KNOWN FALSE POSITIVES. The probe reports "tracing" for any trace, but only
    some traces actually go stale. Measured: bare ``vmap`` and bare
    ``grad``/``jvp``/``vjp`` -- with no jit and no ``lax`` control-flow
    primitive -- retrace on every call, so a global read there is correct and
    the guard is over-strict. Everything else tested does go stale, including
    ``scan``, ``fori_loop``, ``cond`` and ``while_loop`` *outside* jit (their
    bodies are cached on the callable by ``_initial_style_jaxpr``). The
    over-strictness is loud rather than silent, and passing the topology as a
    static argument is the right answer in the false-positive cases too.
    """
    if policy is None:
        return _TRACE_POLICY[0]
    if policy not in ("error", "warn", "allow"):
        raise ValueError(
            f"trace policy must be 'error', 'warn' or 'allow', got {policy!r}"
        )
    _TRACE_POLICY[0] = policy
    return policy


_TRACE_MESSAGE = (
    "get_topology() was called inside a jit trace. jit caches are not keyed on "
    "module globals, so this bakes the CURRENT topology into the compiled "
    "function: after set_topology(...) the kernel keeps returning the old "
    "topology's answer, with no error and no warning. Pass the Topology in as a "
    "static argument instead -- it is frozen and hashable for exactly this "
    "purpose, e.g. @topology_jit or jax.jit(f, static_argnames=('topology',)). "
    "If this really is intentional, wrap the call in "
    "mam4_jax.topology.trace_policy('allow')."
)


def get_topology() -> Topology:
    """The active topology.

    Raises inside a jit trace by default; see :func:`trace_policy`.
    """
    if not _ACTIVE:
        raise RuntimeError(
            "no topology registered; import mam4_jax.data before use"
        )
    if _inside_jit_trace():
        policy = _TRACE_POLICY[0]
        if policy == "error":
            raise RuntimeError(_TRACE_MESSAGE)
        if policy == "warn":
            warnings.warn(_TRACE_MESSAGE, RuntimeWarning, stacklevel=2)
    return _ACTIVE[0]


def topology_jit(fn=None, **jit_kwargs):
    """``jax.jit`` with ``topology`` treated as a static argument.

    The safe counterpart to reading ``get_topology()`` inside a kernel: the
    topology participates in the cache key, so switching topologies retraces
    instead of silently reusing the previous compilation.

    ``topology`` may be passed positionally or by keyword -- JAX resolves
    ``static_argnames`` onto argnums for POSITIONAL_OR_KEYWORD parameters, so
    both are genuinely static.

    Usage::

        @topology_jit
        def kernel(q, *, topology):
            return q * topology.voltonumb_amode[0]

        kernel(q, topology=get_topology())
    """
    import jax

    def wrap(f):
        # .get + a copy, NOT .pop: popping mutates the dict closed over by
        # `wrap`, so reusing a factory result silently dropped the caller's
        # static argnames on every function after the first --
        #     deco = topology_jit(static_argnames=("mode",))
        #     @deco
        #     def k1(...)   # ('mode', 'topology')
        #     @deco
        #     def k2(...)   # ('topology',)  <- 'mode' silently dynamic
        # and when the static arg is only used for indexing that degrades
        # quietly into a dynamic argument rather than raising.
        names = set(jit_kwargs.get("static_argnames", ()))
        names.add("topology")
        rest = {k: v for k, v in jit_kwargs.items() if k != "static_argnames"}
        return jax.jit(f, static_argnames=tuple(sorted(names)), **rest)

    return wrap if fn is None else wrap(fn)


def set_topology(topology: "Topology | str") -> Topology:
    """Make `topology` active, by instance or by registered name.

    Note this does NOT retroactively change ``data.py``'s module-level
    constants, which are bound at import for the default. Threading the active
    topology through the call sites is PR C onward; until then, switching is
    only meaningful for code that reads ``get_topology()`` directly.
    """
    if isinstance(topology, str):
        try:
            topology = _REGISTRY[topology]
        except KeyError:
            raise KeyError(
                f"unknown topology {topology!r}; registered: "
                f"{sorted(_REGISTRY)}"
            ) from None
    _REGISTRY.setdefault(topology.name, topology)
    _ACTIVE[:] = [topology]
    return topology


def available_topologies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# NOTE: the E3SM instance lives at ``mam4_jax.data.E3SM_MAM4_MOM``, built there
# from that module's own literals (see data.py). It is deliberately NOT
# re-exported here: an earlier version declared it as None for data.py to
# populate, and data.py never assigned back, so
# ``from mam4_jax.topology import E3SM_MAM4_MOM`` silently handed out None.
# Use ``get_topology()`` or import from ``mam4_jax.data``.
