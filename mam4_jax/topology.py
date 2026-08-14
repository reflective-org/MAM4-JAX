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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Topology",
    "E3SM_MAM4_MOM",
    "get_topology",
    "set_topology",
    "available_topologies",
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

    # 0-based pcnst indices; -1 marks an unused slot.
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
            if n > len(self.lspectype_amode[m]):
                raise ValueError(
                    f"{self.name}: mode {m} declares {n} species but "
                    f"lspectype_amode[{m}] has only "
                    f"{len(self.lspectype_amode[m])} slots"
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


def _register(topology: Topology, *, make_active: bool = False) -> Topology:
    _REGISTRY[topology.name] = topology
    if make_active or not _ACTIVE:
        _ACTIVE.clear()
        _ACTIVE.append(topology)
    return topology


def get_topology() -> Topology:
    """The active topology."""
    if not _ACTIVE:
        raise RuntimeError(
            "no topology registered; import mam4_jax.data before use"
        )
    return _ACTIVE[0]


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
    _ACTIVE.clear()
    _ACTIVE.append(topology)
    return topology


def available_topologies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# Populated by data.py at import time.
E3SM_MAM4_MOM: Topology | None = None
