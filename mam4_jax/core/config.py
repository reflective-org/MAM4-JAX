"""Namelist-equivalent dataclass configs.

Mirrors the four namelist groups consumed by the Fortran reference's
test driver (`mam4-original-src-code/test_drivers/driver.F90`):
&time_input, &cntl_input, &met_input, &chem_input. Defaults reflect the
values written by `mam4-original-src-code/run_test.csh` for the
canonical 1800 s integration window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class TimeConfig:
    mam_dt: float
    mam_nstep: int


@dataclass(frozen=True)
class ControlConfig:
    mdo_gaschem: int = 0
    mdo_gasaerexch: int = 1
    mdo_rename: int = 1
    mdo_newnuc: int = 1
    mdo_coag: int = 1


@dataclass(frozen=True)
class MetConfig:
    temp: float = 273.0
    press: float = 1.0e5
    rh_clea: float = 0.9


@dataclass(frozen=True)
class ChemConfig:
    numc1: float = 1.0e8
    numc2: float = 1.0e9
    numc3: float = 1.0e5
    numc4: float = 2.0e8
    mfso41: float = 0.3
    mfpom1: float = 0.0
    mfsoa1: float = 0.3
    mfbc1: float = 0.0
    mfdst1: float = 0.0
    mfncl1: float = 0.4
    mfso42: float = 0.3
    mfsoa2: float = 0.3
    mfncl2: float = 0.4
    mfdst3: float = 0.0
    mfncl3: float = 0.4
    mfso43: float = 0.3
    mfbc3: float = 0.0
    mfpom3: float = 0.0
    mfsoa3: float = 0.3
    mfpom4: float = 0.0
    mfbc4: float = 1.0
    qso2: float = 1.0e-4
    qh2so4: float = 1.0e-13
    qsoag: float = 5.0e-10


@dataclass(frozen=True)
class RunConfig:
    time: TimeConfig
    cntl: ControlConfig = field(default_factory=ControlConfig)
    met: MetConfig = field(default_factory=MetConfig)
    chem: ChemConfig = field(default_factory=ChemConfig)


def load_yaml(path: str) -> RunConfig:
    """Load a RunConfig from a YAML file.

    Expected top-level keys: ``time``, ``cntl``, ``met``, ``chem``.
    Missing groups (except ``time``, which has no defaults) use the
    dataclass defaults.
    """
    with open(path) as fh:
        raw: dict[str, dict[str, Any]] = yaml.safe_load(fh)
    return RunConfig(
        time=TimeConfig(**raw["time"]),
        cntl=ControlConfig(**raw.get("cntl", {})),
        met=MetConfig(**raw.get("met", {})),
        chem=ChemConfig(**raw.get("chem", {})),
    )
