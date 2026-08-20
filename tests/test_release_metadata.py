"""Release metadata invariants.

Small, but each of these has already been wrong at least once in this repo:
the wheel shipped without its data file (#62), and `pyproject.toml` said 0.0.1
while the newest tag was v0.2.0-beta.1.
"""
from __future__ import annotations

import importlib.metadata as md

import pytest

import mam4_jax


def test_version_is_single_sourced() -> None:
    """Installed metadata must agree with mam4_jax.__version__.

    pyproject now reads the version from the package attribute, so a drift here
    means someone reintroduced a literal.
    """
    try:
        installed = md.version("mam4-jax")
    except md.PackageNotFoundError:
        pytest.skip("not installed; nothing to compare against")
    assert installed == mam4_jax.__version__


def test_public_api_is_importable_from_the_top_level() -> None:
    """A downstream model must never need a submodule path.

    Submodule paths are internal and do move -- v0.3.0 deleted
    `mam4_jax.processes.*` entirely. Only `__all__` is covered by semver.
    """
    missing = [n for n in mam4_jax.__all__ if not hasattr(mam4_jax, n)]
    assert not missing, f"declared in __all__ but not present: {missing}"


def test_the_h2so4_switch_is_public() -> None:
    """configure_gas_netprod is the hook a host model drives gas chemistry
    through, so it must be reachable without touching mam4_jax.coupling."""
    assert callable(mam4_jax.configure_gas_netprod)
    assert "configure_gas_netprod" in mam4_jax.__all__


def test_coag_tables_are_present() -> None:
    """The .npz is loaded relative to __file__ by the module that needs it, so a
    wrong package-data glob yields an install that raises on import. That has
    happened once already (#62) and the file moved again in v0.3.0."""
    from mam4_jax.physics import coag
    assert coag._TABLES_PATH.exists(), f"missing: {coag._TABLES_PATH}"
