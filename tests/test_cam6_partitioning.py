"""CAM6 saturation concentration on the unchanged MOM local mass basis."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mam4_jax.core import data
from mam4_jax.coupling.amicphys import _mam_soaexch_1subarea_astem


@pytest.mark.parametrize('poa', [0.0, 2e-7])
@pytest.mark.parametrize('temperature', [260.0, 298.0, 310.0])
def test_cam6_saturation_and_organic_conservation(temperature, poa):
    # SOA-rich single absorbing mode, no POA: gas equilibrium is g0 exactly.
    gas = jnp.zeros(2)
    aerosol = jnp.zeros((data.AMICPHYS_NAER, data.NTOT_AMODE))
    aerosol = aerosol.at[data.AMICPHYS_IAER_SOA, 0].set(1e-7)
    aerosol = aerosol.at[data.AMICPHYS_IAER_POM, 0].set(poa)
    uptake = jnp.zeros((2, data.NTOT_AMODE)).at[0, 0].set(0.1)
    def exchange(mw):
        return _mam_soaexch_1subarea_astem(
            gas, gas, aerosol, 10000.0, temperature, 101325.0, uptake, mw)
    cam_gas, _, cam_aer = jax.jit(exchange)(250.0)
    mom_gas, _, mom_aer = exchange(None)
    explicit_mom, _, _ = exchange(150.0)
    np.testing.assert_array_equal(explicit_mom, mom_gas)
    # Independent ideal-gas / Clausius-Clapeyron equilibrium calculation.
    from mam4_jax.core.constants import RGAS
    vapor_pressure = 1e-10 * np.exp(-156e3 / (RGAS / 1000) *
                                   (1 / temperature - 1 / 298.0))
    def equilibrium(g0):
        b = poa * 0.1 + 1e-7 + g0
        return 2 * g0 * 1e-7 / (b + np.sqrt(b*b - 4*g0*1e-7))
    np.testing.assert_allclose(cam_gas[0], equilibrium(vapor_pressure * 250 / 150),
                               rtol=1e-5, atol=1e-22)
    np.testing.assert_allclose(mom_gas[0], equilibrium(vapor_pressure),
                               rtol=1e-5, atol=1e-22)
    for new_gas, new_aer in [(cam_gas, cam_aer), (mom_gas, mom_aer)]:
        np.testing.assert_allclose(new_gas[0] + new_aer[0].sum(), 1e-7,
                                   rtol=1e-13)
    # Physical mass concentration at 298 K, in ug/m3.
    if temperature == 298.0:
        saturation_mass = vapor_pressure * 101325 / (RGAS / 1000 * 298) * 250e6
        np.testing.assert_allclose(saturation_mass, 1.0224, rtol=1e-4)
