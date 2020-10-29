import numpy as np
import pytest

import boltzpy.output as bp_o
from boltzpy.test_simulation import SIMULATIONS


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_temperature(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    assert sim.model.specimen > 0
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        mass = sim.model.masses[s]
        velocities = sim.model.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        state = spc_group["state"][()]
        number_density = spc_group["particle_number"][()]
        mean_velocity = spc_group["mean_velocity"][()]
        pressure = bp_o.pressure(state, dv, velocities,  mass, mean_velocity)
        old_result = spc_group["temperature"][()]
        new_result = bp_o.temperature(pressure, number_density)
        assert old_result.shape == new_result.shape
        assert np.allclose(old_result, new_result)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_energy(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        mass = sim.model.masses[s]
        velocities = sim.model.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        state = spc_group["state"][()]
        old_result = spc_group["energy"][()]
        new_result = bp_o.energy_density(state, dv, velocities, mass)
        assert np.allclose(old_result, new_result)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_momentum_flow(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        mass = sim.model.masses[s]
        velocities = sim.model.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        state = spc_group["state"][()]
        old_result = spc_group["momentum_flow"][()]
        new_result = bp_o.momentum_flow(state, dv,   velocities,  mass)
        assert np.allclose(old_result, new_result)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_energy_flow(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        mass = sim.model.masses[s]
        velocities = sim.model.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        for t in range(sim.timing.size):
            state = spc_group["state"][t]
            old_result = spc_group["energy_flow"][t]
            new_result = bp_o.energy_flow(state,
                                          dv,
                                          velocities,
                                          mass)
            assert np.allclose(old_result, new_result)
