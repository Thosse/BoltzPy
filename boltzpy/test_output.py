import numpy as np
import pytest

import boltzpy.output as bp_o
from boltzpy.test_simulation import SIMULATIONS


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_particle_number(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        spc_group = hdf_group[str(s)]
        for t in range(sim.timing.size):
            state = spc_group["state"][t]
            old_result = spc_group["particle_number"][t]
            new_result = bp_o.particle_number(state, dv)
            assert np.array_equal(old_result, new_result)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_mean_velocity(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        velocities = sim.model.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        for t in range(sim.timing.size):
            state = spc_group["state"][t]
            particle_number = spc_group["particle_number"][t]
            old_result = spc_group["mean_velocity"][t]
            new_result = bp_o.mean_velocity(state,
                                            dv,
                                            velocities,
                                            particle_number)
            assert np.array_equal(old_result, new_result)


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
        for t in range(sim.timing.size):
            state = spc_group["state"][t]
            particle_number = spc_group["particle_number"][t]
            mean_velocity = spc_group["mean_velocity"][t]
            old_result = spc_group["temperature"][t]
            new_result = bp_o.temperature(state,
                                          dv,
                                          velocities,
                                          mass,
                                          particle_number,
                                          mean_velocity)
            assert old_result.shape == new_result.shape
            assert np.array_equal(old_result, new_result)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_momentum(key):
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
            old_result = spc_group["momentum"][t]
            new_result = bp_o.momentum(state,
                                       dv,
                                       velocities,
                                       mass)
            assert np.array_equal(old_result, new_result)


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
        for t in range(sim.timing.size):
            state = spc_group["state"][t]
            old_result = spc_group["energy"][t]
            new_result = bp_o.energy(state,
                                     dv,
                                     velocities,
                                     mass)
            assert np.array_equal(old_result, new_result)


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
        for t in range(sim.timing.size):
            state = spc_group["state"][t]
            old_result = spc_group["momentum_flow"][t]
            new_result = bp_o.momentum_flow(state,
                                            dv,
                                            velocities,
                                            mass)
            assert np.array_equal(old_result, new_result)


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
            assert np.array_equal(old_result, new_result)
