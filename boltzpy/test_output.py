import numpy as np
import pytest

import boltzpy.output as bp_o
from boltzpy.test_simulation import SIMULATIONS


def test_proj():
    for _ in range(10):
        dim = 2 + int(2 * np.random.rand(1))
        assert dim in [2, 3]
        # set up random vectors
        ndim = int(6 * np.random.random(1))
        shape = np.array(1 + 10 * np.random.random(ndim), dtype=int)
        shape = tuple(shape) + (dim,)
        size = np.prod(shape)
        vectors = np.random.random(size).reshape(shape)
        # Test that proj of ith unit vector is the ith components
        for i in range(dim):
            direction = np.zeros(dim)
            direction[i] = 1
            result = bp_o.proj(vectors, direction)
            goal = np.zeros(result.shape)
            goal[..., i] = vectors[..., i]
            assert np.allclose(result, goal)
        # for random directions
        # test that the sum of proj of all orthogonal directions is the original
        direction = np.zeros((dim, dim))
        direction[0] = np.random.random(dim)
        direction[1, 0:2] = [-direction[0, 1], direction[0, 0]]
        if dim == 3:
            direction[2] = [-np.prod(direction[0, [0, 2]]),
                            -np.prod(direction[0, [1, 2]]),
                            np.sum(direction[0, 0:2]**2)]
        result = np.zeros(vectors.shape)
        for i in range(dim):
            result[:] += bp_o.proj(vectors, direction[i])
        assert np.allclose(result, vectors)
    return


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_particle_number(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        spc_group = hdf_group[str(s)]
        state = spc_group["state"][()]
        old_result = spc_group["particle_number"][()]
        new_result = bp_o.number_density(state, dv)
        assert np.allclose(old_result, new_result)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_mean_velocity(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        velocities = sim.model.vGrids[s].pG
        mass = sim.model.masses[s]
        spc_group = hdf_group[str(s)]
        state = spc_group["state"][()]
        particle_number = spc_group["particle_number"][()]
        old_result = spc_group["mean_velocity"][()]
        momentum = bp_o.momentum(state, dv, velocities, mass)
        mass_density = particle_number * mass
        new_result = bp_o.mean_velocity(momentum, mass_density)
        assert np.allclose(old_result, new_result)


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
def test_momentum(key):
    sim = SIMULATIONS[key]
    # load results
    hdf_group = sim.file["results"]
    for s in sim.model.species:
        dv = sim.model.vGrids[s].physical_spacing
        mass = sim.model.masses[s]
        velocities = sim.model.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        state = spc_group["state"][()]
        old_result = spc_group["momentum"][()]
        new_result = bp_o.momentum(state, dv, velocities, mass)
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
            assert np.array_equal(old_result, new_result)
