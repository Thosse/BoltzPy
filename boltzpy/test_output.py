import numpy as np
import pytest
import h5py

import boltzpy.testcase as bp_t
import boltzpy as bp
import boltzpy.output as bp_o


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_particle_number(tf):
    simulation = bp.Simulation.load(tf)
    # load results
    hdf_file = h5py.File(tf, mode="r")
    hdf_group = hdf_file["Results"]
    for s in simulation.sv.species:
        dv = simulation.sv.vGrids[s].physical_spacing
        spc_group = hdf_group[str(s)]
        for t in range(simulation.t.size):
            state = spc_group["state"][t]
            old_result = spc_group["particle_number"][t]
            new_result = bp_o.particle_number(state, dv)
            assert np.array_equal(old_result, new_result)


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_mean_velocity(tf):
    simulation = bp.Simulation.load(tf)
    # load results
    hdf_file = h5py.File(tf, mode="r")
    hdf_group = hdf_file["Results"]
    for s in simulation.sv.species:
        dv = simulation.sv.vGrids[s].physical_spacing
        velocities = simulation.sv.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        for t in range(simulation.t.size):
            state = spc_group["state"][t]
            particle_number = spc_group["particle_number"][t]
            old_result = spc_group["mean_velocity"][t]
            new_result = bp_o.mean_velocity(state,
                                            dv,
                                            velocities,
                                            particle_number)
            assert np.array_equal(old_result, new_result)


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_temperature(tf):
    simulation = bp.Simulation.load(tf)
    # load results
    hdf_file = h5py.File(tf, mode="r")
    hdf_group = hdf_file["Results"]
    assert simulation.sv.specimen > 0
    for s in simulation.sv.species:
        dv = simulation.sv.vGrids[s].physical_spacing
        mass = simulation.sv.masses[s]
        velocities = simulation.sv.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        for t in range(simulation.t.size):
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


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_momentum(tf):
    simulation = bp.Simulation.load(tf)
    # load results
    hdf_file = h5py.File(tf, mode="r")
    hdf_group = hdf_file["Results"]
    for s in simulation.sv.species:
        dv = simulation.sv.vGrids[s].physical_spacing
        mass = simulation.sv.masses[s]
        velocities = simulation.sv.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        for t in range(simulation.t.size):
            state = spc_group["state"][t]
            old_result = spc_group["momentum"][t]
            new_result = bp_o.momentum(state,
                                       dv,
                                       velocities,
                                       mass)
            assert np.array_equal(old_result, new_result)


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_energy(tf):
    simulation = bp.Simulation.load(tf)
    # load results
    hdf_file = h5py.File(tf, mode="r")
    hdf_group = hdf_file["Results"]
    for s in simulation.sv.species:
        dv = simulation.sv.vGrids[s].physical_spacing
        mass = simulation.sv.masses[s]
        velocities = simulation.sv.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        for t in range(simulation.t.size):
            state = spc_group["state"][t]
            old_result = spc_group["energy"][t]
            new_result = bp_o.energy(state,
                                     dv,
                                     velocities,
                                     mass)
            assert np.array_equal(old_result, new_result)


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_momentum_flow(tf):
    simulation = bp.Simulation.load(tf)
    # load results
    hdf_file = h5py.File(tf, mode="r")
    hdf_group = hdf_file["Results"]
    for s in simulation.sv.species:
        dv = simulation.sv.vGrids[s].physical_spacing
        mass = simulation.sv.masses[s]
        velocities = simulation.sv.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        for t in range(simulation.t.size):
            state = spc_group["state"][t]
            old_result = spc_group["momentum_flow"][t]
            new_result = bp_o.momentum_flow(state,
                                            dv,
                                            velocities,
                                            mass)
            assert np.array_equal(old_result, new_result)


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_energy_flow(tf):
    simulation = bp.Simulation.load(tf)
    # load results
    hdf_file = h5py.File(tf, mode="r")
    hdf_group = hdf_file["Results"]
    for s in simulation.sv.species:
        dv = simulation.sv.vGrids[s].physical_spacing
        mass = simulation.sv.masses[s]
        velocities = simulation.sv.vGrids[s].pG
        spc_group = hdf_group[str(s)]
        for t in range(simulation.t.size):
            state = spc_group["state"][t]
            old_result = spc_group["energy_flow"][t]
            new_result = bp_o.energy_flow(state,
                                          dv,
                                          velocities,
                                          mass)
            assert np.array_equal(old_result, new_result)
