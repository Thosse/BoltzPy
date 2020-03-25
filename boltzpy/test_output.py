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
    hdf_group = hdf_file["results"]
    for (s, species_name) in enumerate(simulation.s.names):
        dv = simulation.sv.vGrids[s].physical_spacing
        spc_group = hdf_group[species_name]
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
    hdf_group = hdf_file["results"]
    for (s, species_name) in enumerate(simulation.s.names):
        dv = simulation.sv.vGrids[s].physical_spacing
        velocities = simulation.sv.vGrids[s].pG
        spc_group = hdf_group[species_name]
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
    hdf_group = hdf_file["results"]
    assert len(simulation.s.names) > 0
    for (s, species_name) in enumerate(simulation.s.names):
        dv = simulation.sv.vGrids[s].physical_spacing
        mass = simulation.s.mass[s]
        velocities = simulation.sv.vGrids[s].pG
        spc_group = hdf_group[species_name]
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


