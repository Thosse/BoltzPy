import pytest
import os
import h5py
import numpy as np

import boltzpy.helpers.tests as test_helper
import boltzpy as bp
from boltzpy.test_grid import GRIDS
from boltzpy.test_model import MODELS
from boltzpy.test_geometry import GEOMETRIES


###################################
#           Setup Cases           #
###################################

FILE = test_helper.DIRECTORY + 'Simulations.hdf5'
_open_file = h5py.File(FILE, mode="r")
SIMULATIONS = dict()
SIMULATIONS["2D_small/Simulation"] = bp.Simulation(
    GRIDS["2D_small/timing"],
    GEOMETRIES["2D_small/Geometry"],
    MODELS["2D_small/Model"],
    _open_file["2D_small/Simulation"],
    True)
SIMULATIONS["equalMass/Simulation"] = bp.Simulation(
    GRIDS["equalMass/timing"],
    GEOMETRIES["equalMass/Geometry"],
    MODELS["equalMass/Model"],
    _open_file["equalMass/Simulation"],
    True)


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the simulations test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return
        else:
            _open_file.close()

    with h5py.File(file_address, mode="w") as file:
        for (key, item) in SIMULATIONS.items():
            assert isinstance(item, bp.Simulation)
            file.create_group(key)
            item.compute(file[key])
            item.file = file[key]
    return


#############################
#           Tests           #
#############################
def test_file_exists():
    assert os.path.exists(FILE), (
        "The test file {} is missing.".format(FILE))


def test_setup_creates_same_file():
    setup_file(test_helper.TMP_FILE)
    test_helper.assert_files_are_equal([FILE, test_helper.TMP_FILE])
    return


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_hdf5_groups_exist(key):
    with h5py.File(FILE, mode="r") as file:
        assert key in file.keys(), (
            "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_load_from_file(key):
    with h5py.File(FILE, mode="r") as file:
        hdf_group = file[key]
        old = bp.Simulation.load(hdf_group)
        new = SIMULATIONS[key]
        assert isinstance(old, bp.Simulation)
        assert isinstance(new, bp.Simulation)
        assert old == new


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_computed_state_is_equal(key):
    with h5py.File(FILE, mode="r") as file_old:
        with h5py.File(test_helper.TMP_FILE, mode="r") as file_new:
            keys_old = file_old[key]["results"].keys()
            keys_new = file_new[key]["results"].keys()
            assert keys_old == keys_new
            for s in file_old[key]["results"].keys():
                state_old = file_old[key]["results"][s]["state"][()]
                state_new = file_new[key]["results"][s]["state"][()]
                assert np.allclose(state_old, state_new), (
                    "\n{}\nis not equal to\n\n{}".format(state_old, state_new))


# the file is used in more tests, this is a simple hack to delete it after use
def test_teardown_tmp_file():
    os.remove(test_helper.TMP_FILE)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_particle_number(key):
    with h5py.File(FILE, mode="r") as file:
        hdf_group = file[key]
        sim = bp.Simulation.load(hdf_group)
        model = sim.model
        for s in sim.model.species:
            spc_group = hdf_group["results"][str(s)]
            state = spc_group["state"][()]
            old_result = spc_group["particle_number"][()]
            new_result = model.number_density(state, s)
            assert np.allclose(old_result, new_result)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_momentum(key):
    with h5py.File(FILE, mode="r") as file:
        hdf_group = file[key]
        sim = bp.Simulation.load(hdf_group)
        model = sim.model
        for s in sim.model.species:
            spc_group = hdf_group["results"][str(s)]
            state = spc_group["state"][()]
            old_result = spc_group["momentum"][()]
            new_result = model.momentum(state, s)
            assert np.allclose(old_result, new_result)


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_mean_velocity(key):
    with h5py.File(FILE, mode="r") as file:
        hdf_group = file[key]
        sim = bp.Simulation.load(hdf_group)
        model = sim.model
        for s in sim.model.species:
            spc_group = hdf_group["results"][str(s)]
            state = spc_group["state"][()]
            old_result = spc_group["mean_velocity"][()]
            momentum = model.momentum(state, s)
            mass_density = model.mass_density(state, s)
            new_result = model.mean_velocity(momentum, mass_density)
            assert np.allclose(old_result, new_result)
