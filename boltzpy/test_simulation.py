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
    GRIDS["Grid/2D_small"],
    GEOMETRIES["Geometry/2D_small"],
    MODELS["Model/2D_small"],
    _open_file["2D_small/Simulation"],
    True)
SIMULATIONS["equalMass/Simulation"] = bp.Simulation(
    GRIDS["Grid/equalMass"],
    GEOMETRIES["Geometry/equalMass"],
    MODELS["Model/equalMass"],
    _open_file["equalMass/Simulation"],
    True)

SAVED_MOMENTS = [
    "number_density",
    "momentum",
    "mean_velocity",
    "energy_density",
    "temperature",
    "momentum_flow",
    "energy_flow"]


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
def test_old_and_new_states_are_equal(key):
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


@pytest.mark.parametrize("key", SIMULATIONS.keys())
@pytest.mark.parametrize("moment", SAVED_MOMENTS)
def test_old_and_new_moments_are_equal(key, moment):
    with h5py.File(FILE, mode="r") as file_old:
        with h5py.File(test_helper.TMP_FILE, mode="r") as file_new:
            keys_old = file_old[key]["results"].keys()
            keys_new = file_new[key]["results"].keys()
            assert keys_old == keys_new
            for s in file_old[key]["results"].keys():
                moment_old = file_old[key]["results"][s][moment][()]
                moment_new = file_new[key]["results"][s][moment][()]
                assert np.allclose(moment_old, moment_new), (
                    "\n{}\nis not equal to\n\n{}".format(moment_old - moment_new, 0))


@pytest.mark.parametrize("key", SIMULATIONS.keys())
@pytest.mark.parametrize("moment", SAVED_MOMENTS)
def test_computing_moments_on_old_state_gives_old_results(key, moment):
    with h5py.File(FILE, mode="r") as file:
        hdf_group = file[key]
        sim = SIMULATIONS[key]
        model = sim.model
        compute_moment = model.__getattribute__("cmp_" + moment)
        for s in model.species:
            spc_group = hdf_group["results"][str(s)]
            state = spc_group["state"][()]
            old_result = spc_group[moment][()]
            new_result = compute_moment(state, s)
            assert np.allclose(old_result, new_result)


# the file is used in more tests, this is a simple hack to delete it after use
def test_teardown_tmp_file():
    os.remove(test_helper.TMP_FILE)
