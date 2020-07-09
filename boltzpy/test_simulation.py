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
# TODO remove this, when removing Scheme:
scheme_convergent = bp.Scheme(
    "FirstOrder",
    "FiniteDifferences_FirstOrder",
    np.array([0.0, 0.0]),
    "Convergent",
    # "UniformComplete",
    "EulerScheme")
scheme_complete = bp.Scheme(
    "FirstOrder",
    "FiniteDifferences_FirstOrder",
    np.array([0.0, 0.0]),
    # "Convergent",
    "UniformComplete",
    "EulerScheme")

# TODO remove this, when moving Collisions into sv/model:
coll_small = bp.Collisions()
coll_small.setup(scheme_complete, MODELS["2D_small/Model"])
coll_equal = bp.Collisions()
coll_equal.setup(scheme_convergent, MODELS["equalMass/Model"])

DIRECTORY = __file__[:-26] + 'test_data/'
FILE = DIRECTORY + 'Simulations.hdf5'
TMP_FILE = DIRECTORY + '_tmp_.hdf5'
_open_file = h5py.File(FILE, mode="r")
SIMULATIONS = dict()
SIMULATIONS["2D_small/Simulation"] = bp.Simulation(
    GRIDS["2D_small/timing"],
    GEOMETRIES["2D_small/Geometry"],
    MODELS["2D_small/Model"],
    coll_small,
    scheme_complete,
    True,
    _open_file["2D_small/Simulation"])
SIMULATIONS["equalMass/Simulation"] = bp.Simulation(
    GRIDS["equalMass/timing"],
    GEOMETRIES["equalMass/Geometry"],
    MODELS["equalMass/Model"],
    coll_equal,
    scheme_convergent,
    True,
    _open_file["equalMass/Simulation"])


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the simulations test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return
    _open_file.close()
    file = h5py.File(file_address, mode="w")
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
    setup_file(TMP_FILE)
    test_helper.assert_files_are_equal([FILE, TMP_FILE])
    return


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_hdf5_groups_exist(key):
    file = h5py.File(FILE, mode="r")
    assert key in file.keys(), (
        "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_load_from_file(key):
    file = h5py.File(FILE, mode="r")
    hdf_group = file[key]
    old = bp.Simulation.load(hdf_group)
    new = SIMULATIONS[key]
    assert isinstance(old, bp.Simulation)
    assert isinstance(new, bp.Simulation)
    assert old == new, (
        "\n{}\nis not equal to\n\n{}".format(old, new)
    )


@pytest.mark.parametrize("key", SIMULATIONS.keys())
def test_computed_state_is_equal(key):
    file_old = h5py.File(FILE, mode="r")
    file_new = h5py.File(TMP_FILE, mode="r")
    assert file_old[key]["Results"].keys() == file_new[key]["Results"].keys()
    for s in file_old[key]["Results"].keys():
        state_old = file_old[key]["Results"][s]["state"][()]
        state_new = file_new[key]["Results"][s]["state"][()]
        assert np.array_equal(state_old, state_new), (
            "\n{}\nis not equal to\n\n{}".format(state_old, state_new))
