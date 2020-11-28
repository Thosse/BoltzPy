import pytest
import os
import h5py
import numpy as np

import boltzpy.helpers.tests as test_helper
import boltzpy as bp
from boltzpy.test_grid import GRIDS
# from boltzpy.test_model import MODELS
# from boltzpy.test_geometry import GEOMETRIES

FILE = test_helper.DIRECTORY + 'TestResults.hdf5'
TEST_ELEMENTS = {**GRIDS}

TEST_ATTRIBUTES = set()
for (key, item) in TEST_ELEMENTS.items():
    TEST_ATTRIBUTES.update({(key, attr) for attr in item.attributes()})


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset stored test results. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    with h5py.File(file_address, mode="w") as file:
        for (key, item) in TEST_ELEMENTS.items():
            assert isinstance(item, bp.Grid)
            file.create_group(key)
            item.save(file[key], True)
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
    os.remove(test_helper.TMP_FILE)
    return


@pytest.mark.parametrize("key", TEST_ELEMENTS.keys())
def test_hdf5_groups_exist(key):
    with h5py.File(FILE, mode="r") as file:
        assert key in file.keys(), (
            "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", TEST_ELEMENTS.keys())
def test_load_from_file(key):
    with h5py.File(FILE, mode="r") as file:
        old = bp.BaseClass.load(file[key])
        new = TEST_ELEMENTS[key]
        assert isinstance(old, bp.Grid)
        assert isinstance(new, bp.Grid)
        assert old == new


@pytest.mark.parametrize("key, attr", TEST_ATTRIBUTES)
def test_attributes(key, attr):
    with h5py.File(FILE, mode="r") as file:
        old = file[key][attr][()]
        new = GRIDS[key].__getattribute__(attr)
        if isinstance(new, np.ndarray) and (new.dtype == float):
            assert np.allclose(old, new)
        else:
            assert np.all(old == new)
