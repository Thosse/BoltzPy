import pytest
import os
import h5py
import numpy as np

import boltzpy.helpers.tests as test_helper
import boltzpy as bp
from boltzpy.test_grid import GRIDS
from boltzpy.test_model import MODELS
from boltzpy.test_geometry import GEOMETRIES
from boltzpy.test_rule import RULES

FILE = test_helper.DIRECTORY + 'TestResults.hdf5'
TEST_ELEMENTS = {**GRIDS, **GEOMETRIES, **MODELS, **RULES}

TEST_ATTRIBUTES = list()
for (key_elem, elem) in TEST_ELEMENTS.items():
    TEST_ATTRIBUTES.extend([(key_elem, attr) for attr in elem.attributes()])


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset stored test results. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    with h5py.File(file_address, mode="w") as file:
        for (key, item) in TEST_ELEMENTS.items():
            assert isinstance(item, bp.BaseClass)
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
        assert isinstance(old, bp.BaseClass)
        assert isinstance(new, bp.BaseClass)
        assert old == new


@pytest.mark.parametrize("key, attr", TEST_ATTRIBUTES)
def test_attributes(key, attr):
    with h5py.File(FILE, mode="r") as file:
        read_dict = bp.BaseClass.read_parameters_from_hdf_file(file[key], attr)
        old = read_dict[attr]
        new = TEST_ELEMENTS[key].__getattribute__(attr)
        if isinstance(new, np.ndarray) and (new.dtype == float):
            assert np.allclose(old, new)
        else:
            assert np.all(old == new)
