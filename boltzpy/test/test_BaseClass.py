import pytest
import os
import h5py
import numpy as np

import boltzpy.Tools.tests as test_helper
import boltzpy as bp
from boltzpy.test.test_grid import GRIDS
from boltzpy.test.test_model import MODELS
from boltzpy.test.test_geometry import GEOMETRIES
from boltzpy.test.test_rule import RULES

FILE = bp.TEST_DIR + '/TestResults.hdf5'
TMP_FILE = bp.TEST_DIR + '/_tmp_.hdf5'

TEST_ELEMENTS = {**GRIDS, **GEOMETRIES, **MODELS, **RULES}

TEST_ATTRIBUTES = list()
for (key_elem, elem) in TEST_ELEMENTS.items():
    TEST_ATTRIBUTES.extend([(key_elem, attr) for attr in elem.attributes()])


def setup_file(file_address=FILE):
    with h5py.File(file_address, mode="a") as file:
        # overwrite different classes separately
        for name, elements in zip(["Grids", "Geometries", "Rules", "Models"],
                                  [GRIDS, GEOMETRIES, RULES, MODELS]):
            # ask before overwriting any of Test Data
            if file_address == FILE:
                reply = input("Reset stored {} (yes/no)? ".format(name))
                overwrite = reply == "yes"
            else:
                overwrite = True
            # overwrite all test data of this class
            if overwrite:
                for (key, item) in elements.items():
                    assert isinstance(item, bp.BaseClass)
                    file.require_group(key)
                    item.save(file[key], item.attributes())
            else:
                print("Skipped")
    return


#############################
#           Tests           #
#############################
def test_file_exists():
    assert os.path.exists(FILE), (
        "The test file {} is missing.".format(FILE))


def test_setup_files_keys_did_not_change():
    try:
        setup_file(TMP_FILE)
        # gather all keys in sets
        with h5py.File(FILE, mode="r") as file:
            keys_old = set()
            file.visit(keys_old.add)
        with h5py.File(TMP_FILE, mode="r") as file:
            keys_new = set()
            file.visit(keys_new.add)
        # assert keys are equal
        assert keys_old == keys_new
    finally:
        os.remove(TMP_FILE)



def test_setup_creates_same_file():
    try:
        setup_file(TMP_FILE)
        test_helper.assert_hdf_groups_are_equal(h5py.File(FILE, mode="r"),
                                                h5py.File(TMP_FILE, mode="r"))
    finally:
        os.remove(TMP_FILE)
    return


@pytest.mark.parametrize("key", TEST_ELEMENTS.keys())
def test_hdf5_groups_exist(key):
    assert os.path.exists(FILE)
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


@pytest.mark.parametrize("_, self", TEST_ELEMENTS.items())
def test_loading_saved_state_yields_equal_instance(_, self):
    assert isinstance(self, bp.BaseClass)
    try:
        file = h5py.File(TMP_FILE, mode="w")
        self.save(h5py.File(TMP_FILE))
        other = bp.BaseClass.load(file)
        assert self == other
    finally:
        os.remove(TMP_FILE)


@pytest.mark.parametrize("key, attr", TEST_ATTRIBUTES)
def test_attributes(key, attr):
    with h5py.File(FILE, mode="r") as file:
        old = bp.BaseClass.load_attributes(file[key], attr)
        new = TEST_ELEMENTS[key].__getattribute__(attr)
        if isinstance(new, np.ndarray):
            assert old.shape == new.shape
            assert old.dtype == new.dtype
        if isinstance(new, np.ndarray) and (new.dtype == float):
            assert np.allclose(old, new)
        else:
            assert np.all(old == new)
