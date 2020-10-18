import pytest
import os
import h5py
import numpy as np

import boltzpy.helpers.tests as test_helper
import boltzpy as bp
from boltzpy.test_rule import RULES


###################################
#           Setup Cases           #
###################################
FILE = test_helper.DIRECTORY + 'Geometries.hdf5'
GEOMETRIES = dict()
GEOMETRIES["2D_small/Geometry"] = bp.Geometry(
    (10,),
    0.5,
    [RULES["2D_small/LeftConstant"],
     RULES["2D_small/Interior"],
     RULES["2D_small/RightBoundary"]])
GEOMETRIES["equalMass/Geometry"] = bp.Geometry(
    (10,),
    0.5,
    [RULES["equalMass/LeftBoundary"],
     RULES["equalMass/LeftInterior"],
     RULES["equalMass/RightInterior"],
     RULES["equalMass/RightBoundary"]])


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the geometries test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    with h5py.File(file_address, mode="w") as file:
        for (key, item) in GEOMETRIES.items():
            assert isinstance(item, bp.Geometry)
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


@pytest.mark.parametrize("key", GEOMETRIES.keys())
def test_hdf5_groups_exist(key):
    with h5py.File(FILE, mode="r") as file:
        assert key in file.keys(), (
            "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", GEOMETRIES.keys())
def test_load_from_file(key):
    with h5py.File(FILE, mode="r") as file:
        hdf_group = file[key]
        old = bp.Geometry.load(hdf_group)
        new = GEOMETRIES[key]
        assert isinstance(old, bp.Geometry)
        assert isinstance(new, bp.Geometry)
        assert old == new, (
            "\n{}\nis not equal to\n\n{}".format(old, new))


@pytest.mark.parametrize("key", GEOMETRIES.keys())
@pytest.mark.parametrize("attribute", bp.Geometry.attributes())
def test_attributes(attribute, key):
    with h5py.File(FILE, mode="r") as file:
        if attribute == "rules":
            old = np.empty(file[key]["rules"].attrs["size"],
                           dtype=bp.Rule)
            for r in range(old.size):
                old[r] = bp.Rule.load(file[key]["rules"][str(r)])
        else:
            old = file[key][attribute][()]
        new = GEOMETRIES[key].__getattribute__(attribute)
        assert np.all(old == new)