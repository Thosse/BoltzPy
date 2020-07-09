import pytest
import os
import h5py

import boltzpy.helpers.tests as test_helper
import boltzpy as bp

###################################
#           Setup Cases           #
###################################
FILE = test_helper.DIRECTORY + 'Models.hdf5'
MODELS = dict()
MODELS["2D_small/Model"] = bp.SVGrid(
    masses=[2, 3],
    shapes=[[5, 5], [7, 7]],
    delta=1/8,
    spacings=[6, 4],
    collision_factors=[[50, 50], [50, 50]])
MODELS["equalSpacing/Model"] = bp.SVGrid(
    masses=[1, 2],
    shapes=[[7, 7], [7, 7]],
    delta=1/7,
    spacings=[2, 2],
    collision_factors=[[50, 50], [50, 50]])
MODELS["equalMass/Model"] = bp.SVGrid(
    masses=[3, 3],
    shapes=[[5, 5], [5, 5]],
    delta=3/8,
    spacings=[2, 2],
    collision_factors=[[50, 50], [50, 50]])
MODELS["evenShapes/Model"] = bp.SVGrid(
    masses=[2, 3],
    shapes=[[4, 4], [6, 6]],
    delta=1/12,
    spacings=[2, 2],
    collision_factors=[[50, 50], [50, 50]])
MODELS["mixed_spacing/Model"] = bp.SVGrid(
    masses=[3, 4],
    shapes=[[6, 6], [7, 7]],
    delta=1/14,
    spacings=[2, 2],
    collision_factors=[[50, 50], [50, 50]])


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the model test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    file = h5py.File(file_address, mode="w")
    for (key, item) in MODELS.items():
        assert isinstance(item, bp.SVGrid)
        file.create_group(key)
        item.save(file[key])
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


@pytest.mark.parametrize("key", MODELS.keys())
def test_hdf5_groups_exist(key):
    file = h5py.File(FILE, mode="r")
    assert key in file.keys(), (
        "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", MODELS.keys())
def test_load_from_file(key):
    file = h5py.File(FILE, mode="r")
    hdf_group = file[key]
    old = bp.SVGrid.load(hdf_group)
    new = MODELS[key]
    assert isinstance(old, bp.SVGrid)
    assert isinstance(new, bp.SVGrid)
    assert old == new, (
        "\n{}\nis not equal to\n\n{}".format(old, new)
    )


# Todo proper test of find_index and get_specimen seems hard, implement differently?
#  Implement collisions only for specimen tuples
#  -> no need for get_specimen
#  -> no need for find_index

# Todo __get__ für model -> nach species index
#  + Grid __get__ nach index -> arbeiten mit lokalen indices möglich
#  + für (get_)idx(specimen, indices) hinzu -Y generiert globale indices
