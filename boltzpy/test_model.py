import pytest
import os
import h5py
import numpy as np

import boltzpy.helpers.tests as test_helper
import boltzpy as bp

###################################
#           Setup Cases           #
###################################
FILE = test_helper.DIRECTORY + 'Models.hdf5'
MODELS = dict()
MODELS["2D_small/Model"] = bp.Model(
    masses=[2, 3],
    shapes=[[5, 5], [7, 7]],
    delta=1/8,
    spacings=[6, 4],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")
MODELS["2D_small_convergent/Model"] = bp.Model(
    masses=[2, 3],
    shapes=[[5, 5], [7, 7]],
    delta=1/8,
    spacings=[6, 4],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="convergent",
    algorithm_weights="uniform")
# Todo This might lead to weird results! Check this!
MODELS["equalSpacing/Model"] = bp.Model(
    masses=[1, 2],
    shapes=[[7, 7], [7, 7]],
    delta=1/7,
    spacings=[2, 2],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")
MODELS["equalMass/Model"] = bp.Model(
    masses=[3, 3],
    shapes=[[5, 5], [5, 5]],
    delta=3/8,
    spacings=[2, 2],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")
# Todo Allow equal spacing
# MODELS["evenShapes/Model"] = bp.Model(
#     masses=[2, 3],
#     shapes=[[4, 4], [6, 6]],
#     delta=1/12,
#     spacings=[2, 2],
#     collision_factors=[[50, 50], [50, 50]],
#     algorithm_relations="all",
#     algorithm_weights="uniform")
MODELS["mixed_spacing/Model"] = bp.Model(
    masses=[3, 4],
    shapes=[[6, 6], [7, 7]],
    delta=1/14,
    spacings=[8, 6],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")
MODELS["3D_small/Model"] = bp.Model(
    masses=[2, 3],
    shapes=[[2, 2, 2], [2, 2, 2]],
    delta=1/8,
    spacings=[6, 4],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the model test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    with h5py.File(file_address, mode="w") as file:
        for (key, item) in MODELS.items():
            assert isinstance(item, bp.Model)
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


@pytest.mark.parametrize("key", MODELS.keys())
def test_hdf5_groups_exist(key):
    with h5py.File(FILE, mode="r") as file:
        assert key in file.keys(), (
            "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", MODELS.keys())
def test_load_from_file(key):
    with h5py.File(FILE, mode="r") as file:
        hdf_group = file[key]
        old = bp.Model.load(hdf_group)
        new = MODELS[key]
        assert isinstance(old, bp.Model)
        assert isinstance(new, bp.Model)
        assert old == new, (
            "\n{}\nis not equal to\n\n{}".format(old, new))


@pytest.mark.parametrize("key", MODELS.keys())
@pytest.mark.parametrize("attribute", bp.Model.attributes())
def test_attributes(attribute, key):
    with h5py.File(FILE, mode="r") as file:
        old = file[key][attribute][()]
        new = MODELS[key].__getattribute__(attribute)
        assert np.all(old == new)

# Todo proper test of find_index and get_specimen seems hard, implement differently?
#  Implement collisions only for specimen tuples
#  -> no need for get_specimen
#  -> no need for find_index

# Todo __get__ für model -> nach species index
#  + Grid __get__ nach index -> arbeiten mit lokalen indices möglich
#  + für (get_)idx(specimen, indices) hinzu -Y generiert globale indices
