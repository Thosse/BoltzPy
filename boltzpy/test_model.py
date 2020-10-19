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
        assert old == new

@pytest.mark.parametrize("key", MODELS.keys())
@pytest.mark.parametrize("attribute", bp.Model.attributes())
def test_attributes(attribute, key):
    with h5py.File(FILE, mode="r") as file:
        old = file[key][attribute][()]
        new = MODELS[key].__getattribute__(attribute)
        if isinstance(new, np.ndarray) and (new.dtype == float):
            assert np.allclose(old, new)
        else:
            assert np.all(old == new)


@pytest.mark.parametrize("key", MODELS.keys())
def test_get_spc_on_shuffled_grid(key):
    model = MODELS[key]
    # setup the species each gridpoint/velocity belongs to, for comparison
    species = np.zeros(model.size)
    for s in model.species:
        beg, end = model.index_offset[s:s+2]
        species[beg:end] = s
    # shuffle velocities/species
    rng = np.random.default_rng()
    shuffled_idx = rng.permutation(model.size)
    # test for 0d arrays/elements
    for idx in shuffled_idx:
        assert model.get_spc(idx).ndim == 0
        assert model.get_spc(idx) == species[idx]
    # test for different ndims >= 1
    for ndmin in range(1, 6):
        shuffled_idx = np.array(shuffled_idx, ndmin=ndmin)
        shuffles_spc = species[shuffled_idx]
        assert np.all(model.get_spc(shuffled_idx) == shuffles_spc)


@pytest.mark.parametrize("key", MODELS.keys())
def test_get_idx_on_shuffled_grid_with(key):
    model = MODELS[key]
    rng = np.random.default_rng()
    # test 0d species arrays
    for s in model.species:
        beg, end = model.index_offset[s:s+2]
        indices = np.arange(beg, end)
        shuffle = rng.permutation(indices.size)
        shuffled_idx = indices[shuffle]
        # test 1d velocity arrays
        for idx in shuffled_idx:
            assert np.all(model.get_idx(s, model.iMG[idx]) == idx)
        # test 2d velocity arrays (n_Vels x dim)
        shuffled_vels = model.iMG[shuffled_idx]
        assert shuffled_vels.ndim == 2
        assert np.all(model.get_idx(s, shuffled_vels) == shuffled_idx)
        # test 3d velocity arrays (n_vels x 1 x dim)
        shuffled_idx = indices[shuffle][:, np.newaxis]
        shuffled_vels = model.iMG[shuffled_idx]
        assert shuffled_idx.ndim == 2
        assert shuffled_vels.ndim == 3
        assert np.all(model.get_idx(s, shuffled_vels) == shuffled_idx)
    # Todo add tests for 1d species array
    # test 1d species arrays
    if model.specimen <= 2:
        return

# Todo/Idea: Implement collisions only for specimen tuples
#  -> no need for get_idx  with its warnings

# Todo __get__ für model -> nach species index
#  + Grid __get__ nach index -> arbeiten mit lokalen indices möglich
#  + für (get_)idx(specimen, indices) hinzu -Y generiert globale indices
