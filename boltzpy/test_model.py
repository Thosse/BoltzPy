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
        read_dict = bp.BaseClass.read_parameters_from_hdf_file(file[key], attribute)
        old = read_dict[attribute]
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


def assert_all_moments_are_zero(model, state):
    # number density of each species stays the same
    for s in model.species:
        number_density = model.number_density(state, s)
        assert np.isclose(number_density, 0)
    # assert total number/mass stays the same (should be unnecessary)
    number_density = model.number_density(state)
    assert np.isclose(number_density, 0)
    mass_density = model.mass_density(state)
    assert np.isclose(mass_density, 0)
    momentum = model.momentum(state)
    assert np.allclose(momentum, 0)
    energy = model.energy_density(state)
    assert np.isclose(energy, 0)


@pytest.mark.parametrize("key", MODELS.keys())
def test_invariance_of_moments_under_collision_operator(key):
    model = MODELS[key]
    for _ in range(100):
        state = np.random.random(model.size)
        # test that all momenta of the collision_differences are zero
        # since these are additive this tests that the momenta are collision invariant
        collision_differences = model.collision_operator(state)
        assert_all_moments_are_zero(model, collision_differences)


@pytest.mark.parametrize("key", MODELS.keys())
def test_maxwellians_are_invariant_unter_collision_operator(key):
    model = MODELS[key]
    # choose random mean velocity and temperature parameters
    mass_array = model.get_array(model.masses)
    for _ in range(100):
        max_v = model.maximum_velocity
        mean_velocity = 2 * max_v * np.random.random(model.ndim) - max_v
        temperature = 100 * np.random.random() + 1
        # use parameters directly, dont initialize here (might lead to errors, definitely complicates things!)
        state = model.maxwellian(velocities=model.velocities,
                                 temperature=temperature,
                                 mass=mass_array,
                                 mean_velocity=mean_velocity)
        collision_differences = model.collision_operator(state)
        assert np.allclose(collision_differences, 0)


def test_project_velocities():
    for _ in range(10):
        dim = 2 + int(2 * np.random.rand(1))
        assert dim in [2, 3]
        # set up random vectors
        ndim = int(6 * np.random.random(1))
        shape = np.array(1 + 10 * np.random.random(ndim), dtype=int)
        shape = tuple(shape) + (dim,)
        size = np.prod(shape)
        vectors = np.random.random(size).reshape(shape)
        # Test that proj of ith unit vector is the ith components
        for i in range(dim):
            direction = np.zeros(dim)
            length = 10 * np.random.random(1)
            direction[i] = length
            result = bp.Model.project_velocities(vectors, direction)
            angle = direction / np.linalg.norm(direction)
            shape = result.shape
            size = result.size
            result = result.reshape((size, 1))
            result = result * angle[np.newaxis, :]
            result = result.reshape(shape + (dim,))
            goal = np.zeros(result.shape)
            goal[..., i] = vectors[..., i]
            assert np.allclose(result, goal)
        # for random directions
        # test that the sum of proj of all orthogonal directions is the original
        direction = np.zeros((dim, dim))
        length = 10 * np.random.random(1)
        direction[0] = length * np.random.random(dim)
        direction[1, 0:2] = [-direction[0, 1], direction[0, 0]]
        if dim == 3:
            direction[2] = [-np.prod(direction[0, [0, 2]]),
                            -np.prod(direction[0, [1, 2]]),
                            np.sum(direction[0, 0:2]**2)]
        result = np.zeros(vectors.shape)
        for i in range(dim):
            res = bp.Model.project_velocities(vectors, direction[i])
            angle = direction[i] / np.linalg.norm(direction[i])
            angle = angle.reshape((1, dim))
            shape = res.shape
            size = res.size
            res = res.reshape((size, 1))
            res = res * angle
            result[:] += res.reshape(shape + (dim,))
        assert np.allclose(result, vectors)
    return


@pytest.mark.parametrize("key", MODELS.keys())
def test_mf_orthogonal_stress_is_orthogonal(key):
    model = MODELS[key]
    # moment functions are only orthogonal if the mean velocity is 0,
    # this is due to discretization effects!
    mean_velocity = np.zeros(model.ndim, dtype=float)
    mass_array = model.get_array(model.masses)
    for _ in range(10):
        # choose random temperature
        temperature = 100 * np.random.random() + 1
        # use parameters directly, dont initialize here (might lead to errors, definitely complicates things!)
        state = model.maxwellian(velocities=model.velocities,
                                 temperature=temperature,
                                 mass=mass_array,
                                 mean_velocity=mean_velocity)
        # choose random direction1
        for __ in range(10):
            direction1 = np.random.random(model.ndim)
            # test parallel stress mf
            mf = model.mf_orthogonal_stress(mean_velocity,
                                            direction1,
                                            direction1)
            assert_all_moments_are_zero(model, mf)
            assert_all_moments_are_zero(model, mf * state)
            # ro test nonparallel stress, choose simple orthogonal direction
            direction2 = np.zeros(model.ndim)
            direction2[:2] = (-direction1[1], direction1[0])
            mf = model.mf_orthogonal_stress(mean_velocity,
                                            direction1,
                                            direction2)
            assert_all_moments_are_zero(model, mf)
            assert_all_moments_are_zero(model, mf * state)


@pytest.mark.parametrize("key", MODELS.keys())
def test_mf_orthogonal_heat_flow_is_orthogonal(key):
    model = MODELS[key]
    # moment functions are only orthogonal if the mean velocity is 0,
    # this is due to discretization effects!
    # in this case, the number density is usually != 0
    mean_velocity = np.zeros(model.ndim, dtype=float)
    mass_array = model.get_array(model.masses)
    for _ in range(10):
        # choose random temperature
        temperature = 100 * np.random.random() + 1
        # use parameters directly, dont initialize here (might lead to errors, definitely complicates things!)
        state = model.maxwellian(velocities=model.velocities,
                                 temperature=temperature,
                                 mass=mass_array,
                                 mean_velocity=mean_velocity)
        # choose random direction1
        for __ in range(10):
            direction1 = np.random.random(model.ndim)
            # test parallel stress mf
            mf = model.mf_orthogonal_heat_flow(state, direction1)
            assert_all_moments_are_zero(model, mf * state)


# Todo/Idea: Implement collisions only for specimen tuples
#  -> no need for get_idx  with its warnings

# Todo __get__ für model -> nach species index
#  + Grid __get__ nach index -> arbeiten mit lokalen indices möglich
#  + für (get_)idx(specimen, indices) hinzu -Y generiert globale indices

# Todo add tests for initialization, moment functions and moments
