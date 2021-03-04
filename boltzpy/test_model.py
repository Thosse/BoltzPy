import pytest
import numpy as np

import boltzpy as bp

###################################
#           Setup Cases           #
###################################
MODELS = dict()
MODELS["Model/2D_small"] = bp.CollisionModel(
    masses=[2, 3],
    shapes=[[5, 5], [7, 7]],
    base_delta=1/8,
    spacings=[6, 4],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")
MODELS["Model/2D_small_convergent"] = bp.CollisionModel(
    masses=[2, 3],
    shapes=[[5, 5], [7, 7]],
    base_delta=1/8,
    spacings=[6, 4],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="convergent",
    algorithm_weights="uniform")
# Todo This might lead to weird results! Check this!
MODELS["Model/equalSpacing"] = bp.CollisionModel(
    masses=[1, 2],
    shapes=[[7, 7], [7, 7]],
    base_delta=1/7,
    spacings=[2, 2],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")
MODELS["Model/equalMass"] = bp.CollisionModel(
    masses=[3, 3],
    shapes=[[5, 5], [5, 5]],
    base_delta=3/8,
    spacings=[2, 2],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")
# Todo Allow equal spacing
# MODELS["evenShapes/Model"] = bp.Model(
#     masses=[2, 3],
#     shapes=[[4, 4], [6, 6]],
#     base_delta=1/12,
#     spacings=[2, 2],
#     collision_factors=[[50, 50], [50, 50]],
#     algorithm_relations="all",
#     algorithm_weights="uniform")
MODELS["Model/mixed_spacing"] = bp.CollisionModel(
    masses=[3, 4],
    shapes=[[6, 6], [7, 7]],
    base_delta=1/14,
    spacings=[8, 6],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")
MODELS["Model/3D_small"] = bp.CollisionModel(
    masses=[2, 3],
    shapes=[[2, 2, 2], [2, 2, 2]],
    base_delta=1/8,
    spacings=[6, 4],
    collision_factors=[[50, 50], [50, 50]],
    algorithm_relations="all",
    algorithm_weights="uniform")


#############################
#           Tests           #
#############################
@pytest.mark.parametrize("key", MODELS.keys())
def test_get_spc_on_shuffled_grid(key):
    model = MODELS[key]
    # determines the species for each velocity
    species = model.get_array(model.species)
    # generate shuffled velocity indices
    rng = np.random.default_rng()
    shuffled_vel_idx = rng.permutation(model.nvels)
    # test for 0d arrays/elements
    for idx in shuffled_vel_idx:
        assert model.get_spc(idx).ndim == 0
        assert model.get_spc(idx) == species[idx]
    # test_get_spc for higher dimensions
    for ndmin in range(1, 6):
        # prepend ndmin 1s to the shape
        shuffled_idx = np.array(shuffled_vel_idx, ndmin=ndmin)
        shuffles_spc = species[shuffled_idx]
        assert np.all(model.get_spc(shuffled_idx) == shuffles_spc)


@pytest.mark.parametrize("key", MODELS.keys())
def test_get_idx_on_shuffled_grid(key):
    model = MODELS[key]
    # generate shuffled velocities and (matching) species indices
    rng = np.random.default_rng()
    shuffled_idx = rng.permutation(model.nvels)
    spc_idx = model.get_array(model.species)
    shuffled_spc_idx = spc_idx[shuffled_idx]
    shuffled_vels = model.i_vels[shuffled_idx]
    # Test vectorized
    result_idx = model.get_idx(shuffled_spc_idx, shuffled_vels)
    assert np.all(result_idx == shuffled_idx)


@pytest.mark.parametrize("key", MODELS.keys())
def test_get_idx_on_random_integers(key):
    model = MODELS[key]
    # number of random values
    nvals = 1000
    # generate random species
    spc_idx = np.random.randint(model.nspc, size=(nvals,))
    # generate random integer velocities
    max_val = np.max(model.spacings[:, np.newaxis] * model.shapes)
    i_vels = np.random.randint(max_val, size=(nvals, model.ndim))
    # Test vectorized
    result_idx = model.get_idx(spc_idx, i_vels)

    # check accepted values, returned index must point to the original velocity
    pos_match = np.flatnonzero(result_idx >= 0)
    idx_match = result_idx[pos_match]
    vels_match = i_vels[pos_match]
    assert np.array_equal(model.i_vels[idx_match], vels_match)

    # rejected values (result_idx == -1), must not be in the grid
    pos_miss = np.flatnonzero(result_idx < 0)
    subgrids = model.subgrids()
    for i in pos_miss:
        s = spc_idx[i]
        assert i_vels[i] not in subgrids[s]


def assert_all_moments_are_zero(model, state):
    # number density of each species stays the same
    # Todo replace using "seperately"
    for s in model.species:
        number_density = model.cmp_number_density(state, s)
        assert np.isclose(number_density, 0)
    # assert total number/mass stays the same (should be unnecessary)
    number_density = model.cmp_number_density(state)
    assert np.isclose(number_density, 0)
    mass_density = model.cmp_mass_density(state)
    assert np.isclose(mass_density, 0)
    momentum = model.cmp_momentum(state)
    assert np.allclose(momentum, 0)
    energy = model.cmp_energy_density(state)
    assert np.isclose(energy, 0)


def get_random_orthogonal_vectors(n_vectors, dimension):
    assert n_vectors <= dimension
    random_matrix = np.random.rand(n_vectors, dimension)
    # singular value decomposition
    # random_matrix = u @ diag @ vh
    # with u and vh orthogonal
    u, diag, vh = np.linalg.svd(random_matrix, full_matrices=False)
    return vh


@pytest.mark.parametrize("key", MODELS.keys())
def test_invariance_of_moments_under_collision_operator(key):
    model = MODELS[key]
    for _ in range(100):
        state = np.random.random(model.nvels)
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
        max_v = model.max_vel
        mean_velocity = 2 * max_v * np.random.random(model.ndim) - max_v
        temperature = 100 * np.random.random() + 1
        # use parameters directly, dont initialize here (might lead to errors, definitely complicates things!)
        state = model.maxwellian(velocities=model.vels,
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
            result = bp.CollisionModel.p_vels(vectors, direction)
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
            res = bp.CollisionModel.p_vels(vectors, direction[i])
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
        state = model.maxwellian(velocities=model.vels,
                                 temperature=temperature,
                                 mass=mass_array,
                                 mean_velocity=mean_velocity)
        # choose random directions
        for __ in range(10):
            directions = np.zeros((2, model.ndim))
            # test parallel stress mf
            directions[:] = np.random.random(model.ndim)
            mf = model.mf_stress(mean_velocity,
                                 directions,
                                 orthogonalize=True)
            assert_all_moments_are_zero(model, mf)
            assert_all_moments_are_zero(model, mf * state)
            # test orthogonal stress, choose simple orthogonal direction
            directions = get_random_orthogonal_vectors(2, model.ndim)
            assert model.is_orthogonal(directions[0], directions[1])
            mf = model.mf_stress(mean_velocity,
                                 directions,
                                 orthogonalize=True)
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
        state = model.maxwellian(velocities=model.vels,
                                 temperature=temperature,
                                 mass=mass_array,
                                 mean_velocity=mean_velocity)
        # choose random direction1
        for __ in range(10):
            direction1 = np.random.random(model.ndim)
            # test parallel stress mf
            mf = model.mf_heat_flow(mean_velocity, direction1, orthogonalize_state=state)
            assert_all_moments_are_zero(model, mf * state)


# Todo/Idea: Implement collisions only for specimen tuples
#  -> no need for get_idx  with its warnings

# Todo __get__ für model -> nach species index
#  + Grid __get__ nach index -> arbeiten mit lokalen indices möglich
#  + für (get_)idx(specimen, indices) hinzu -Y generiert globale indices

# Todo add tests for initialization
