import numpy as np
import pytest

import boltzpy as bp

###################################
#           Setup Cases           #
###################################
GRIDS = dict()
GRIDS["Grid/2D_small"] = bp.Grid(shape=(50,),
                                 delta=0.002,
                                 spacing=5)
GRIDS["Grid/equalMass"] = bp.Grid(shape=(50,),
                                  delta=0.0015,
                                  spacing=7)
GRIDS["Grid/1D_small"] = bp.Grid(shape=(10,),
                                 delta=0.01/4,
                                 spacing=4)
GRIDS["Grid/1D_large"] = bp.Grid(shape=(10000,),
                                 delta=0.0001,
                                 spacing=10)
GRIDS["Grid/2D_centered"] = bp.Grid(shape=(25, 25),
                                    delta=0.1,
                                    spacing=6,
                                    is_centered=True)
GRIDS["Grid/3D_centered"] = bp.Grid(shape=(5, 5, 5),
                                    delta=0.1,
                                    spacing=2,
                                    is_centered=True)
GRIDS["Grid/2D_rectangle"] = bp.Grid(shape=(9, 33),
                                     delta=0.1,
                                     spacing=4)
GRIDS["Grid/6D_centered"] = bp.Grid(shape=(5, 3, 4, 2, 4, 3),
                                    delta=0.1,
                                    spacing=2,
                                    is_centered=True)


#############################
#           Tests           #
#############################
@pytest.mark.parametrize("key", GRIDS.keys())
def test_get_index_on_shuffled_grid(key):
    grid = GRIDS[key]
    rng = np.random.default_rng()
    shuffled_idx = rng.permutation(grid.size)
    shuffled_vals = grid.iG[shuffled_idx]
    assert np.all(grid.get_idx(shuffled_vals) == shuffled_idx)


@pytest.mark.parametrize("key", GRIDS.keys())
def test_get_index_on_shifted_grid(key):
    grid = GRIDS[key]
    for shift in range(1, grid.spacing):
        shifted_vals = grid.iG + shift
        assert np.all(grid.get_idx(shifted_vals) == - 1)


@pytest.mark.parametrize("key", GRIDS.keys())
def test_key_distance(key):
    grid = GRIDS[key]
    vals = np.random.randint(5000, size=(100, grid.ndim))
    distances = grid.key_distance(vals)
    shifted_vals = vals - distances - grid.offset
    assert np.all(shifted_vals % grid.spacing == 0)


@pytest.mark.parametrize("key", [key for key, grid in GRIDS.items()
                                 if grid.ndim in [2, 3]])
def test_recreate_distance_from_partitioned_distance(key):
    grid = GRIDS[key]
    # array of matrices (the order is important!)
    sym_matrices = bp.BaseModel(masses=[1], shapes=[grid.shape]).symmetry_matrices
    # get random values, to compute distances for
    vals = np.random.randint(5000, size=(100, grid.ndim))
    # compute distances
    distances = grid.key_distance(vals)
    # compute partitioned distance for same values
    key_partitioned_distance = grid.key_sorted_distance(vals)
    # first part [0:ndim] are sorted and absolutes of the distances
    partitioned_distances = key_partitioned_distance[..., :-1]
    # matrices that SHOULD restore distances from partitioned_distances
    restoring_matrices = sym_matrices[key_partitioned_distance[..., -1]]
    # restore distances by reversing the sorting and absolute
    restored_distances = np.einsum("ijk,ik->ij",
                                   restoring_matrices,
                                   partitioned_distances)
    assert np.all(restored_distances == distances)
    # transposed matrices implement the sorting and absolute
    sorted_distances = np.einsum("ikj,ik->ij",
                                 restoring_matrices,
                                 distances)
    assert np.all(sorted_distances == partitioned_distances)


@pytest.mark.parametrize("key", [key for key, grid in GRIDS.items()
                                 if grid.ndim not in [2, 3]])
def test_recreate_distance_fails_for_different_dimensions(key):
    # AssertionError is raised when setting up the BaseModel
    with pytest.raises(AssertionError):
        test_recreate_distance_from_partitioned_distance(key)
