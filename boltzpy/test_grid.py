import numpy as np
import pytest
import os
import h5py

import boltzpy.helpers.tests as test_helper
import boltzpy as bp

###################################
#           Setup Cases           #
###################################
DIRECTORY = __file__[:-20] + 'test_data/'
FILE = DIRECTORY + 'Grids.hdf5'
GRIDS = dict()
GRIDS["1D_small"] = bp.Grid(shape=(10,),
                            delta=0.01/4,
                            spacing=4)
GRIDS["1D_large"] = bp.Grid(shape=(10000,),
                            delta=0.0001,
                            spacing=10)
GRIDS["2D_centered"] = bp.Grid(shape=(25, 25),
                               delta=0.1,
                               spacing=4,
                               is_centered=True)
GRIDS["3D_centered"] = bp.Grid(shape=(5, 5, 5),
                               delta=0.1,
                               spacing=20,
                               is_centered=True)
GRIDS["2D_rectangle"] = bp.Grid(shape=(9, 33),
                                delta=0.1,
                                spacing=4)
GRIDS["6D_centered"] = bp.Grid(shape=(2, 3, 4, 5, 6, 7),
                               delta=0.1,
                               spacing=2,
                               is_centered=True)


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the grid test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    file = h5py.File(file_address, mode="w")
    for (key, item) in GRIDS.items():
        assert isinstance(item, bp.Grid)
        file.create_group(key)
        item.save(file[key])
    return


#############################
#           Tests           #
#############################
def test_file_exists():
    assert os.path.exists(FILE)


# Todo factor this out
def test_setup_creates_same_file():
    temp_file_address = DIRECTORY + '_tmp_.hdf5'
    setup_file(temp_file_address)
    test_helper.assert_files_are_equal([FILE, temp_file_address])
    os.remove(temp_file_address)
    return


@pytest.mark.parametrize("group", GRIDS.keys())
def test_hdf5_groups_exist(group):
    file = h5py.File(FILE, mode="r")
    assert group in file.keys()


@pytest.mark.parametrize("grid_tuple", GRIDS.items())
def test_load_from_file(grid_tuple):
    file = h5py.File(FILE, mode="r")
    hdf_group = file[grid_tuple[0]]
    old = bp.Grid.load(hdf_group)
    new = grid_tuple[1]
    assert isinstance(old, bp.Grid)
    assert isinstance(new, bp.Grid)
    assert old == new


@pytest.mark.parametrize("grid", GRIDS.values())
def test_get_index_on_shuffled_grid(grid):
    # Todo allow get_idx for all dimensions
    if grid.ndim != 2:
        return
    # Todo allow get_idx for rectangles
    if len(set(grid.shape)) != 1:
        return
    rng = np.random.default_rng()
    shuffled_idx = rng.permutation(grid.size)
    shuffled_vals = grid.iG[shuffled_idx]
    assert np.all(grid.get_idx(shuffled_vals) == shuffled_idx)


@pytest.mark.parametrize("grid", GRIDS.values())
def test_get_index_on_shifted_grid(grid):
    # Todo allow get_idx for all dimensions
    if grid.ndim != 2:
        return
    # Todo allow get_idx for rectangles
    if len(set(grid.shape)) != 1:
        return
    for factor in [2, 3]:
        ext_grid = grid.extension(factor)
        for shift in range(1, grid.spacing):
            shifted_vals = ext_grid.iG + shift
            assert np.all(grid.get_idx(shifted_vals) == - 1)
