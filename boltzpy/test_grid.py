import numpy as np
import pytest
import os
import h5py

import boltzpy.helpers.tests as test_helper
import boltzpy as bp

###################################
#           Setup Cases           #
###################################
FILE = test_helper.DIRECTORY + 'Grids.hdf5'
GRIDS = dict()
GRIDS["2D_small/timing"] = bp.Grid(shape=(50,),
                                   delta=0.002,
                                   spacing=5)
GRIDS["equalMass/timing"] = bp.Grid(shape=(50,),
                                    delta=0.0015,
                                    spacing=7)
GRIDS["1D_small"] = bp.Grid(shape=(10,),
                            delta=0.01/4,
                            spacing=4)
GRIDS["1D_large"] = bp.Grid(shape=(10000,),
                            delta=0.0001,
                            spacing=10)
GRIDS["2D_centered"] = bp.Grid(shape=(25, 25),
                               delta=0.1,
                               spacing=6,
                               is_centered=True)
GRIDS["3D_centered"] = bp.Grid(shape=(5, 5, 5),
                               delta=0.1,
                               spacing=2,
                               is_centered=True)
GRIDS["2D_rectangle"] = bp.Grid(shape=(9, 33),
                                delta=0.1,
                                spacing=4)
GRIDS["6D_centered"] = bp.Grid(shape=(5, 3, 4, 2, 4, 3),
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
    assert os.path.exists(FILE), (
        "The test file {} is missing.".format(FILE))


def test_setup_creates_same_file():
    setup_file(test_helper.TMP_FILE)
    test_helper.assert_files_are_equal([FILE, test_helper.TMP_FILE])
    os.remove(test_helper.TMP_FILE)
    return


@pytest.mark.parametrize("key", GRIDS.keys())
def test_hdf5_groups_exist(key):
    file = h5py.File(FILE, mode="r")
    assert key in file.keys(), (
        "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", GRIDS.keys())
def test_load_from_file(key):
    file = h5py.File(FILE, mode="r")
    old = bp.Grid.load(file[key])
    new = GRIDS[key]
    assert isinstance(old, bp.Grid)
    assert isinstance(new, bp.Grid)
    assert old == new, (
        "\n{}\nis not equal to\n\n{}".format(old, new)
    )


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
    for factor in [2, 3]:
        ext_grid = grid.extension(factor)
        for shift in range(1, grid.spacing):
            shifted_vals = ext_grid.iG + shift
            assert np.all(grid.get_idx(shifted_vals) == - 1)
