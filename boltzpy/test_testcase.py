
import boltzpy as bp
import boltzpy.testcase as bp_t
import pytest
import numpy as np
import os
import h5py


@pytest.mark.parametrize("tc", bp_t.CASES)
def test_file_exists(tc):
    assert isinstance(tc, bp_t.TestCase)
    assert os.path.exists(tc.file_address)


@pytest.mark.parametrize("tc", bp_t.CASES)
def test_file_initializes_the_correct_simulation(tc):
    sim = bp.Simulation.load(tc.file_address)
    assert sim.__eq__(tc, True)
    assert not tc.__eq__(sim, False)
    new_tc = bp_t.TestCase.load(tc.file_address)
    assert new_tc == tc


@pytest.mark.parametrize("tc", bp_t.CASES)
def test_new_file_is_equal_to_current_one(tc):
    assert isinstance(tc, bp_t.TestCase)
    # TODO move this into prerun of pytest
    # temporary file, if it exists already
    if os.path.exists(tc.temporary_file):
        os.remove(tc.temporary_file)

    # create new file with results
    tc.compute(tc.temporary_file)
    # load old and new data
    new_file = h5py.File(tc.temporary_file, mode='r')
    old_file = h5py.File(tc.file_address, mode='r')
    # get all keys of the files
    new_keys = get_all_keys(new_file)
    old_keys = get_all_keys(old_file)
    # keys must be equal
    assert set(new_keys) == set(old_keys), (
        "The Files contain different Groups or Data sets!"
        "\nnew_keys: ", new_keys,
        "\nold_keys: ", old_keys
    )
    for key in new_keys:
        # Both must have the same hdf type (Group or Dataset)
        assert isinstance(new_file[key], type(old_file[key]))
        if isinstance(new_file[key], h5py.Dataset):
            # load values
            new_value = new_file[key][()]
            old_value = old_file[key][()]
            assert isinstance(new_file[key], type(old_file[key]))
            if isinstance(new_value, np.ndarray):
                assert np.array_equal(new_value, old_value)
            else:
                assert new_value == old_value
    os.remove(tc.temporary_file)


def get_all_keys(hdf_object):
    keys = []
    if isinstance(hdf_object, h5py.Dataset):
        return keys
    if isinstance(hdf_object, h5py.Group):
        for (key, subgroup) in hdf_object.items():
            subkeys = get_all_keys(subgroup)
            keys.append(key)
            keys = keys + ["{}/{}".format(key, subkey)
                           for subkey in subkeys]
    else:
        raise Exception
    return keys
