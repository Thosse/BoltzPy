import h5py
import numpy as np

DIRECTORY = __file__[:-24] + 'test_data/'
TMP_FILE = DIRECTORY + '_tmp_.hdf5'


# Todo make this a set of strings
def get_all_keys(hdf_object):
    """:obj:`list` [ :obj:`str` ]:
    The list of all groups in the hdf5 file.
    """
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


def assert_files_are_equal(addresses):
    assert len(addresses) >= 2, (
        "At least 2 files must be given")
    n_files = len(addresses)

    for idx in range(n_files - 1):
        with h5py.File(addresses[idx], mode='r') as file_1:
            keys_1 = get_all_keys(file_1)
            with h5py.File(addresses[idx + 1], mode='r') as file_2:
                keys_2 = get_all_keys(file_2)
                assert keys_1 == keys_2, (
                    "The Files:\n\t{}\n\t{}"
                    "\ncontain different Groups or Data sets"
                    "".format(addresses[idx], addresses[idx + 1]))

            # All Datasets must have equal values
                for key in keys_1:
                    # Both must have the same hdf type (Group or Dataset)
                    assert isinstance(file_1[key], type(file_2[key])), (
                            "Differing data types for {}:\t"
                            "{} vs {}"
                            "".format(key, type(file_1[key]), type(file_2[key])))
                    # skip groups
                    if not isinstance(file_1[key], h5py.Dataset):
                        continue
                    val_1 = file_1[key][()]
                    val_2 = file_2[key][()]
                    assert isinstance(val_1, type(val_2)), (
                            "Differing data types for {}:\t"
                            "{} vs {}".format(key, type(val_1), type(val_2)))
                    if isinstance(val_1, np.ndarray):
                        assert np.array_equal(val_1, val_2), (
                            "Differing values for {}:\t"
                            "{} vs {}".format(key, val_1, val_2))
                    else:
                        assert val_1 == val_2, (
                            "Differing values for {}:\t"
                            "{} vs {}".format(key, val_1, val_2))
    return
