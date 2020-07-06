import h5py
import numpy as np


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


def assert_files_are_equal(list_of_file_addresses):
    assert len(list_of_file_addresses) >= 2, (
        "At least 2 files must be given")
    # open the files
    files = [h5py.File(address, mode='r') for address in list_of_file_addresses]
    # get all keys of the files
    keys = [get_all_keys(file) for file in files]

    # all files must have the same groups and datasets
    for f in range(len(files) - 1):
        assert keys[f] == keys[f + 1], (
            "The Files:\n\t{}\n\t{}"
            "\ncontain different Groups or Data sets:"
            "\n{}\n{}",
            "".format(list_of_file_addresses[f],
                      list_of_file_addresses[f + 1],
                      keys[f],
                      keys[f+1]))

    # All Datasets must have equal values
    for f in range(len(files) - 1):
        file_1 = files[f]
        file_2 = files[f + 1]
        for key in keys[f]:
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
