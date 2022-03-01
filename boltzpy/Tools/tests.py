import h5py
import numpy as np


def assert_hdf_groups_are_equal(*hdf5_groups):
    assert len(hdf5_groups) >= 2, (
        "At least 2 files must be given")
    n_groups = len(hdf5_groups)
    for idx in range(n_groups - 1):
        group_1 = hdf5_groups[idx]
        keys_1 = group_1.keys()
        group_2 = hdf5_groups[idx + 1]
        keys_2 = group_2.keys()
        # Keys must be equal!
        assert keys_1 == keys_2, (keys_1, keys_2)
        # Todo compare attributes as well?

        for key in keys_1:
            value_1 = group_1[key]
            value_2 = group_1[key]
            # Both must have the same hdf type (Group or Dataset)
            assert isinstance(value_1, type(value_2))
            # test groups recursively
            if isinstance(value_1, h5py.Group):
                assert_hdf_groups_are_equal(value_1, value_2)
            # Datasets must be equal
            elif isinstance(value_1, h5py.Dataset):
                value_1 = value_1[()]
                value_2 = value_2[()]
                assert isinstance(value_1, type(value_2))
                if isinstance(value_1, np.ndarray):
                    assert value_1.shape == value_2.shape
                    assert value_1.dtype == value_2.dtype
                    if value_1.dtype == float:
                        assert np.allclose(value_1, value_2)
                    else:
                        assert np.array_equal(value_1, value_2)
                else:
                    assert value_1 == value_2
    return
