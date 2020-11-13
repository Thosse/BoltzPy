import numpy as np
import scipy.sparse
import h5py


class BaseClass:
    def __eq__(self, other, ignore=None, print_message=True):
        if ignore is None:
            ignore = []
        # This explicitly allows other to be an child class of self
        if not isinstance(other, type(self)):
            if print_message:
                print("Objects are of different type:",
                      "\n\ttype(self) = ", type(self),
                      "\n\ttype(other) = ", type(other))
            return False
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            if print_message:
                print("Objects have different attributes:",
                      "\n\tself.keys = ", set(self.__dict__.keys()),
                      "\n\tother.keys = ", set(other.__dict__.keys()))
            return False
        for (key, value) in self.__dict__.items():
            if key in ignore:
                continue
            other_value = other.__dict__[key]
            if type(value) != type(other_value):
                if print_message:
                    print("An attribute is of different type:",
                          "\n\tAttribute = ", key,
                          "\n\ttype(self) = ", type(value),
                          "\n\ttype(other) = ", type(other_value))
                return False
            if isinstance(value, scipy.sparse.csr_matrix):
                value = value.toarray()
                other_value = other_value.toarray()
            if isinstance(value, np.ndarray):
                if value.shape != other_value.shape:
                    if print_message:
                        print("An attribute has differing shapes:",
                              "\n\tAttribute = ", key,
                              "\n\tself.attr.shape = ", value.shape,
                              "\n\tother.attr.shape) = ", other_value.shape)
                    return False
                if value.dtype == float:
                    if not np.allclose(value, other_value):
                        if print_message:
                            print("An attribute has differing values:",
                                  "\n\tAttribute = ", key)
                        return False
                else:
                    if not np.array_equal(value, other_value):
                        if print_message:
                            print("An attribute has differing values:",
                                  "\n\tAttribute = ", key)
                        return False
            else:
                if value != other_value:
                    if print_message:
                        print("An attribute has differing values:",
                              "\n\tAttribute = ", key)
                    return False
        return True

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def parameters():
        raise NotImplementedError

    @staticmethod
    def attributes():
        raise NotImplementedError

    @staticmethod
    def read_parameters_from_hdf_file(hdf5_group, parameters):
        """Read a set of parameters from a given HDF5 group.
        Returns a dictionary of the read values.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        parameters : :obj:`str`, :obj:`list`, or  :obj:`set`

        Returns
        -------
        self : :obj:`dict`
        """
        assert isinstance(hdf5_group, h5py.Group)
        if type(parameters) is str:
            parameters = [parameters]
        else:
            assert type(parameters) in [list, set]
            assert all(type(p) is str for p in parameters)
        result = dict()
        for p in parameters:
            val = hdf5_group[p][()]
            if type(val) == bytes:
                result[p] = hdf5_group[p].asstr()[()]
            else:
                result[p] = val
        return result

    @staticmethod
    def load(hdf5_group):
        """Read parameters from the given HDF5 group,
        initialize and return an instance based on these parameters.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        raise NotImplementedError

    def save(self, hdf5_group, write_all=False):
        """Write the main parameters of the :class:`Model` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        write_all : :obj:`bool`
            If True, write all attributes and properties to the file,
            even the unnecessary ones. Useful for testing,
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity()

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = self.__class__.__name__
        # choose attributes to save
        if write_all:
            attributes = self.attributes()
        else:
            attributes = self.parameters()
        # save attributes to file
        for attr in attributes:
            value = self.__getattribute__(attr)
            # transform sparse matrices to normal np.arrays
            if isinstance(value, scipy.sparse.csr_matrix):
                value = value.toarray()
            # save Base class attributes in subgroup
            if isinstance(value, BaseClass):
                value.save(hdf5_group[attr])
            # save arrays of objects in sub-subgroups
            is_array_of_objects = isinstance(value, np.ndarray) and value.dtype == 'object'
            if is_array_of_objects:
                # create subgroup
                hdf5_group.create_group(attr)
                hdf5_group[attr].attrs["class"] = "Array"
                hdf5_group[attr].attrs["size"] = value.size
                # save elements iteratively
                for (idx, element) in enumerate(value):
                    assert isinstance(element, BaseClass)
                    idx = str(idx)
                    hdf5_group[attr].create_group(idx)
                    element.save(hdf5_group[attr][idx])
            else:
                hdf5_group[attr] = value

        # check that the class can be reconstructed from the save
        other = self.load(hdf5_group)
        assert self == other
        return

    def check_integrity(self):
        """Sanity Check."""
        raise NotImplementedError

