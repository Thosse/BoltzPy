import numpy as np
import scipy.sparse
import h5py
import boltzpy as bp


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
    def subclasses(subclass=None):
        """Returns the Class object of the given subclass.
        If no subclass is given, then returns a dictionary of all subclasses.

        Parameters
        ----------
        subclass : :obj:`str`, optional

        Returns
        -------
        self : :obj:`type`
        """
        # this distionary contains all possible subclasses
        subclasses = {
            'Grid': bp.Grid,
            'CollisionModel': bp.CollisionModel,
            'Geometry': bp.Geometry,
            'InnerPointRule': bp.InnerPointRule,
            'ConstantPointRule': bp.ConstantPointRule,
            'BoundaryPointRule': bp.BoundaryPointRule,
            'HomogeneousRule': bp.HomogeneousRule,
            "Simulation": bp.Simulation}
        # this is mainly used for asserts
        if subclass is None:
            return subclasses
        else:
            return subclasses[subclass]

    @staticmethod
    def parameters():
        """The set of initialization parameters, including optionals."""
        raise NotImplementedError

    @staticmethod
    def attributes():
        """The set of all class attributes and propertes."""
        raise NotImplementedError

    @staticmethod
    def load_attributes(hdf5_group, attributes):
        """Read a set of parameters from a given HDF5 group.
        Returns a dictionary of the read values
        or the value, if only a single attribute was given as a string.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        attributes : :obj:`str`, :obj:`list`, or  :obj:`set`
        """
        # return values directly, if a single attribute is given as a string
        if type(attributes) is str:
            result = BaseClass.load_attributes(hdf5_group, [attributes])
            return result[attributes]
        # otherwise return a dictionary
        assert isinstance(hdf5_group, h5py.Group)
        assert type(attributes) in [list, set]
        # read parameters and store in dict
        result = dict()
        for p in attributes:
            assert p in hdf5_group.keys()
            # Datasets are read directly
            if isinstance(hdf5_group[p], h5py.Dataset):
                val = hdf5_group[p][()]
                # strings must be decoded into urf-8
                # simply reading would return a byte string
                if type(val) == bytes:
                    result[p] = hdf5_group[p].asstr()[()]
                else:
                    result[p] = val
            # Groups contain either a Baseclass, or an object-array
            elif isinstance(hdf5_group[p], h5py.Group):
                cls = hdf5_group[p].attrs["class"]
                # group contains a Baseclass
                if cls in BaseClass.subclasses().keys():
                    subclass = BaseClass.subclasses(cls)
                    result[p] = subclass.load(hdf5_group[p])
                # group is an array of BaseClasses
                elif cls == "Array":
                    size = hdf5_group[p].attrs["size"]
                    result[p] = np.empty(size, dtype=object)
                    for idx in range(size):
                        grp = hdf5_group[p][str(idx)]
                        subclass = BaseClass.subclasses(grp.attrs["class"])
                        result[p][idx] = subclass.load(grp)
                # group must contain one of those two options
                else:
                    raise ValueError
            else:
                raise ValueError
        return result

    @staticmethod
    def load(hdf5_group):
        """Read parameters from the given HDF5 group,
        initialize and return an instance based on these parameters.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`BaseClass`
        """
        assert hdf5_group.attrs["class"] in BaseClass.subclasses().keys()
        subclass = BaseClass.subclasses(hdf5_group.attrs["class"])
        # read parameters from file
        parameters = BaseClass.load_attributes(hdf5_group,
                                               subclass.parameters())
        return subclass(**parameters)

    @staticmethod
    def save_attribute(hdf5_group, key, value):
        # save Base class attributes in subgroup
        if isinstance(value, BaseClass):
            hdf5_group.create_group(key)
            value.save(hdf5_group[key])
        # save arrays of objects in sub-subgroups
        elif isinstance(value, np.ndarray) and value.dtype == 'object':
            # create subgroup
            hdf5_group.create_group(key)
            hdf5_group[key].attrs["class"] = "Array"
            hdf5_group[key].attrs["size"] = value.size
            # save elements iteratively
            for (idx, element) in enumerate(value):
                if isinstance(element, BaseClass):
                    idx = str(idx)
                    hdf5_group[key].create_group(idx)
                    element.save(hdf5_group[key][idx])
                else:
                    raise NotImplementedError
        # transform sparse matrices to normal np.arrays
        elif isinstance(value, scipy.sparse.csr_matrix):
            value = value.toarray()
            hdf5_group[key] = value
        else:
            hdf5_group[key] = value

    def save(self, hdf5_group, write_all=False):
        """Write the parameters of the Class into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        write_all : :obj:`bool`
            If True, write all attributes and properties to the file,
            even the unnecessary ones. Useful for testing,
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity()
        # Remove all keys and attributes in  current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        # save class name for automatic loading/initialization
        hdf5_group.attrs["class"] = self.__class__.__name__
        # choose attributes to save
        if write_all:
            attributes = self.attributes()
        else:
            attributes = self.parameters()
        # save attributes to file
        for attr in attributes:
            self.save_attribute(hdf5_group, attr, self.__getattribute__(attr))

        # Todo move into testcase
        # check that the class can be reconstructed from the save
        other = self.load(hdf5_group)
        assert self == other
        return

    def check_integrity(self):
        """Sanity Check."""
        assert isinstance(self.parameters(), set)
        assert isinstance(self.attributes(), set)
        assert self.parameters().issubset(self.attributes())
