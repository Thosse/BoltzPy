
import h5py

import boltzpy.constants as bp_c


# Todo Scheme might needs a data (cpu/GPU) parameter
class Scheme:
    """Encapsulates all parameters related to the computation schemes.

    Essentially behaves like a :obj:`dict`.

    Attributes
    ----------
    dictionary : :obj:`dict` [:obj:`str`, :obj:`str`]

    """
    def __init__(self, **kwargs):
        self.dictionary = dict()
        self.dictionary["Approach"] = None
        for (key, val) in kwargs.items():
            self.dictionary[key] = val
        self.check_integrity(False)
        return

    def __getitem__(self, item):
        return self.dictionary[item]

    def __setitem__(self, key, value):
        if key not in self.dictionary.keys():
            msg = "Invalid Key!"
            raise KeyError(msg)
        # Add new keys required by new value, if any
        if value in bp_c.REQ_SCHEME_PARAMETERS.keys():
            for new_key in bp_c.REQ_SCHEME_PARAMETERS[value]:
                self.dictionary[new_key] = None
        # Set new value
        assert value in bp_c.SUPP_SCHEME_VALUES[key]
        self.dictionary[key] = value
        # delete old entries, that are not required anymore
        current_keys = set(self.dictionary.keys())
        useless_keys = current_keys.difference(self._required_keys())
        for _key in useless_keys:
            del self.dictionary[_key]
        return

    @property
    def is_configured(self):
        """Check if all necessary attributes of the instance are set.

        Returns
        -------
        :obj:`bool`
        """
        return all(value is not None for(key, value) in self.items())

    def keys(self):
        """Returns a list of all scheme parameters.

        Returns
        -------
        :obj:`list` [:obj:`str`]
        """
        return list(self.dictionary.keys())

    def items(self):
        """Returns a list of all scheme parameters and their value.

        Returns
        -------
        :obj:`list` [:obj:`tuple` [:obj:`str`, :obj:`str`]]
        """
        return list(self.dictionary.items())

    def _required_keys(self):
        required_keys = ["Approach"]
        unchecked_keys = required_keys.copy()
        while unchecked_keys:
            key = unchecked_keys.pop()
            value = self[key]
            # add required parameters, if any
            if value in bp_c.REQ_SCHEME_PARAMETERS.keys():
                new_keys = bp_c.REQ_SCHEME_PARAMETERS[value]
                additional_keys = [key for key in new_keys
                                   if key not in required_keys]
                unchecked_keys.extend(additional_keys)
                required_keys.extend(additional_keys)

        # Type casting into a set removes duplicates
        return set(required_keys)

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`Scheme` instance
        based on the attributes in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group`

        Returns
        -------
        :class:`Scheme`
        """
        assert isinstance(hdf5_group, h5py.Group)

        self = Scheme()
        for key in hdf5_group.attrs:
            self.dictionary[key] = hdf5_group.attrs[key]
        for key in self._required_keys():
            if key not in self.keys():
                self.dictionary[key] = None
        self.check_integrity()
        return self

    def save(self, hdf5_group):
        """Write all parameters of the :obj:`Scheme` instance
        to attributes of the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity(False)

        # Clean State of Current group
        for key in hdf5_group.attrs:
            del hdf5_group[key]

        # write all set attributes to file
        for (key, value) in self.items():
            if value is not None:
                hdf5_group.attrs[key] = value
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self, complete_check=True):
        """Sanity Check.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be assigned (not None).
        """
        assert isinstance(self.dictionary, dict)
        assert set(self.keys()) == self._required_keys()
        for (key, value) in self.items():
            assert isinstance(key, str)
            assert value is None or value in bp_c.SUPP_SCHEME_VALUES[key]
        if complete_check:
            for (key, value) in self.items():
                assert value is not None

    def __str__(self):
        ret = ""
        for key in sorted(self.keys()):
            ret += "{key} = {val}\n".format(key=key,
                                            val=self[key])
        return ret

    def __repr__(self, namespace=None):
        rep = "Scheme("
        # add custom namespace
        if namespace is not None:
            assert isinstance(namespace, str)
            rep = namespace + "." + rep
        for (key, val) in self.items():
            if isinstance(val, str):
                rep += "{key}='{val}', ".format(key=key, val=val)
            elif isinstance(val, int):
                rep += "{key}={val}, ".format(key=key, val=val)
            else:
                raise NotImplementedError
        rep = rep[:-2]
        rep += ')'
        return rep

    def __eq__(self, other):
        if not isinstance(other, Scheme):
            return False
        if self.items() != other.items():
            return False
        return True
