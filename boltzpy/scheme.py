import h5py
import numpy as np
import boltzpy as bp


# Todo Scheme might needs a data (cpu/GPU) parameter
# noinspection PyPep8Naming
class Scheme:
    """Encapsulates all parameters related to the
    choice or adjustment of computation algorithms.

    Parameters
    ----------
    OperatorSplitting : :obj:`str`, optional
    Transport : :obj:`str`, optional
    Transport_VelocityOffset : :obj:`~numpy.array` [:obj:`float`], optional
        If this is left empty (None),
        then the Velocity Offset is set to zero.
    Collisions_Generation : :obj:`str`, optional
    Collisions_Computation : :obj:`str`, optional
    """
    def __init__(self,
                 OperatorSplitting=None,
                 Transport=None,
                 Transport_VelocityOffset=None,
                 Collisions_Generation=None,
                 Collisions_Computation=None):
        self.OperatorSplitting = OperatorSplitting
        self.Transport = Transport
        if type(Transport_VelocityOffset) in [list, tuple]:
            Transport_VelocityOffset = np.array(Transport_VelocityOffset,
                                                dtype=float)
        self.Transport_VelocityOffset = Transport_VelocityOffset
        self.Collisions_Generation = Collisions_Generation
        self.Collisions_Computation = Collisions_Computation
        self.check_integrity(complete_check=False)
        return

    #: :obj:`dict` [:obj:`str`, :obj:`list`]
    SUPP_VALUES = {
        "OperatorSplitting": ["FirstOrder",
                              # NoTransport
                              ],
        "Transport": ["FiniteDifferences_FirstOrder"],
        "Collisions_Generation": ["UniformComplete",
                                  # "NoCollisions",
                                  ],
        "Collisions_Computation": ["EulerScheme",
                                   # NoCollisions,
                                   ]
    }

    #: :obj:`dict` [:obj:`str`, :obj:`type`]
    SUPP_VALUE_TYPE = {
        "OperatorSplitting": str,
        "Transport": str,
        "Transport_VelocityOffset": np.ndarray,
        "Collisions_Generation": str,
        "Collisions_Computation": str
    }

    @property
    def necessary_attributes(self):
        """ :obj:`set` [:obj:`str`]
        Returns the attributes of the instance that must be set
        for the instance to be fully configured.
        """
        required_attributes = {"OperatorSplitting",
                               "Transport",
                               "Collisions_Generation",
                               "Collisions_Computation"}
        return required_attributes

    # Todo add test
    @property
    def is_configured(self):
        """:obj:`bool` :
        True, if all :meth:`necessary_attributes` of the instance are set.
        False Otherwise.
        """
        return all(self.__dict__[attribute] is not None
                   for attribute in self.necessary_attributes)

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`Scheme` instance
        based on the attributes in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Scheme`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "Scheme"
        self = Scheme()

        for (key, value) in hdf5_group.items():
            assert key in self.__dict__.keys()
            self.__dict__[key] = value[()]
        self.check_integrity(False)
        return self

    def save(self, hdf5_group):
        """Write all set attributes of the :obj:`Scheme` instance
        to attributes of the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity(False)

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = "Scheme"

        # write all set attributes to file
        for (key, value) in self.__dict__.items():
            if value is not None:
                hdf5_group[key] = value
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self, complete_check=True, context=None):
        """Sanity Check.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be assigned (not None).
        context : :class:`Simulation`, optional
            The Simulation, which this instance belongs to.
            Allows additional checks.
        """
        for (key, value) in self.__dict__.items():
            if value is not None:
                assert type(value) is self.SUPP_VALUE_TYPE[key]
                if key in self.SUPP_VALUES.keys():
                    assert value in self.SUPP_VALUES[key]
        if self.Transport_VelocityOffset is not None:
            assert self.Transport_VelocityOffset.dtype == float
        if context is not None:
            assert isinstance(context, bp.Simulation)
            if self.Transport_VelocityOffset is not None:
                assert len(self.Transport_VelocityOffset) == context.sv.dim
        if complete_check:
            for (key, value) in self.__dict__.items():
                assert value is not None
        return

    def __str__(self):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance.
        """
        ret = ""
        for key in sorted(self.__dict__.keys()):
            ret += "{key} = {value}\n".format(key=key,
                                              value=self.__dict__[key])
        return ret

    def __repr__(self, namespace=None):
        rep = "Scheme("
        # add custom namespace
        if namespace is not None:
            assert isinstance(namespace, str)
            rep = namespace + "." + rep
        for (key, value) in self.__dict__.items():
            if isinstance(value, str):
                rep += "{key}='{value}', ".format(key=key, value=value)
            elif isinstance(value, np.ndarray):
                rep += "{key}={value}, ".format(key=key, value=list(value))
            else:
                raise NotImplementedError
        rep = rep[:-2]
        rep += ')'
        return rep

    def __eq__(self, other):
        if not isinstance(other, Scheme):
            return False
        # can't compare __dict__ because comparing the offset arrays
        # returns list of bools. This leads to errors.
        if self.__dict__.keys() != other.__dict__.keys():
            return False
        if any(self.__dict__[key] != other.__dict__[key]
               for key in self.__dict__.keys()
               if key != "Transport_VelocityOffset"):
            return False
        if not np.allclose(self.Transport_VelocityOffset,
                           other.Transport_VelocityOffset,
                           atol=1e-14):
            return False
        return True
