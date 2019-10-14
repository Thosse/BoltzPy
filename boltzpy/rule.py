
import numpy as np
import h5py

import boltzpy.constants as bp_c


class Rule:
    """Encapsulates all data to initialize a :class:`~boltzpy.Grid` point.

    A rule must be applied to every point of the
    :attr:`simulation.p <boltzpy.Simulation>`
    :class:`boltzpy.Grid`.
    It determines the points:

        * initial distribution in the velocity space
          based on :attr:`initial_rho`, :attr:`initial_drift`, and :attr:`initial_temp`.
        * behaviour (see :const:`~boltzpy.constants.SUPP_BEHAVIOUR_TYPES`)
          during the :mod:`computation`

    Parameters
    ----------
    behaviour_type : :obj:`str`
        Determines the behaviour during the simulation.
        Must be in :const:`~boltzpy.constants.SUPP_BEHAVIOUR_TYPES`.
    initial_rho : :obj:`~numpy.array` [:obj:`float`]
    initial_drift : :obj:`~numpy.array` [:obj:`float`]
    initial_temp : :obj:`~numpy.array` [:obj:`float`]
    affected_points : :obj:`~numpy.array`[:obj:`int`], optional
    name : :obj:`str`, optional
    color : :obj:`str`, optional

    Attributes
    ----------
    behaviour_type : :obj:`str`
        Determines the behaviour during the simulation.
        Must be in :const:`~boltzpy.constants.SUPP_BEHAVIOUR_TYPES`.
    initial_rho : :obj:`~numpy.array` [:obj:`float`]
        Correlates to the initial amount of particles in
        :class:`P-Grid <boltzpy.Grid>` point.
    initial_drift : :obj:`~numpy.array` [:obj:`float`]
        Describes the mean velocity,
        i.e. the first moment (expectancy value) of the
        velocity distribution.
    initial_temp : :obj:`~numpy.array` [:obj:`float`]
        Correlates to the Energy,
        i.e. the second moment (variance) of the
        velocity distribution.
    affected_points : :obj:`~numpy.array`[:obj:`int`]
        Contains all indices of the space points, where this rule applies
    name : :obj:`str`
        Is displayed in the GUI to visualize the initialization.
    color : :obj:`str`
        Is Displayed in the GUI to visualize the initialization.
    """
    def __init__(self,
                 behaviour_type=None,
                 initial_rho=None,
                 initial_drift=None,
                 initial_temp=None,
                 affected_points=None,
                 name=None,
                 color=None):
        self.check_parameters(behaviour_type=behaviour_type,
                              initial_rho=initial_rho,
                              initial_drift=initial_drift,
                              initial_temp=initial_temp,
                              name=name,
                              color=color)
        self.behaviour_type = behaviour_type
        self.initial_rho = initial_rho
        self.initial_drift = initial_drift
        self.initial_temp = initial_temp

        if name is not None:
            self.name = name
        else:
            self.name = ''

        if color is not None:
            self.color = color
        else:
            self.color = 'black'

        self.check_integrity(False)
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`Rule` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Rule`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "Rule"
        self = Rule()

        # read attributes from file
        try:
            self.behaviour_type = hdf5_group["Category"][()]
        except KeyError:
            self.behaviour_type = None
        try:
            self.initial_rho = hdf5_group["Mass"][()]
        except KeyError:
            self.initial_rho = None
        try:
            self.initial_drift = hdf5_group["Mean Velocity"][()]
        except KeyError:
            self.initial_drift = None
        try:
            self.initial_temp = hdf5_group["Temperature"][()]
        except KeyError:
            self.initial_temp = None
        try:
            self.name = hdf5_group["Name"][()]
        except KeyError:
            self.name = ''
        try:
            self.color = hdf5_group["Color"][()]
        except KeyError:
            self.color = 'black'
        self.check_integrity(False)
        return self

    def save(self, hdf5_group):
        """Write the main parameters of the :class:`Rule` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity(False)

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = "Rule"

        # write all set attributes to file
        if self.behaviour_type is not None:
            hdf5_group["Category"] = self.behaviour_type
        if self.initial_rho is not None:
            hdf5_group["Mass"] = self.initial_rho
        if self.initial_drift is not None:
            hdf5_group["Mean Velocity"] = self.initial_drift
        if self.initial_temp is not None:
            hdf5_group["Temperature"] = self.initial_temp
        if self.name != '':
            hdf5_group["Name"] = self.name
        if self.color != 'black':
            hdf5_group["Color"] = self.color
        return

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self, complete_check=True):
        """Sanity Check.
        Assert all conditions in :meth:`check_parameters`
        and the correct type of all attributes of the instance.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all attributes must be assigned (not None).
            If False, then unassigned attributes are ignored.
        """
        self.check_parameters(behaviour_type=self.behaviour_type,
                              initial_rho=self.initial_rho,
                              initial_drift=self.initial_drift,
                              initial_temp=self.initial_temp,
                              name=self.name,
                              color=self.color,
                              complete_check=complete_check)
        return

    @staticmethod
    def check_parameters(behaviour_type=None,
                         initial_rho=None,
                         initial_drift=None,
                         initial_temp=None,
                         name=None,
                         color=None,
                         complete_check=False):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        behaviour_type : :obj:`str`, optional
        initial_rho : :obj:`~numpy.array` [:obj:`float`], optional
        initial_drift : :obj:`~numpy.array` [:obj:`float`], optional
        initial_temp : :obj:`~numpy.array` [:obj:`float`], optional
        name : :obj:`str`, optional
        color : :obj:`str`, optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        """
        # For complete check, assert that all parameters are assigned
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all([param is not None for param in locals().values()])

        # Set up basic constants
        n_categories = len(bp_c.SUPP_BEHAVIOUR_TYPES)
        n_species = None

        # check all parameters, if set
        if behaviour_type is not None:
            assert isinstance(behaviour_type, str)
            assert behaviour_type in bp_c.SUPP_BEHAVIOUR_TYPES

        if initial_rho is not None:
            assert isinstance(initial_rho, np.ndarray)
            assert initial_rho.dtype == float
            assert initial_rho.ndim == 1
            assert np.min(initial_rho) > 0
            if n_species is not None:
                assert initial_rho.shape[0] == n_species
            n_species = initial_rho.shape[0]

        if initial_drift is not None:
            assert isinstance(initial_drift, np.ndarray)
            assert initial_drift.dtype == float
            assert initial_drift.ndim == 2
            if n_species is not None:
                assert initial_drift.shape[0] == n_species
            n_species = initial_drift.shape[0]
            assert initial_drift.shape[1] in [2, 3]

        if initial_temp is not None:
            assert isinstance(initial_temp, np.ndarray)
            assert initial_temp.dtype == float
            assert initial_temp.ndim == 1
            assert np.min(initial_temp) > 0
            if n_species is not None:
                assert initial_temp.shape[0] == n_species
            # n_species = initial_temp.shape[0]

        if name is not None:
            assert isinstance(name, str)

        if color is not None:
            assert isinstance(color, str)
            assert color in bp_c.SUPP_COLORS
        return

    def __str__(self, idx=None):
        """Convert the instance to a string, describing all attributes."""
        description = ''
        if idx is None:
            description += 'Rule = {}\n'.format(self.name)
        else:
            description += 'Rule_{} = {}\n'.format(idx, self.name)
        description += 'Behaviour Type = ' + self.behaviour_type + '\n'
        description += 'Rho:\n\t'
        description += self.initial_rho.__str__() + '\n'
        description += 'Drift:\n\t'
        description += self.initial_drift.__str__().replace('\n', '\n\t') + '\n'
        description += 'Temperature: \n\t'
        description += self.initial_temp.__str__() + '\n'
        return description
