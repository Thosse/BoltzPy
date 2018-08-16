import boltzpy.constants as b_const

import numpy as np
import h5py


# Todo What exactly is rho? drift and temperature? divided by rho?
class Rule:
    """Encapsulates all data to initialize a :class:`~boltzpy.Grid` point.

    An initialization rule must be applied to every point of the
    :attr:`simulation.p <boltzpy.Simulation>`
    :class:`boltzpy.Grid`.
    It determines:

        * the initial distribution in the velocity space
          based on :attr:`rho`, :attr:`drift`, and :attr:`temp`.
        * the
          :const:`category <boltzpy.constants.SUPP_GRID_POINT_CATEGORIES>`
          of the :class:`P-Grid <boltzpy.configuration.Grid>` point
          which determines the behaviour during the :mod:`computation`

    Parameters
    ----------
    category : :obj:`str`
        Category of the :class:`P-Grid <boltzpy.Grid>` point.
        Must be in :const:`~boltzpy.constants.SUPP_GRID_POINT_CATEGORIES`.
    rho : :obj:`list` [:obj:`float`]
    drift : :obj:`list` [:obj:`float`]
    temp : :obj:`list` [:obj:`float`]
    name : :obj:`str`, optional
    color : :obj:`str`, optional

    Attributes
    ----------
    i_cat : :obj:`int`, optional
        determines the behaviour during the simulation
        Denotes the index of an element of
        :const:`~boltzpy.constants.SUPP_GRID_POINT_CATEGORIES`.
    rho : :obj:`~numpy.ndarray` [:obj:`float`], optional
        Correlates to the total amount of particles in
        the area of the :class:`P-Grid <boltzpy.Grid>` point.
    drift : :obj:`~numpy.ndarray` [:obj:`float`], optional
        Describes the mean velocity,
        i.e. the first moment (expectancy value) of the
        velocity distribution.
    temp : :obj:`~numpy.ndarray` [:obj:`float`], optional
        Correlates to the Energy,
        i.e. the second moment (variance) of the
        velocity distribution.
    name : :obj:`str`, optional
        Is displayed in the GUI to visualize the initialization.
    color : :obj:`str`, optional
        Is Displayed in the GUI to visualize the initialization.
    """
    # Todo get a clear understanding of the meaning of temperature

    # todo Ask Hans Drift and temperature should be equal for all specimen?

    # Todo add attribute and setup method to create initial state

    def __init__(self,
                 category=None,
                 rho=None,
                 drift=None,
                 temp=None,
                 name=None,
                 color=None):
        self.check_parameters(category=category,
                              rho=rho,
                              drift=drift,
                              temp=temp,
                              name=name,
                              color=color)
        if category is not None:
            self.i_cat = b_const.SUPP_GRID_POINT_CATEGORIES.index(category)
        else:
            self.i_cat = None

        self.rho = rho
        self.drift = drift
        self.temp = temp

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
        hdf5_group : :obj:`h5py.Group`

        Returns
        -------
        :class:`Rule`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "Rule"
        self = Rule()

        # read attributes from file
        try:
            category = hdf5_group["Category"].value
            category_idx = b_const.SUPP_GRID_POINT_CATEGORIES.index(category)
            self.i_cat = int(category_idx)
        except KeyError:
            self.i_cat = None
        try:
            self.rho = hdf5_group["Mass"].value
        except KeyError:
            self.rho = None
        try:
            self.drift = hdf5_group["Mean Velocity"].value
        except KeyError:
            self.drift = None
        try:
            self.temp = hdf5_group["Temperature"].value
        except KeyError:
            self.temp = None
        try:
            self.name = hdf5_group["Name"].value
        except KeyError:
            self.name = ''
        try:
            self.color = hdf5_group["Color"].value
        except KeyError:
            self.color = 'black'
        self.check_integrity(False)
        return self

    def save(self, hdf5_group):
        """Write the main parameters of the :class:`Rule` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity(False)

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = "Rule"

        # write all set attributes to file
        if self.i_cat is not None:
            category = b_const.SUPP_GRID_POINT_CATEGORIES[self.i_cat]
            hdf5_group["Category"] = category
        if self.rho is not None:
            hdf5_group["Mass"] = self.rho
        if self.drift is not None:
            hdf5_group["Mean Velocity"] = self.drift
        if self.temp is not None:
            hdf5_group["Temperature"] = self.temp
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
        if self.i_cat is not None:
            category = b_const.SUPP_GRID_POINT_CATEGORIES[self.i_cat]
        else:
            category = None
        self.check_parameters(category=category,
                              category_idx=self.i_cat,
                              rho=self.rho,
                              drift=self.drift,
                              temp=self.temp,
                              name=self.name,
                              color=self.color,
                              complete_check=complete_check)
        return

    @staticmethod
    def check_parameters(category=None,
                         category_idx=None,
                         rho=None,
                         drift=None,
                         temp=None,
                         name=None,
                         color=None,
                         complete_check=False):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        category : :obj:`str`, optional
        category_idx : :obj:`int`, optional
        rho : :obj:`~numpy.array` [:obj:`float`], optional
        drift : :obj:`~numpy.array` [:obj:`float`], optional
        temp : :obj:`~numpy.array` [:obj:`float`], optional
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
        n_categories = len(b_const.SUPP_GRID_POINT_CATEGORIES)
        n_species = None

        # check all parameters, if set
        if category is not None:
            assert isinstance(category, str)
            assert category in b_const.SUPP_GRID_POINT_CATEGORIES
            if category_idx is None:
                category_list = b_const.SUPP_GRID_POINT_CATEGORIES
                category_idx = category_list.index(category)
            else:
                category_list = b_const.SUPP_GRID_POINT_CATEGORIES
                assert category_idx == category_list.index(category)

        if category_idx is not None:
            assert isinstance(category_idx, int)
            assert category_idx in range(n_categories)
            if category is None:
                category_list = b_const.SUPP_GRID_POINT_CATEGORIES
                category = category_list[category_idx]
            else:
                category_list = b_const.SUPP_GRID_POINT_CATEGORIES
                assert category == category_list[category_idx]

        if rho is not None:
            assert isinstance(rho, np.ndarray)
            assert rho.dtype == float
            assert rho.ndim == 1
            assert np.min(rho) > 0
            if n_species is not None:
                assert rho.shape[0] == n_species
            n_species = rho.shape[0]

        if drift is not None:
            assert isinstance(drift, np.ndarray)
            assert drift.dtype == float
            assert drift.ndim == 2
            if n_species is not None:
                assert drift.shape[0] == n_species
            n_species = drift.shape[0]
            assert drift.shape[1] in [2, 3]

        if temp is not None:
            assert isinstance(temp, np.ndarray)
            assert temp.dtype == float
            assert temp.ndim == 1
            np.min(temp) > 0
            if n_species is not None:
                assert temp.shape[0] == n_species
            # n_species = temp.shape[0]

        if name is not None:
            assert isinstance(name, str)

        if color is not None:
            assert isinstance(color, str)
            assert color in b_const.SUPP_COLORS
        return

    def __str__(self, idx=None):
        """Convert the instance to a string, describing all attributes."""
        description = ''
        if idx is None:
            description += 'Rule = {}\n'.format(self.name)
        else:
            description += 'Rule_{} = {}\n'.format(idx, self.name)
        category = b_const.SUPP_GRID_POINT_CATEGORIES[self.i_cat]
        description += 'Category = ' + category + '\n'
        description += 'Rho:\n\t'
        description += self.rho.__str__() + '\n'
        description += 'Drift:\n\t'
        description += self.drift.__str__().replace('\n', '\n\t') + '\n'
        description += 'Temperature: \n\t'
        description += self.temp.__str__() + '\n'
        return description
