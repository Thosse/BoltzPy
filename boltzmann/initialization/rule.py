import boltzmann.constants as b_const

import numpy as np


class Rule:
    """Encapsulates all data of  a single :class:`Initialization` Rule.

    An initialization Rule must be applied to every point of the
    :class:`P-Grid <boltzmann.configuration.Grid>`.
    It determines:

        * the
          :const:`category <boltzmann.constants.SUPP_GRID_POINT_CATEGORIES>`
          of the :class:`P-Grid <boltzmann.configuration.Grid>` point
          and thus its behaviour during the simulation
        * the initial values of each
          :class:`specimens <boltzmann.configuration.Specimen>`
          velocity space at the
          :class:`P-Grid <boltzmann.configuration.Grid>` point point.

    The initialization values are chosen
    to match the conserved quantities
    mass (:attr:`rho`),
    mean velocity (:attr:`drift`)
    and temperature (:attr:`temp`).


    Parameters
    ----------
    category : :obj:`str`
        Category of the :class:`P-Grid <boltzmann.configuration.Grid>` point.
        Must be in :const:`~boltzmann.constants.SUPP_GRID_POINT_CATEGORIES`.
    rho : :obj:`list` [:obj:`float`]
    drift : :obj:`list` [:obj:`float`]
    temp : :obj:`list` [:obj:`float`]
    name : :obj:`str`, optional
    color : :obj:`str`, optional

    Attributes
    ----------
    i_cat : :obj:`int`
        Specifies the behavior in the
        :class:`~boltzmann.calculation.Calculation`.
        Denotes the index of an element of
        :const:`~boltzmann.constants.SUPP_GRID_POINT_CATEGORIES`.
    rho : :obj:`~numpy.ndarray` [:obj:`float`]
        Array of the rho parameters for each specimen.
        Correlates to the total weight/amount of particles in
        the area of the
        :class:`P-Grid <boltzmann.configuration.Grid>` point.
    drift : :obj:`~numpy.ndarray` [:obj:`float`]
        Array of the drift parameters for each specimen.
        Describes the mean velocity,
        i.e. the first moment (expectancy value) of the
        velocity distribution.
    temp : :obj:`~numpy.ndarray` [:obj:`float`]
        Array of the temperature parameters for each specimen.
        Correlates to the Energy,
        i.e. the second moment (variance) of the
        velocity distribution.
    name : :obj:`str`
        Displayed in the GUI to visualize the initialization.
    color : :obj:`str`
        Displayed in the GUI to visualize the initialization.
    """
    # Todo get a clear understanding of the meaning of temperature

    # todo Drift and temperature should be equal for all specimen?

    # Todo add attribute and setup method to create initial state

    def __init__(self,
                 category,
                 rho,
                 drift,
                 temp,
                 name=None,
                 color=None):
        self.check_parameters(category=category,
                              rho=rho,
                              drift=drift,
                              temp=temp,
                              name=name,
                              color=color)
        self.i_cat = b_const.SUPP_GRID_POINT_CATEGORIES.index(category)
        self.rho = np.array(rho,
                            dtype=float)
        self.drift = np.array(drift,
                              dtype=float)
        self.temp = np.array(temp,
                             dtype=float)
        if name is None:
            self.name = ''
        else:
            self.name = name
        if color is None:
            self.color = 'gray'
        else:
            self.color = color
        self.check_integrity()
        return

    #####################################
    #           Serialization           #
    #####################################
    def load(self, hdf5_group):
        """Creates the :class:`Rule` instance,
        based on the parameters in the given group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group`
            Opened HDF5 :obj:`Rule` Group.

        Returns
        -------
        :obj:`Rule`
        """
        # read attributes from file
        category = hdf5_group["Category"].value
        index = b_const.SUPP_GRID_POINT_CATEGORIES.index(category)
        self.i_cat = int(index)
        self.name = hdf5_group["Name"].value
        self.color = hdf5_group["Color"].value
        self.rho = hdf5_group["Mass"].value
        self.drift = hdf5_group["Mean Velocity"].value
        self.temp = hdf5_group["Energy"].value
        self.check_integrity(True)
        return

    def save(self, hdf5_group):
        """Writes the parameters of the :obj:`Rule` instance
        into the given group.

        Parameters
        ----------
        hdf5_group : h5py.Group
            Opened HDF5 :obj:`Rule` group.
        """
        # clear any existing elements in the group
        for key in hdf5_group.keys():
            del hdf5_group[key]

        # Set special data type for String-Arrays
        #  noinspection PyUnresolvedReferences
        h5py_string_type = h5py.special_dtype(vlen=str)
        # Write Attributes
        category = b_const.SUPP_GRID_POINT_CATEGORIES[self.i_cat]
        hdf5_group["Category"] = np.array(category,
                                          dtype=h5py_string_type)
        hdf5_group["Name"] = np.array(self.name,
                                      dtype=h5py_string_type)
        hdf5_group["Color"] = np.array(self.color,
                                       dtype=h5py_string_type)

        hdf5_group["Mass"] = self.rho
        hdf5_group["Mean Velocity"] = self.drift
        hdf5_group["Energy"] = self.rho
        return

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self, complete_check=True):
        """Sanity Check.
        Besides asserting all conditions in :meth:`check_parameters`
        it asserts the correct type of all attributes of the instance.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all attributes must be set (not None).
            If False, then unassigned attributes are ignored.
        """
        category = b_const.SUPP_GRID_POINT_CATEGORIES[self.i_cat]
        self.check_parameters(category=category,
                              category_index=self.i_cat,
                              rho=self.rho,
                              drift=self.drift,
                              temp=self.temp,
                              name=self.name,
                              color=self.color,
                              complete_check=complete_check)
        # Additional Conditions on instance:
        # parameters can also be a list,
        # instance attributes must be ndarray
        assert isinstance(self.rho, np.ndarray)
        assert isinstance(self.drift, np.ndarray)
        assert isinstance(self.temp, np.ndarray)
        # parameters can also be list/array of ints,
        # instance attribute must be ndarray of floats
        assert self.rho.dtype == float
        assert self.drift.dtype == float
        assert self.temp.dtype == float
        return

    @staticmethod
    def check_parameters(category=None,
                         category_index=None,
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
        category_index : :obj:`int`, optional
        rho : :obj:`list` [:obj:`float`], optional
        drift : :obj:`list` [:obj:`float`], optional
        temp : :obj:`list` [:obj:`float`], optional
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

        if category_index is not None:
            assert isinstance(category_index, int)
            assert category_index in range(n_categories)

        if rho is not None:
            if isinstance(rho, list):
                rho = np.array(rho)
            assert isinstance(rho, np.ndarray)
            assert rho.dtype in [int, float]
            assert rho.ndim == 1
            assert np.min(rho) > 0
            if n_species is not None:
                assert rho.shape[0] == n_species
            n_species = rho.shape[0]

        if drift is not None:
            if isinstance(drift, list):
                drift = np.array(drift)
            assert isinstance(drift, np.ndarray)
            assert drift.dtype in [int, float]
            assert drift.ndim == 2
            if n_species is not None:
                assert drift.shape[0] == n_species
            n_species = drift.shape[0]
            assert drift.shape[1] in [2, 3]

        if temp is not None:
            if isinstance(temp, list):
                temp = np.array(temp)
            assert isinstance(temp, np.ndarray)
            assert temp.dtype in [int, float]
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

    # Todo replace by __str__ method
    def print(self, list_of_category_names=None):
        """Prints all Properties for Debugging Purposes

        If a list of category names is given,
        then the category of the :class:`Rule` is printed.
        Otherwise the category index
        (:attr:`cat`) is printed"""
        print('Name of Rule = {}'.format(self.name))
        if list_of_category_names is not None:
            print('Category = {}'
                  ''.format(list_of_category_names[self.i_cat]))
        else:
            print('Category index = {}'.format(self.i_cat))
        print('Rho: '
              '{}'.format(self.rho))
        print('Drift:\n'
              '{}'.format(self.drift))
        print('Temperature: '
              '{}'.format(self.temp))
        print('')
        return
