
from . import species as b_spc
from . import grid as b_grd
from . import svgrid as b_svg
import boltzmann.constants as b_const

import numpy as np
import h5py

import os


class Configuration:
    r"""Handles User Input and sets up the Simulation Parameters

    .. todo::
        - allow save/load functions for incompletely initialized
          Configurations / Initializations
        - Add Knudsen Number Attribute or Property?
        - link Species and SVGrid somehow
          -> adding Species, after setting up SVGrid
          should delete SVGrid or at least update it

          * Idea: each class has an is_set_up flag
          * after any change -> check flags of depending classes
          * main classes need to be linked for that!

        :class:`Grid`

        - Add unit tests
        - Add Circular shape
        - Add rotation of grid (useful for velocities)
        - Enable non-uniform/adaptive Grids
          (see :class:`~boltzmann.calculation.Calculation`)
        - Add Plotting-function to grids

    Parameters
    ----------
    file_name

    Attributes
    ----------
    coll_select_scheme : :obj:`str`
        Selection scheme for the collision partners.
        Must be an element of
        :const:`~boltzmann.constants.SUPP_COLL_SELECTION_SCHEMES`
    coll_substeps : :obj:`int`
        Number of collision substeps per time step.
    conv_order_os : :obj:`int`
        Convergence Order of Operator Splitting.
        Must be in :const:`~boltzmann.constants.SUPP_ORDERS_OS`.
    conv_order_coll : :obj:`int`
        Convergence Order of Quadrature Formula
        for Approximation of the Collision Operator.
        Must be in
        :const:`~boltzmann.constants.SUPP_ORDERS_COLL`.
    conv_order_transp : :obj:`int` :
        Convergence Order of Transport Step (PDE).
        Must be in
        :const:`~boltzmann.constants.SUPP_ORDERS_TRANSP`.
    """
    # Todo Give Warning, when Specifically using "default"
    # Todo move into constant? Change to None?
    def __init__(self, file_name="default"):
        # Most Attributes are Private and set up separately
        # Read Only Properties
        self._s = b_spc.Species()
        self._t = b_grd.Grid()
        self._p = b_grd.Grid()
        self._sv = b_svg.SVGrid()
        # Todo remove default parameters -> None
        # Default Parameters
        # Todo Move parameters into small Numerical_Scheme class?
        self.coll_select_scheme = 'Complete'
        self.coll_substeps = 1
        self.conv_order_os = 1
        self.conv_order_transp = 1
        self.conv_order_coll = 1
        self._animated_moments = np.array([['Mass',
                                            'Momentum_X'],
                                           ['Momentum_X',
                                            'Momentum_Flow_X'],
                                           ['Energy',
                                            'Energy_Flow_X']])
        # Setup HDF5 File (stores all Configuration Data)
        self._file_address = ''
        self.file_address = file_name

        # Load non-default file, if it exists
        if file_name != "default" and os.path.exists(self.file_address):
            self.load()
        # Todo this should be changed
        # -> compare the current configuration, to the saved one.
        # If its equal, then there is no need to run the calculation again
        # Clear default file, if it exists
        elif file_name == "default" and os.path.exists(self.file_address):
            os.remove(self.file_address)
        # Todo Add optional Description of Simulation -> very long String
        return

    @property
    def s(self):
        """:obj:`Species` :
        Contains all data about the simulated Specimen.
        """
        return self._s

    @property
    def t(self):
        """:obj:`Grid` :
        Contains all data about simulation time and time step size.

        * :attr:`Grid.G` denotes the time steps
          at which the results are written out to HDD
        * :attr:`Grid.multi` denotes the number of calculation steps
          between two writes.

        """
        return self._t

    @property
    def p(self):
        """:obj:`Grid` :
        Contains all data about Position-Space.
        """
        return self._p

    @property
    def sv(self):
        """:obj:`SVGrid` :
        Contains all data about the Velocity-Space of each Specimen.
        V-Spaces of distinct Specimen differ in step size
        and number of grid points.
        Maximum physical values may differ slightly between specimen.
        """
        return self._sv

    @property
    def animated_moments(self):
        """:obj:`~numpy.ndarray` of :obj:`str` :
        Array of the moments to be stored and animated.

        Each moment is an element of
        :const:`~boltzmann.constants.SUPP_OUTPUT`.
        """
        return self._animated_moments

    @animated_moments.setter
    def animated_moments(self, array_of_moments):
        if type(array_of_moments) is list:
            array_of_moments = np.array(array_of_moments)
        self.check_parameters(animated_moments=array_of_moments)
        self._animated_moments = array_of_moments
        return

    @property
    def file_address(self):
        """:obj:`str` :
        Path to this Configuration file.
        Important for reading and writing to/from HDD.
        """
        return self._file_address

    @file_address.setter
    def file_address(self, new_address):
        self.check_parameters(file_address=new_address)
        # if no path given, add default path
        if new_address.rfind('/') == -1:
            new_address = b_const.DEFAULT_SIMULATION_PATH + new_address
        if new_address[-4:] != '.sim':
            new_address += '.sim'
        self._file_address = new_address
        return

    #####################################
    #           Configuration           #
    #####################################
    # TODO properly add docu and parameters
    def add_specimen(self,
                     **kwargs):
        """Adds a Specimen to :attr:`~Configuration.s`.
        Directly calls :meth:`Species.add_specimen`
        """
        self.s.add_specimen(**kwargs)

    # Todo Choose between step size or number of time steps
    def configure_time(self,
                       max_time,
                       number_time_steps,
                       calculations_per_time_step=1):
        """Sets up :attr:`~Configuration.t`.

        1. Calculates step size
        2. Calls :meth:`Grid.setup`
        3. Calls :meth:`Grid.reshape`:
           Changes shape from (1,1) to (1,)
        """
        step_size = max_time / (number_time_steps - 1)
        self.t.setup(1,
                     [number_time_steps],
                     step_size,
                     multi=calculations_per_time_step)
        self.t.reshape((self.t.size,))
        return

    def configure_position_space(self,
                                 dimension,
                                 list_number_of_points_per_dimension,
                                 step_size):
        """Sets up :attr:`~Configuration.p`.

        Directly calls :meth:`Grid.setup`.

        Parameters
        ----------
        dimension : :obj:`int`
        list_number_of_points_per_dimension : :obj:`list` of :obj:`int`
        step_size : :obj:`float`
        """
        self.p.setup(dimension,
                     list_number_of_points_per_dimension,
                     step_size)
        return

    def configure_velocity_space(self,
                                 dimension,
                                 grid_points_x_axis,
                                 max_v,
                                 shape='rectangular',
                                 offset=None):
        """Sets up :attr:`~Configuration.sv`.

        1. Generates a default Velocity :class:`Grid`
        2. Calls :meth:`SVGrid.setup`
           with the newly generated Velocity :class:`Grid`
           as a parameter
        """
        step_size = 2 * max_v / (grid_points_x_axis - 1)
        number_of_points_per_dimension = [grid_points_x_axis] * dimension
        v = b_grd.Grid()
        v.setup(dimension,
                number_of_points_per_dimension,
                step_size,
                shape)
        self.sv.setup(self.s,
                      v,
                      offset)
        return

    #####################################
    #           Serialization           #
    #####################################
    def load(self, file_address=None):
        """Sets all parameters of the :obj:`Configuration` instance
        to the ones specified in the given HDF5-file.

        Parameters
        ----------
        file_address : str
            Complete path to the HDF5-file.
        """
        if file_address is None:
            file_address = self.file_address
        elif file_address != self.file_address:
            # Todo change file_address of self? IS this useful/harmful?
            raise NotImplementedError
        file = h5py.File(file_address, mode='r')
        if "Configuration" not in file.keys():
            msg = 'No group "Configuration" found in file:\n' \
                  '{}'.format(self.file_address)
            raise KeyError(msg)
        file_c = file["Configuration"]
        # load Species
        try:
            self._s = b_spc.Species.load(file_c["Species"])
        except KeyError:
            self._s = b_spc.Species()
        # load Time Space
        try:
            self._t = b_grd.Grid.load(file_c["Time_Space"])
            self._t.reshape((self.t.size,))
        except KeyError:
            self._t = b_grd.Grid()
        # load Position Space
        try:
            self._p = b_grd.Grid.load(file_c["Position_Space"])
        except KeyError:
            self._p = b_grd.Grid()
        # TODO self._sv = b_svg.SVGrid() is not implemented so far
        # Default Parameters
        try:
            key = "Collision_Selection_Scheme"
            self.coll_select_scheme = file_c[key].value
        except KeyError:
            self.coll_select_scheme = None
        try:
            key = "Collision_Substeps"
            self.coll_substeps = int(file_c[key].value)
        except KeyError:
            self.coll_substeps = None
        try:
            key = "Convergence_Order_Operator_Splitting"
            self.conv_order_os = int(file_c[key].value)
        except KeyError:
            self.conv_order_os = None
        try:
            key = "Convergence_Order_Transport"
            self.conv_order_transp = int(file_c[key].value)
        except KeyError:
            self.conv_order_transp = None
        try:
            key = "Convergence_Order_Collision_Operator"
            self.conv_order_coll = int(file_c[key].value)
        except KeyError:
            self.conv_order_coll = None
        try:
            key = "Animated_Moments"
            shape = file_c[key].attrs["shape"]
            self.animated_moments = file_c[key].value.reshape(shape)
        except KeyError:
            self._animated_moments = None
        file.close()
        self.check_integrity()
        return

    def save(self, file_address=None):
        """Writes all parameters of the :obj:`Configuration` object
        to the given HDF5-file.

        Parameters
        ----------
        file_address : str, optional
            Complete path to a :class:`Configuration` (.sim)
            :attr:`~Configuration.file`.
        """
        # Todo Add Overwrite Protection
        self.check_integrity()
        if file_address is None:
            file_address = self.file_address
        else:
            # Check if file exists, don't overwrite existing files!
            raise NotImplementedError
        # Open file
        file = h5py.File(file_address, mode='w')
        # Clear currently saved Configuration, if any
        if "Configuration" in file.keys():
            del file["Configuration"]
        # Create and open empty "Configuration" group
        file.create_group("Configuration")
        file_c = file["Configuration"]
        # Save Species
        file_c.create_group("Species")
        self.s.save(file_c["Species"])
        # Save Time Space
        file_c.create_group("Time_Space")
        self.t.save(file_c["Time_Space"])
        # Save Position Space
        file_c.create_group("Position_Space")
        self.p.save(file_c["Position_Space"])
        # TODO self._sv = b_svg.SVGrid()
        # Save Velocity Space
        # file_c.create_group("Velocity_Space")
        # self.p.save(file_c["Velocity_Space"])
        # Save other Parameters
        # Todo only save, if not None?
        file_c["Collision_Selection_Scheme"] = self.coll_select_scheme
        file_c["Collision_Substeps"] = self.coll_substeps
        file_c["Convergence_Order_Operator_Splitting"] = self.conv_order_os
        file_c["Convergence_Order_Transport"] = self.conv_order_transp
        file_c["Convergence_Order_Collision_Operator"] = self.conv_order_coll
        # Save Animated Moments (and shape attribute)
        #  noinspection PyUnresolvedReferences
        h5py_string_type = h5py.special_dtype(vlen=str)
        file_c["Animated_Moments"] = np.array(self.animated_moments,
                                              dtype=h5py_string_type).flatten()
        _shape = self.animated_moments.shape
        file_c["Animated_Moments"].attrs["shape"] = _shape
        file.close()
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
        self.check_parameters(species=self._s,
                              time_grid=self._t,
                              position_grid=self._p,
                              species_velocity_grid=self._sv,
                              file_address=self.file_address,
                              animated_moments=self.animated_moments,
                              coll_select_scheme=self.coll_select_scheme,
                              coll_substeps=self.coll_substeps,
                              conv_order_os=self.conv_order_os,
                              conv_order_transp=self.conv_order_transp,
                              conv_order_coll=self.conv_order_coll,
                              complete_check=complete_check)
        # Additional Conditions on instance:
        # parameter can be list, instance attributes must be nd.array
        assert isinstance(self.animated_moments, np.ndarray)
        return

    @staticmethod
    def check_parameters(species=None,
                         time_grid=None,
                         position_grid=None,
                         species_velocity_grid=None,
                         file_address=None,
                         animated_moments=None,
                         coll_select_scheme=None,
                         coll_substeps=None,
                         conv_order_os=None,
                         conv_order_transp=None,
                         conv_order_coll=None,
                         complete_check=False):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        species : :obj:`Species`, optional
        time_grid : :obj:`Grid`, optional
        position_grid : :obj:`Grid`, optional
        species_velocity_grid : :obj:`SVGrid`, optional
        file_address : str, optional
        animated_moments : list of lists or np.ndarray(2d), optional
        coll_select_scheme : str, optional
        coll_substeps : int, optional
        conv_order_os : int, optional
        conv_order_transp : int, optional
        conv_order_coll : int, optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        """
        # For complete check, assert that all parameters are assigned
        if complete_check is True:
            assert all([param is not None for param in locals().values()])
        else:
            assert isinstance(complete_check, bool)

        # check all parameters, if set
        if species is not None:
            assert isinstance(species, b_spc.Species)
            species.check_integrity()

        if time_grid is not None:
            assert isinstance(time_grid, b_grd.Grid)
            time_grid.check_integrity()
            assert time_grid.dim == 1
            assert time_grid.G.shape == (time_grid.size,)

        if position_grid is not None:
            assert isinstance(position_grid, b_grd.Grid)
            position_grid.check_integrity()

        if species_velocity_grid is not None:
            assert isinstance(species_velocity_grid, b_svg.SVGrid)
            species_velocity_grid.check_integrity()

        if position_grid is not None and species_velocity_grid is not None:
            assert species_velocity_grid.dim >= position_grid.dim

        if species is not None and species_velocity_grid is not None:
            assert species_velocity_grid.size.size == species.n

        if file_address is not None:
            assert type(file_address) is str
            # remove ending, if any
            if file_address[-4:] == '.sim':
                file_address = file_address[:-4]
            # find separator of file_path and file_name
            sep_pos = file_address.rfind('/')
            # isolate file_name
            file_name = file_address[sep_pos + 1:]
            # file_address must not end  with '/'
            assert sep_pos != len(file_address) - 1,    \
                'The provided file address is invalid:' \
                'It is a Folder, not a File:\n' \
                '{}'.format(file_address)
            # Assert validity of file name
            for char in b_const.INVALID_CHARACTERS:
                assert char not in file_name,   \
                    "The provided file name is invalid:\n" \
                    "It contains invalid characters: '{}'" \
                    "{}".format(char, file_name)
            # isolate file_path
            path = file_address[0:sep_pos + 1]
            if path == "":
                path = b_const.DEFAULT_SIMULATION_PATH
            # Assert file path exists
            assert os.path.exists(path), \
                'The specified file path does not exist.' \
                'You need to create it first:\n{}'.format(path)

        if animated_moments is not None:
            # lists are also accepted as parameters
            if isinstance(animated_moments, list):
                animated_moments = np.array(animated_moments)
            assert isinstance(animated_moments, np.ndarray)
            assert len(animated_moments.shape) is 2
            assert all([mom in b_const.SUPP_OUTPUT
                        for mom in animated_moments.flatten()])

        if coll_select_scheme is not None:
            assert isinstance(coll_select_scheme, str)
            selection_schemes = b_const.SUPP_COLL_SELECTION_SCHEMES
            assert coll_select_scheme in selection_schemes

        if coll_substeps is not None:
            assert isinstance(coll_substeps, int)
            assert coll_substeps >= 0

        if conv_order_os is not None:
            assert isinstance(conv_order_os, int)
            assert conv_order_os in b_const.SUPP_ORDERS_OS
            if conv_order_os != 1:
                raise NotImplementedError

        if conv_order_coll is not None:
            assert isinstance(conv_order_coll, int)
            assert conv_order_coll in b_const.SUPP_ORDERS_COLL
            if conv_order_coll != 1:
                raise NotImplementedError

        if conv_order_transp is not None:
            assert isinstance(conv_order_transp, int)
            assert conv_order_transp in b_const.SUPP_ORDERS_TRANSP
            if conv_order_transp != 1:
                raise NotImplementedError
        return

    def __str__(self,
                write_physical_grids=False):
        """Converts the instance to a string, describing all attributes."""
        description = ''
        description += '========CONFIGURATION========\n'
        description += 'Configuration File Address:\n '
        description += '\t' + self.file_address
        description += '\n'
        description += 'Animated Moments:\n'
        moment_string = self.animated_moments.__str__()
        description += '\t' + moment_string.replace('\n', '\n\t')
        description += '\n'
        description += 'Collision Selection Scheme = ' \
                       '{}'.format(self.coll_select_scheme)
        description += '\n'
        description += 'Collision Steps per Time Step = ' \
                       '{}'.format(self.coll_substeps)
        description += '\n'
        description += '\n'
        description += 'Specimen:\n'
        description += '---------\n'
        description += '\t' + self.s.__str__().replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Time Data:\n'
        description += '----------\n'
        time_string = self.t.__str__(write_physical_grids)
        description += '\t' + time_string.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Position-Space Data:\n'
        description += '--------------------\n'
        position_string = self.p.__str__(write_physical_grids)
        description += '\t' + position_string.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Velocity-Space Data:\n'
        description += '--------------------\n'
        velocity_string = self.sv.__str__(write_physical_grids)
        description += '\t' + velocity_string.replace('\n', '\n\t')
        return description
