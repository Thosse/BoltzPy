
from . import species as b_spc
from . import grid as b_grd
from . import svgrid as b_svg
import boltzmann.constants as b_const
import boltzmann as b_sim

import numpy as np
import h5py

import os


class Configuration:
    r"""Handles User Input and sets up the Simulation Parameters

    .. todo::
        - write __isequal__ magic methods for configuration (and subclasses)
        - write unittests for save/load(__init__) methods
        - Add Knudsen Number Attribute or Property?

            * Add method to get candidate for characteristic length
            * show this char length in GUI
            * automatically calculate Knudsen number with this information

        - add attribute to svgrid, so that velocity arrays
          can be stored in 2d/3d shape
          (just create a link, with the right shape)
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

        :class:`Species` and :class:`Specimen`

        - add attribute parent/parent_array to specimen
            - add index attribute to specimen
            - move most integrity checks from species to specimen
            - move editing of collision_rate to specimen
            - if index is None, collision_rate is stored locally (for init)
            - if index is int, collision rate is read from matrix from parent
            - when adding specimen to species -> give index and add coll_rate

    Parameters
    ----------
    simulation : :class:`~boltzmann.Simulation`
    file_address : :obj:`str`, optional
        Can be either a full path, a base file name or a file root.

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
    def __init__(self, simulation, file_name=None):
        # Todo assert parameters (sim + file_name -> path + root)
        assert isinstance(simulation, b_sim.Simulation)
        # Todo simulation.check_integrity(complete_check=False)
        self._sim = simulation

        # Todo get file to load data from file_name  or _sim.file_address
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

        * :attr:`Grid.iG` denotes the time steps
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

    #####################################
    #           Configuration           #
    #####################################
    def add_specimen(self,
                     name=None,
                     color=None,
                     mass=None,
                     collision_rate=None):
        """Adds a Specimen to :attr:`~Configuration.s`.
        Directly calls :meth:`Species.add_specimen`

        Parameters
        ----------
        name : :obj:`str`, optional
        color : :obj:`str`, optional
        mass : int, optional
        collision_rate : :obj:`list` of :obj:`float`, optional
            Determines the collision probability between two specimen.
            Row (and column) of :attr:`collision_rate_matrix`.
        """
        self.s.add_specimen(name,
                            color,
                            mass,
                            collision_rate)

    # Todo Choose between step size or number of time steps
    def set_time_grid(self,
                      max_time,
                      number_time_steps,
                      calculations_per_time_step=1):
        """Sets up :attr:`~Configuration.t`.

        1. Calculates step size
        2. Calls :meth:`Grid.setup`
        3. Calls :meth:`Grid.reshape`:
           Changes shape from (1,1) to (1,)

        Parameters
        ----------
        max_time : :obj:`float`
        number_time_steps : :obj:`int`
        calculations_per_time_step : :obj:`int`
        """
        step_size = max_time / (number_time_steps - 1)
        self._t = b_grd.Grid(grid_form='rectangular',
                             grid_dimension=1,
                             grid_shape=[number_time_steps],
                             grid_spacing=step_size,
                             grid_multiplicator=calculations_per_time_step)
        return

    def set_position_grid(self,
                          grid_dimension,
                          grid_shape,
                          grid_spacing):
        """Sets up :attr:`~Configuration.p`.

        Directly calls :meth:`Grid.setup`.

        Parameters
        ----------
        grid_dimension : :obj:`int`
        grid_shape : :obj:`list` [:obj:`int`]
        grid_spacing : :obj:`float`
        """
        if isinstance(grid_shape, int):
            assert grid_dimension == 1
            grid_shape = [grid_shape]
        self._p = b_grd.Grid(grid_form='rectangular',
                             grid_dimension=grid_dimension,
                             grid_shape=grid_shape,
                             grid_spacing=grid_spacing)
        # Update initialization_array
        self._sim.initialization.init_arr = np.full(shape=self.p.size,
                                                    fill_value=-1,
                                                    dtype=int)
        return

    def set_velocity_grids(self,
                           grid_dimension,
                           min_points_per_axis,
                           max_velocity,
                           grid_form='rectangular',
                           velocity_offset=None):
        """Sets up attribute
        :class:`sv <boltzmann.configuration.SVGrid>`.

        1. Generates a minimal :class:`Grid` for the Velocities.
        2. Calls :meth:`SVGrid.setup`
           with the newly generated Velocity :class:`Grid`
           as a parameter.

        Parameters
        ----------
        grid_dimension : :obj:`int`
        min_points_per_axis : :obj:`int`
        max_velocity : :obj:`float`
        grid_form : :obj:`str`, optional
        velocity_offset : :obj:`np.ndarray` [:obj:`float`], optional
        """
        self._sv = b_svg.SVGrid(grid_form=grid_form,
                                grid_dimension=grid_dimension,
                                min_points_per_axis=min_points_per_axis,
                                max_velocity=max_velocity,
                                velocity_offset=velocity_offset,
                                species_array=self.s)
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
            Full path to a :class:`~boltzmann.Simulation` file.
        """
        # Open file
        if file_address is None:
            file_address = self._sim.file_address
        else:
            assert os.path.exists(file_address)
        # Todo Assert it is a simulation file!
        file = h5py.File(file_address, mode='r')

        # Check if Configuration Group exists
        if "Configuration" not in file.keys():
            msg = 'No group "Configuration" found in file:\n' \
                  '{}'.format(file_address)
            raise KeyError(msg)
        file_c = file["Configuration"]
        # load Species
        try:
            self._s = b_spc.Species.load(file_c["Species"])
        except KeyError:
            self._s = b_spc.Species()
        # load Time Space
        try:
            self._t.load(file_c["Time_Space"])
        except KeyError:
            self._t = b_grd.Grid()
        # load Position Space
        try:
            self._p.load(file_c["Position_Space"])
        except KeyError:
            self._p = b_grd.Grid()
        try:
            self._sv.load(file_c["Velocity_Space"],
                          file_c["Species"])
        except KeyError:
            self._sv = b_svg.SVGrid()
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
        self.check_integrity(complete_check=False)
        return

    def save(self, file_address=None):
        """Writes all parameters of the :class:`Configuration` instance
        to the given HDF5-file.

        Parameters
        ----------
        file_address : str, optional
            Full path to a :class:`~boltzmann.Simulation` HDF5-file.
        """
        self.check_integrity()
        if file_address is None:
            file_address = self._sim.file_address
        else:
            # Todo assert the directory exists and check the rights
            pass

        # Open file
        file = h5py.File(file_address, mode='a')

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

        # Save Velocity Space
        file_c.create_group("Velocity_Space")
        self.sv.save(file_c["Velocity_Space"])

        # Save other Parameters
        if self.coll_select_scheme is not None:
            key = "Collision_Selection_Scheme"
            file_c[key] = self.coll_select_scheme
        if self.coll_substeps is not None:
            key = "Collision_Substeps"
            file_c[key] = self.coll_substeps
        if self.conv_order_os is not None:
            key = "Convergence_Order_Operator_Splitting"
            file_c[key] = self.conv_order_os
        if self.conv_order_transp is not None:
            key = "Convergence_Order_Transport"
            file_c[key] = self.conv_order_transp
        if self.conv_order_coll is not None:
            key = "Convergence_Order_Collision_Operator"
            file_c[key] = self.conv_order_coll

        # Save Animated Moments (and shape attribute)
        if self.animated_moments is not None:
            #  noinspection PyUnresolvedReferences
            h5py_string_type = h5py.special_dtype(vlen=str)
            key = "Animated_Moments"
            file_c[key] = np.array(self.animated_moments,
                                   dtype=h5py_string_type).flatten()
            file_c[key].attrs["shape"] = self.animated_moments.shape
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
                              file_address=self._sim.file_address,
                              animated_moments=self.animated_moments,
                              coll_select_scheme=self.coll_select_scheme,
                              coll_substeps=self.coll_substeps,
                              conv_order_os=self.conv_order_os,
                              conv_order_transp=self.conv_order_transp,
                              conv_order_coll=self.conv_order_coll,
                              complete_check=complete_check)
        # Additional Conditions on instance:
        # parameter can be list, instance attributes must be np.ndarray
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
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all([param is not None for param in locals().values()])

        # check all parameters, if set
        if species is not None:
            assert isinstance(species, b_spc.Species)
            species.check_integrity()

        if time_grid is not None:
            assert isinstance(time_grid, b_grd.Grid)
            time_grid.check_integrity(complete_check)
            assert time_grid.dim == 1

        if position_grid is not None:
            assert isinstance(position_grid, b_grd.Grid)
            position_grid.check_integrity(complete_check)
            if position_grid.dim is not 1:
                msg = "Currently only 1D Simulations are supported!"
                raise NotImplementedError(msg)

        if species_velocity_grid is not None:
            assert isinstance(species_velocity_grid, b_svg.SVGrid)
            species_velocity_grid.check_integrity(complete_check)

        if position_grid is not None and species_velocity_grid is not None:
            assert species_velocity_grid.dim >= position_grid.dim

        if species is not None and species_velocity_grid is not None:
            assert species_velocity_grid.n_grids == species.n

        if file_address is not None:
            assert isinstance(file_address, str)
            # remove ending, if any
            if file_address[-5:] == '.hdf5':
                file_address = file_address[:-5]
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
                path = b_const.DEFAULT_DIRECTORY
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
