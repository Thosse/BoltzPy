
from . import species as b_spc
from . import grid as b_grd
from . import svgrid as b_svg

import numpy as np
import h5py

import os
from sys import stdout as stdout


class Configuration:
    r"""Handles User Input and sets up the Simulation Parameters

    .. todo::
        - Add Knudsen Number Attribute or Property?
        - improve name and path attributes:
          Currently you need to change path first, then name,
          otherwise unnecessary folders are created in the old path
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

    """
    # Todo Give Warning, when Specifically using "default"
    def __init__(self, file_name="default"):
        # Most Attributes are Private and set up separately
        # Read Only Properties
        self._s = b_spc.Species()
        self._t = b_grd.Grid()
        self._p = b_grd.Grid()
        self._sv = b_svg.SVGrid()
        # Todo remove default parameters -> None
        # Todo add properties supported_conv_order_XXX
        # Default Parameters
        self._coll_select_scheme = 'Complete'
        self._coll_substeps = 1
        self._conv_order_os = 1
        self._conv_order_transp = 1
        self._conv_order_coll = 1
        self._animated_moments = np.array([['Mass',
                                            'Momentum_X'],
                                           ['Momentum_X',
                                            'Momentum_Flow_X'],
                                           ['Energy',
                                            'Energy_Flow_X']])
        # Setup HDF5 File (stores all Configuration Data)
        self._fileAddress = ''
        self.file_address = file_name
        # Load non-default file, if it exists
        if file_name != "default" and os.path.exists(self.file_address):
            self.load()
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
    def supported_output(self):
        """:obj:`set` of :obj:`str` :
        Set of all currently supported moments.
        """
        supported_output = {'Mass',
                            'Momentum_X',
                            'Momentum_Y',
                            'Momentum_Z',
                            'Momentum_Flow_X',
                            'Momentum_Flow_Y',
                            'Momentum_Flow_Z',
                            'Energy',
                            'Energy_Flow_X',
                            'Energy_Flow_Y',
                            'Energy_Flow_Z'}
        return supported_output

    @property
    def supported_selection_schemes(self):
        """:obj:`set` of :obj:`str` :
        Set of all currently supported selection schemes
        for :class:`~boltzmann.calculation.Collisions`.
        """
        supported_selection_schemes = {'Complete'}
        return supported_selection_schemes

    @property
    def animated_moments(self):
        """:obj:`~numpy.ndarray` of :obj:`str` :
        Array of the moments to be stored and animated.

        Every single moment is an element of :attr:`supported_output`.
        """
        return self._animated_moments

    @animated_moments.setter
    def animated_moments(self, array_of_moments):
        # Todo move into Check integrity for all setters
        # + check that shape is 2d
        for mom in array_of_moments.flatten():
            if mom not in self.supported_output:
                message = "Unsupported Output: {}" \
                          "".format(mom)
                raise AttributeError(message)
        self._animated_moments = np.array(array_of_moments)
        return

    @property
    def coll_select_scheme(self):
        """:obj:`str` :
        Selection Scheme for :class:`~boltzmann.calculation.Collisions`,
        is an element of :attr:`supported_selection_schemes`"""
        return self._coll_select_scheme

    @coll_select_scheme.setter
    def coll_select_scheme(self, scheme):
        if type(scheme) is not str:
            message = 'Invalid Type ' \
                      'for Collision Selection Scheme:' \
                      '{}'.format(type(scheme))
            raise TypeError(message)
        if scheme not in self.supported_selection_schemes:
            message = 'Unsupported Selection Scheme:' \
                      '{}'.format(self.coll_select_scheme)
            raise ValueError(message)
        self._coll_select_scheme = scheme
        return

    @property
    def coll_substeps(self):
        """":obj:`int`:
            Number of Collision Steps in each Time Step
        """
        return self._coll_substeps

    @coll_substeps.setter
    def coll_substeps(self, number_of_steps):
        if type(number_of_steps) is not int:
            message = 'Invalid Type for Collision sub steps:' \
                      '{}'.format(type(number_of_steps))
            raise TypeError(message)
        if number_of_steps < 0:
            message = 'Invalid Number of Collision sub steps:' \
                      '{}'.format(number_of_steps)
            raise ValueError(message)
        self._coll_substeps = number_of_steps
        return

    @property
    def conv_order_os(self):
        """":obj:`int` in [1, 2]:
            Convergence Order of Operator Splitting
        """
        return self._conv_order_os

    @conv_order_os.setter
    def conv_order_os(self, conv_order):
        if type(conv_order) is not int:
            msg = 'Invalid Type for Convergence Order:' \
                      '{}'.format(type(conv_order))
            raise TypeError(msg)
        if conv_order not in [1, 2]:
            message = 'Invalid Convergence Order:' \
                      '{}'.format(conv_order)
            raise ValueError(message)
        if conv_order != 1:
            raise NotImplementedError
        self._conv_order_os = conv_order
        return

    @property
    def conv_order_coll(self):
        """":obj:`int` in [1, 2]:
            Convergence Order of Quadrature Formula
            for Approximation of the Collision Operator.
        """
        return self._conv_order_coll

    @conv_order_coll.setter
    def conv_order_coll(self, conv_order):
        if type(conv_order) is not int:
            msg = 'Invalid Type for Convergence Order:' \
                  '{}'.format(type(conv_order))
            raise TypeError(msg)
        if conv_order not in [1, 2, 3]:
            message = 'Invalid Convergence Order:' \
                      '{}'.format(conv_order)
            raise ValueError(message)
        if conv_order != 1:
            raise NotImplementedError
        self._conv_order_coll = conv_order
        return

    @property
    def conv_order_transp(self):
        """":obj:`int` in [1, 2]:
            Convergence Order of Transport Step
            (Partial Differential Equation)
        """
        return self._conv_order_transp

    @conv_order_transp.setter
    def conv_order_transp(self, conv_order):
        if type(conv_order) is not int:
            msg = 'Invalid Type for Convergence Order:' \
                  '{}'.format(type(conv_order))
            raise TypeError(msg)
        if conv_order not in [1, 2]:
            message = 'Invalid Convergence Order:' \
                      '{}'.format(conv_order)
            raise ValueError(message)
        if conv_order != 1:
            raise NotImplementedError
        self._conv_order_transp = conv_order
        return

    @property
    def file_address(self):
        """:obj:`str` :
        Path to this Configuration file.
        Important for reading and writing to/from HDD.
        """
        return self._fileAddress

    @file_address.setter
    def file_address(self, new_address):
        assert type(new_address) is str
        # isolate path from address
        sep = new_address.rfind('/')
        if sep == -1:
            # Todo Setup proper default path for simulations
            # Todo Write default path into setup.py?
            # set default path
            path = __file__[:-40] + 'Simulations/'
        elif sep == len(new_address) - 1:
            msg = 'The provided file address is invalid:' \
                  'It is a Folder, not a File:\n' \
                  '{}'.format(new_address)
            raise ValueError(msg)
        else:
            path = new_address[0:sep+1]
        # Assert file path exists
        if not os.path.exists(path):
            message = 'The specified file path does not exist.' \
                      'You need to create it first: ' \
                      '{}'.format(path)
            raise FileNotFoundError(message)

        # isolate file name
        file_name = new_address[sep+1:]
        # temporary remove '.sim' ending (if any)
        if file_name[-4:] == '.sim':
            file_name = file_name[:-4]
        # Assert validity of file name
        invalid_chars = ['.', '"', "'", '/', '§', '$', '&']
        for char in invalid_chars:
            if char in file_name:
                msg = 'The provided file name is invalid: ' \
                      'It contains invalid characters: "{}"\n' \
                      '{}'.format(char, file_name)
                raise ValueError(msg)
        # Add '.sim' ending to file name again
        file_name += '.sim'
        self._fileAddress = path + file_name
        return

    #####################################
    #           Configuration           #
    #####################################
    def add_specimen(self,
                     **kwargs):
        """Adds a Specimen to :attr:`~Configuration.s`.

        Directly calls :meth:`Species.add_specimen`
        """
        self.s.add_specimen(**kwargs)

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
        # TODO self._sv = b_svg.SVGrid()
        # Default Parameters
        try:
            key = "Collision_Selection_Scheme"
            self.coll_select_scheme = file_c[key].value
        except KeyError:
            self._coll_select_scheme = None
        try:
            key = "Collision_Substeps"
            self.coll_substeps = int(file_c[key].value)
        except KeyError:
            self._coll_substeps = None
        try:
            key = "Convergence_Order_Operator_Splitting"
            self.conv_order_os = int(file_c[key].value)
        except KeyError:
            self._conv_order_os = None
        try:
            key = "Convergence_Order_Transport"
            self.conv_order_transp = int(file_c[key].value)
        except KeyError:
            self._conv_order_transp = None
        try:
            key = "Convergence_Order_Collision_Operator"
            self.conv_order_coll = int(file_c[key].value)
        except KeyError:
            self._conv_order_coll = None
        try:
            key = "Animated_Moments"
            shape = file_c[key].attrs["shape"]
            self.animated_moments = file_c[key].value.reshape(shape)
        except KeyError:
            self._animated_moments = None
        file.close()
        # Todo Ignore None/Unset Values for check Integrity?
        # self.check_integrity()
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
        file_c["Collision_Selection_Scheme"] = self.coll_select_scheme
        file_c["Collision_Substeps"] = self.coll_substeps
        # Todo move into subgroup or attributes?
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
    def check_integrity(self):
        """Sanity Check. Checks Integrity of all Attributes"""
        self.s.check_integrity()
        self.t.check_integrity()
        self.p.check_integrity()
        self.sv.check_integrity()
        assert self.t.dim == 1
        assert self.t.G.shape == (self.t.size,)
        assert self.sv.dim >= self.p.dim
        assert self.sv.size.size == self.s.n
        assert all([type(_) is str for _ in self._fileAddress])
        return

    def print(self,
              physical_grids=False):
        """Prints all Properties for Debugging Purposes"""
        print('\n========CONFIGURATION========\n')
        print('Configuration File Address: {}'.format(self.file_address))
        print('Animated Moments:\n{}'.format(self.animated_moments))
        print('Collision Selection Scheme: '
              '{}'.format(self.coll_select_scheme))
        print('Collision Steps per Time Step: {}'
              ''.format(self.coll_substeps))
        print('\nSpecimen:')
        print('---------')
        self.s.print()
        print('')
        print('Time Data:')
        print('----------')
        self.t.print(physical_grids)
        print('Position-Space Data:')
        print('--------------------')
        self.p.print(physical_grids)
        print('Velocity-Space Data:')
        print('--------------------')
        self.sv.print(physical_grids)
        stdout.flush()
