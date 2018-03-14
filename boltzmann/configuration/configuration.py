from . import species as b_spc
from . import grid as b_grd
from . import svgrid as b_svg
from . import collisions as b_col

import numpy as np

from sys import stdout as stdout


class Configuration:
    r"""Handles User Input and sets up the Simulation Parameters

    .. todo::
        - add proper file_name initialization/property
          config_file_name : :obj:`str`
          file_path : :obj:`str`
        - link Species and SVGrid somehow
          -> adding Species, after setting up SVGrid
          should delete SVGrid or at least update it

          * Idea: each class has an is_set_up flag
          * after any change -> check flags of depending classes
          * main classes need to be linked for that!

        - Add Plotting-function to grids
    """
    def __init__(self):
        # Most Attributes are Private and set up separately
        # Read Only Properties
        self._s = b_spc.Species()
        self._t = b_grd.Grid()
        self._p = b_grd.Grid()
        self._sv = b_svg.SVGrid()
        self._cols = b_col.Collisions(self)
        # Read-Write Properties
        self._animated_moments = np.array([['Mass',
                                            'Momentum_X'],
                                           ['Momentum_X',
                                            'Momentum_Flow_X'],
                                           ['Energy',
                                            'Energy_Flow_X']])
        self._collision_selection_scheme = 'Complete'
        # Todo self.knudsen_number = 1
        self.collision_steps_per_time_step = 1
        self.order_operator_splitting = 1
        self.order_transport = 1
        self.order_collision = 1
        # Todo self._config_file_name = None
        # Todo self.file_path = None
        return

    @property
    def s(self):
        """:obj:`Species`:
        Contains all data about the simulated Specimen.
        """
        return self._s

    @property
    def t(self):
        """:obj:`Grid`
        Contains all data about simulation time and time step size.

        * :attr:`Grid.G` denotes the time steps
          at which the results are written out to HDD
        * :attr:`Grid.multi` denotes the number of calculation steps
          between two writes.

        """
        return self._t

    @property
    def p(self):
        """:obj:`Grid`
        Contains all data about Position-Space.
        """
        return self._p

    @property
    def sv(self):
        """:obj:`SVGrid`
        Contains all data about the Velocity-Space of each Specimen.
        V-Spaces of distinct Specimen differ in step size
        and number of grid points.
        Maximum physical values may differ slightly between specimen.
        """
        return self._sv

    @property
    def cols(self):
        """:obj:`Collisions`
        Describes the collisions on :attr:`sv.G`.
         """
        return self._cols

    @property
    def supported_output(self):
        """:obj:`set` of :obj:`str`:
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
        """:obj:`set` of :obj:`str`:
        Set of all currently supported selection schemes
        for collisions.
        """
        supported_selection_schemes = {'Complete'}
        return supported_selection_schemes

    @property
    def animated_moments(self):
        """:obj:`~numpy.ndarray` of :obj:`str`:
        Array of the moments to be stored and animated.

        Every single moment is an element of :attr:`supported_output`.
        """
        return self._animated_moments

    @animated_moments.setter
    def animated_moments(self, array_of_moments):
        for mom in array_of_moments.flatten():
            if mom not in self.supported_output:
                message = "Unsupported Output: {}" \
                          "".format(mom)
                raise AttributeError(message)
        self._animated_moments = np.array(array_of_moments)
        return

    @property
    def collision_selection_scheme(self):
        """:obj:`str`:
        Selection Scheme for Collisions,
        is an element of :attr:`supported_selection_schemes`"""
        return self._collision_selection_scheme

    @collision_selection_scheme.setter
    def collision_selection_scheme(self, scheme):
        if scheme not in self.supported_selection_schemes:
            message = 'Unsupported Selection Scheme:' \
                      '{}'.format(self.collision_selection_scheme)
            raise AttributeError(message)
        self._collision_selection_scheme = scheme
        return

    #####################################
    #           Configuration           #
    #####################################
    def add_specimen(self,
                     mass=1,
                     alpha_list=None,
                     name=None,
                     color=None):
        """Adds a Specimen to :attr:`~Configuration.s`.

        Directly calls :meth:`Species.add_specimen`
        """
        self.s.add_specimen(mass,
                            alpha_list,
                            name,
                            color)

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

    # Todo move to calculation
    def setup(self):
        """Prepares Configuration for Initialization,
         by generating the Collisions"""
        self.check_integrity()
        self.cols.setup()
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
        return

    def print(self,
              physical_grids=False):
        """Prints all Properties for Debugging Purposes"""
        print('\n========CONFIGURATION========\n')
        print('Configuration Name: ')  # Todo + self.file_name)
        print('Animated Moments:\n{}'.format(self.animated_moments))
        print('Collision Selection Scheme: '
              '{}'.format(self.collision_selection_scheme))
        print('Collision Steps per Time Step: {}'
              ''.format(self.collision_steps_per_time_step))
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
