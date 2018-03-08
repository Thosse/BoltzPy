from . import species as b_spc
from . import grid as b_grd
from . import svgrid as b_svg
from . import collisions as b_col

import numpy as np


class Configuration:
    r"""Handles Setup of General Simulation Parameters

    * based solely on User Input
    * specifies the Specimen-Parameters
    * generates the Time and Position-Space grids
    * generates Velocity-Space grids
      (Combined Velocity Grid for each Specimen)
    * generates Collisions (list and weights)

    .. todo::
        - add moments attribute and attributes for integration parameters
          (dictionary?)
        - add proper file_name initialization/property
        - for Grid and SVGrid -> check print function for multi - attribute
        - Add Attributes:
          * Calculations_per_Frame
          * Collisions_per_Calculation
        - link Species and SVGrid somehow
          -> adding Species, after setting up SVGrid
          should delete SVGrid or at least update it

          * Idea: each class has an is_set_up flag
          * after any change -> check flags of depending classes
          * main classes need to be linked for that!

        - Add Plotting-function to grids
        - Where to specify integration order?

    Attributes
    ----------
    s : :class:`~boltzmann.configuration.Species`
        Contains all data about the simulated specimen.
    t : :class:`~boltzmann.configuration.Grid`
        Contains all data about simulation time and time step size.
        :attr:`~boltzmann.configuration.Configuration.t.G`
        denotes the times at which the results are written out to HDD.
    p : :class:`~boltzmann.configuration.Grid`
        Contains all data about Position-Space.
    sv : :class:`~boltzmann.configuration.SVGrid`
        Contains all data about the Velocity-Space of each Specimen.
        V-Spaces of distinct Specimen differ in step size
        and number of grid points.
        Maximum physical values may differ slightly between specimen.
    cols : :class:`~boltzmann.configuration.Collisions`
        Describes the collisions on the SV-Grid.
    file_name : :obj:`str`
    file_path : :obj:`str`
    """

    def __init__(self):
        # Empty Initialization here
        # Most Attributes are set up separately
        self._animated_moments = ['Mass']
        self._collision_selection_scheme = 'Complete'
        self.s = b_spc.Species()
        self.t = b_grd.Grid()
        self.p = b_grd.Grid()
        self.sv = b_svg.SVGrid()
        self.cols = b_col.Collisions(self)
        # Todo Add these Attributes properly
        # self.file_name = 'default'
        # self.file_path = ''

        return

    @property
    def supported_output(self):
        """Set of all currently supported moments."""
        supported_output = {'Mass',
                            'Mass_Flow',
                            'Momentum',
                            'Momentum_Flow',
                            'Energy',
                            'Energy_Flow'}
        return supported_output

    @property
    def supported_selection_schemes(self):
        """Set of all currently supported collision selection schemes."""
        supported_selection_schemes = {'Complete'}
        return supported_selection_schemes

    # Todo use np.array -> make use of shape property?
    @property
    def animated_moments(self):
        """:obj:`list` of :obj:`str`:
        List of the Moments to be stored and animated"""
        return self._animated_moments

    @animated_moments.setter
    def animated_moments(self, list_of_moments):
        if any([mom not in self.supported_output
                for mom in list_of_moments]):
            # Todo throw exception
            assert False
        self._animated_moments = list_of_moments
        return

    @property
    def collision_selection_scheme(self):
        """:obj:`str`:
        Selection Scheme for Collisions"""
        return self._collision_selection_scheme

    @collision_selection_scheme.setter
    def collision_selection_scheme(self, scheme):
        if scheme not in self.supported_selection_schemes:
            # Todo throw exception
            assert False
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
        self.s.add_specimen(mass,
                            alpha_list,
                            name,
                            color)

    def configure_time(self,
                       max_time,
                       number_time_steps,
                       calculations_per_time_step=1):
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
        self.p.setup(dimension,
                     list_number_of_points_per_dimension,
                     step_size, )
        return

    def configure_velocity_space(self,
                                 dimension,
                                 grid_points_x_axis,
                                 max_v,
                                 shape='rectangular',
                                 offset=None):
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

    def setup_collisions(self):
        self.cols.scheme = self.collision_selection_scheme
        self.cols.generate_collisions()
        return

    #####################################
    #            Verification           #
    #####################################

    def check_integrity(self):
        """Sanity Check"""
        self.s.check_integrity()
        assert self.s.mass.dtype == np.float64
        assert self.s.alpha.dtype == np.float64
        self.t.check_integrity()
        assert type(self.t.dim) is int
        assert type(self.t.d) is float
        self.p.check_integrity()
        assert type(self.p.dim) is int
        assert type(self.p.d) is float
        self.sv.check_integrity()
        assert self.sv.dim >= self.p.dim
        assert type(self.sv.dim) is int
        assert type(self.sv.d) is float
        # Todo Assert Collisions
        return

    def print(self,
              physical_grids=False):
        """Prints all Properties for Debugging Purposes"""
        print('========CONFIGURATION========')
        print('Configuration Name: ')  # Todo + self.file_name)
        print('Specimen:')
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
