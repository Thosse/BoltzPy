from . import species as b_spc
from . import grid as b_grd
from . import svgrid as b_svg
from . import collisions as b_col
import numpy as np


class Configuration:
    r"""Handles User Input and sets up the Simulation Parameters

    .. todo::
        - add proper file_name initialization/property
        - for Grid and SVGrid -> check print function for multi - attribute
        - link Species and SVGrid somehow
          -> adding Species, after setting up SVGrid
          should delete SVGrid or at least update it

          * Idea: each class has an is_set_up flag
          * after any change -> check flags of depending classes
          * main classes need to be linked for that!

        - Add Plotting-function to grids

    Attributes
    ----------
    s : :class:`Species`:
        Contains all data about the simulated Specimen.
    t : :class:`Grid`
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
    config_file_name : :obj:`str`
    file_path : :obj:`str`
    """

    def __init__(self):
        # Most Attributes are set up separately
        # Public Attributes
        self.s = b_spc.Species()
        self.t = b_grd.Grid()
        self.p = b_grd.Grid()
        self.sv = b_svg.SVGrid()
        self.cols = b_col.Collisions(self)
        # Read-Write Properties
        self._animated_moments = ['Mass']
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
    def supported_output(self):
        """:obj:`set` of :obj:`str`:
        Set of all currently supported moments."""
        supported_output = {'Mass',
                            'Mass_Flow',
                            'Momentum',
                            'Momentum_Flow',
                            'Energy',
                            'Energy_Flow'}
        return supported_output

    @property
    def supported_selection_schemes(self):
        """:obj:`set` of :obj:`str`:
        Set of all currently supported selection schemes for collisions."""
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

    def setup(self):
        """Prepares Configuration for Initialization """
        self.check_integrity(ignore_collisions=True)
        self.cols.setup()
        self.check_integrity()
        return

    # Todo implement this function
    # def get_config_file_address(self, name?):
    #     """Returns a files address
    #     of the config file or stored moments or stored animation"""
    #     files = [self.cnf.file_name + '_' + mom
    #              for mom in moments]
    #     return files

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self,
                        ignore_collisions=False):
        """Sanity Check"""
        self.s.check_integrity()
        self.t.check_integrity()
        self.p.check_integrity()
        self.sv.check_integrity()
        assert self.sv.dim >= self.p.dim
        # Todo Assert Collisions
        # if not ignore_collisions:
        # assert collisions
        return

    def print(self,
              physical_grids=False):
        """Prints all Properties for Debugging Purposes"""
        print('\n========CONFIGURATION========\n')
        print('Configuration Name: ')  # Todo + self.file_name)
        print('Animated Moments:\n{}'.format(self.animated_moments))
        print('Collision Selection Scheme: '
              '{}'.format(self.collision_selection_scheme))
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
