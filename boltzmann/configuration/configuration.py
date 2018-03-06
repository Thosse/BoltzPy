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
        - add proper file_name initialization/property
        - add documentation: d is actually halved in v and sv
          due to integer representation (so far dv is multiples of 2)
        - Add Attributes:
          * Calculations_per_Frame
          * Collisions_per_Calculation
        - link Species and SVGrid somehow
          -> adding Species, after setting up SVGrid
          should delete SVGrid or at least update it
        - Add Plotting-function to grids
        - Where to specify integration order?

    Attributes
    ----------
    s : Species
        Contains all data about the simulated specimen.
    t : Grid
        Contains all data about simulation time and time step size.
    p : Grid
        Contains all data about Position-Space.
    sv : SVGrid
        Contains all data about the Velocity-Space of each Specimen.
        V-Spaces of distinct Specimen differ in step size
        and number of grid points.
        Maximum physical values may differ slightly between specimen.
    cols : Collisions
        Describes the collisions on the SV-Grid.
    file_name : str
    """

    def __init__(self):
        # Empty Initialization here
        # remaining Attributes are set up separately
        self.s = b_spc.Species()
        self.t = b_grd.Grid()
        self.t = b_grd.Grid()
        self.p = b_grd.Grid()
        self.sv = b_svg.SVGrid()
        self.cols = b_col.Collisions()
        self.file_name = 'default'
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
                       number_time_steps):
        step_size = max_time / (number_time_steps - 1)
        self.t.setup(1,
                     [number_time_steps],
                     step_size,
                     shape='rectangular')
        self.t.G = self.t.G.reshape((self.t.n[-1],))
        return

    def configure_position_space(self,
                                 dimension,
                                 list_number_of_points_per_dimension,
                                 step_size):
        self.p.setup(dimension,
                     list_number_of_points_per_dimension,
                     step_size,
                     'rectangular')
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

    def configure_collisions(self, selection_scheme):
        assert selection_scheme in b_col.Collisions.SELECTION_SCHEMES
        self.cols.scheme = selection_scheme
        self.cols.generate_collisions(self.s,
                                      self.sv)
        return

    def check_integrity(self):
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
        print('========CONFIGURATION========')
        print(self.file_name)
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
