# import boltzmann.configuration.configuration as b_cnf
import boltzmann.configuration.species as b_spc
import boltzmann.configuration.grid as b_grd
import boltzmann.configuration.svgrid as b_svg
import boltzmann.configuration.collisions as b_col
# import boltzmann.configuration.initialization as b_ini
# import boltzmann.configuration.calculation as b_clc
# import boltzmann.configuration.animation as b_ani

import numpy as np


class BoltzmannPDE:
    r"""
    Main class, its functionalities can be divided
    into four parts:

    *Configuration*
        * based solely on User Input
        * sets the Specimen-Parameters
        * generates the Time and Position-Space grids
        * generates Velocity-Space grids
          (Combined Velocity Grid for each Specimen)
        * generates Collisions (list and weights)

    *Initialization*
        * based on physical grids and User Input
        * initializes V-distribution in PSV Grid
        * sets flags for each P-Grid point (inner point/boundary point).
        * saves/loads conf-file to/from HDD
          (each init-operation adds an element to an instruction-vector)
          -> save/load-module

    *Calculation*
        * acts alternately on the PSV-Grid and a copy of it
        * contains several implementations
          for transport and collision steps
        * during calculations intermediate results are processed
          and written to HDD.

    *Animation*
        * reads the stored results on the HDD
        * generates animations of specified variables

    .. todo::
        - implement slimmer __init__
        - Remove Configuration from submodules -> replace by specific classes
        - Possibly move other submodules into classes as well
        - Add Offset parameter for setup velocity method
        - Add Plot-function of grids
        - time steps apply to calculation or animation?
        - add CUDA Support (PyTorch)
        - Add File with general constants (e.g. data Types)
        - Where to specify integration order?
        - Create separate module to save/load conf-files

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
        and number of grid points. Boundaries can alsp differ slightly.
    cols : Collisions
        Describes the collisions on the SV-Grid.
    psv : PSVGrid
    fType : np.dtype
        Determines data type of floats.
    iType : np.dtype
        Determines data type of integers.
    flag_gpu : bool
        Decides computation device (GPU/CPU)
    flag_single_v_grid : bool
        Use equal velocity grids for all specimen (yes/no)
    """
    def __init__(self,
                 use_gpu=False,
                 use_single_v_grid=False):
        self.flag_gpu = use_gpu
        self.flag_single_v_grid = use_single_v_grid
        # Todo choose type accordingly
        self.fType = np.float32
        self.iType = np.int32
        # Empty Initialization here
        # custom classes are set up separately
        self.s = b_spc.Species()
        self.t = b_grd.Grid(0,
                            [],
                            0.0,
                            shape='not_initialized',
                            check_integrity=False,
                            create_grid=False)
        self.p = b_grd.Grid(0,
                            [],
                            0.0,
                            shape='not_initialized',
                            check_integrity=False,
                            create_grid=False)
        # Todo dummy_ is only a temporary fix
        dummy_species = b_spc.Species()
        dummy_species.add_specimen()
        self.sv = b_svg.SVGrid(dummy_species,
                               self.p,
                               check_integrity=False,
                               create_grid=False)
        self.cols = b_col.Collisions(self.fType,
                                     self.iType)
        # self.psv = b_psv.PSVGrid()
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

    def setup_time(self,
                   max_time,
                   number_time_steps):
        step_size = max_time / (number_time_steps - 1)
        self.t = b_grd.Grid(1,
                            [number_time_steps],
                            step_size,
                            shape='rectangular',
                            float_data_type=self.fType,
                            integer_data_type=self.iType)
        self.t.G = self.t.G.reshape((self.t.n[-1],))
        return

    def setup_position_space(self,
                             dimension,
                             list_number_of_points_per_dimension,
                             step_size):
        self.p = b_grd.Grid(dimension,
                            list_number_of_points_per_dimension,
                            step_size,
                            'rectangular',
                            self.fType,
                            self.iType)
        return

    # Todo make parameters more intuitive, compared to other grids
    def setup_velocity_space(self,
                             dimension,
                             grid_points_x_axis,
                             max_v):
        step_size = 2*max_v/(grid_points_x_axis-1)
        number_of_points_per_dimension = [grid_points_x_axis] * dimension
        # Offset is chosen s.t. the grid is centered around zero
        offset = [-max_v] * dimension
        _v = b_grd.Grid(dimension,
                        number_of_points_per_dimension,
                        step_size,
                        'rectangular',
                        self.fType,
                        self.iType,
                        offset=offset)
        self.sv = b_svg.SVGrid(self.s,
                               _v)
        return

    def setup_collisions(self, selection_scheme):
        self.cols.generate_collisions(self.s,
                                      self.sv,
                                      selection_scheme)
        return

    def check_configuration(self):
        self.s.check_integrity()
        assert self.s.mass.dtype == self.iType
        assert self.s.alpha.dtype == self.fType
        self.t.check_integrity()
        assert self.t.iType == self.iType
        assert self.t.fType == self.fType
        self.p.check_integrity()
        assert self.p.iType == self.iType
        assert self.p.fType == self.fType
        self.sv.check_integrity()
        assert self.sv.dim >= self.p.dim
        assert self.sv.iType == self.iType
        assert self.sv.fType == self.fType
        return

    def print(self,
              physical_grids=False):
        print('========CONFIGURATION========')
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
        print('Flags and Types:')
        print('----------------')
        print("Float Data Type    = {}".format(self.fType))
        print("Integer Data Type  = {}".format(self.iType))
        print("flag_gpu  = {}".format(self.flag_gpu))
        print("flag_single_v_grid  = {}".format(self.flag_single_v_grid))

    #####################################
    #           Initialization          #
    #####################################
