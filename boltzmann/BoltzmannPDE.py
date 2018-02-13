# import boltzmann.configuration.configuration as b_cnf
import boltzmann.configuration.species as b_spc
import boltzmann.configuration.grid as b_grd
import boltzmann.configuration.svgrid as b_svg
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
        * generates the physical grids of T-Space, P-Space
        * generates physical grid of SV-Space
          (Combined Velocity Grid for each Specimen)
        * Generates collision-list and collision-weights

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
        - Remove Configuration from submodules -> replace by functions
        - Possibly move other submodules into functions as well
        - Add Offset parameter for setup velocity method
        - Add Plot-function of grids
        - time steps apply to calculation or animation?
        - Where to specify integration order?
        - Create separate module to save/load conf-files

    Attributes
    ----------
    s : Species
        Contains all data about the simulated specimen.
    t : Grid
        Contains all data about simulation time and time step size.
    tG : np.ndarray
        Physical Time-Grid.
        Array of shape=(t.n[-1],) and dtype=fType.
    p : Grid
        Contains all data about Position-Space.
    pG : np.ndarray
        Physical Grid of Position-Space.
        Array of shape=(p.n[-1], p.dim) and dtype=fType.
    sv : SVGrid
        Contains all data about the Velocity-Space of each Specimen.
        V-Spaces of different Specimen differ only in the step size
        and number of grid points.
    svG : np.ndarray
        Concatenated Physical Grids of the Velocity-Spaces of all Specimen.
        Array of dtype=fType.
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
        # Empty Initialization here, is filled seperately
        self.s = b_spc.Species()
        self.t = b_grd.Grid(0,
                            [],
                            0.0,
                            shape='not_initialized',
                            check_integrity=False)
        self.tG = np.zeros((0,), dtype=self.fType)
        self.p = b_grd.Grid(0,
                            [],
                            0.0,
                            shape='not_initialized',
                            check_integrity=False)
        self.pG = np.zeros((0,), dtype=self.fType)
        # Todo dummy_ is only a temporary fix
        dummy_species = b_spc.Species()
        dummy_species.add_specimen()
        self.sv = b_svg.SVGrid(dummy_species,
                               self.p,
                               check_integrity=False)
        self.svG = np.zeros((0,), dtype=self.fType)
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
        self.tG = self.t.make_grid().reshape((self.t.n[-1],))
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
        self.pG = self.p.make_grid()
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
        self.svG = self.sv.make_grid()
        return

    def run_configuration(self):
        print('TODO')

    def check_configuration(self):
        self.s.check_integrity()
        assert self.s.mass.dtype == self.iType
        assert self.s.alpha.dtype == self.fType
        self.t.check_integrity()
        assert self.t.iType == self.iType
        assert self.t.fType == self.fType
        assert self.tG.shape is (self.t.n,)
        self.p.check_integrity()
        assert self.p.iType == self.iType
        assert self.p.fType == self.fType
        assert self.pG.shape is (self.p.n, self.p.dim)
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
        self.t.print()
        if physical_grids:
            print('Physical Time Grid:')
            print(self.tG)
        print('')
        print('Position-Space Data:')
        print('--------------------')
        self.p.print()
        if physical_grids:
            print('Physical P-Space Grid:')
            print(self.pG)
        print('')
        print('Velocity-Space Data:')
        print('--------------------')
        self.sv.print()
        if physical_grids:
            for _s in range(self.s.n):
                print('Physical V-Space Grid of Specimen_{}:'.format(_s))
                beg = self.sv.index[_s]
                end = self.sv.index[_s + 1]
                print(self.svG[beg:end])
        print("Float Data Type    = {}".format(self.fType))
        print("Integer Data Type  = {}".format(self.iType))
        print("flag_gpu  = {}".format(self.flag_gpu))
        print("flag_single_v_grid  = {}".format(self.flag_single_v_grid))

    #####################################
    #           Initialization          #
    #####################################
