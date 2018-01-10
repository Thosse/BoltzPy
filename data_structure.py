import Grid as bGrid


class BoltzmannPDE:
    """Main class, encapsulating User Interface.
    Initializes Simulation Problems and provides a framework to animate results.

    Attributes:
        number_frames (int): Total number of frames to be animated.
    """
    def __init__(self,
                 number_frames,
                 calculation_steps_per_frame,
                 collision_steps_per_calculation):
        assert type(number_frames) is int and number_frames > 0
        assert type(calculation_steps_per_frame) is int
        assert calculation_steps_per_frame >= 0
        assert type(collision_steps_per_calculation) is int
        assert collision_steps_per_calculation >= 0

        self.n_frames = number_frames
        self.calcsPerFrame = calculation_steps_per_frame
        self.collsPerCalc = collision_steps_per_calculation
        self.single_grid_for_all_specimen = None
        self.__s = bGrid.Species()
        self.__p = bGrid.Grid
        self.__t = bGrid.Grid
        self.__v = bGrid.Grid

    def add_specimen(self,
                     mass=1,
                     alpha_list=None,
                     name=None,
                     color=None):
        # Don't Change Values after grid-construction
        assert self.__v.isConstructed is False
        self.__s.add_specimen(mass, alpha_list, name, color)

    def initialize_position_grid(self,
                                 dimension,
                                 boundaries,
                                 d=None,
                                 n=None,
                                 shape='rectangular'):
        # Don't Change Values after grid-construction
        assert self.__p.isConstructed is False
        self.__p = bGrid.Grid(dimension,
                              boundaries,
                              d=d,
                              n=n,
                              shape=shape)

    def initialize_time_grid(self,
                             max_time,
                             dt=None,
                             number_time_steps=None):
        # Don't Change Values after grid-construction
        assert self.__t.isConstructed is False
        self.__t = bGrid.Grid(1,
                              [0.0, max_time],
                              d=dt,
                              n=number_time_steps,
                              shape='rectangular')

    def initialize_velocity_grid(self,
                                 dimension,
                                 max_v,
                                 dv=None,
                                 number_of_velocities=None,
                                 offset=None,
                                 shape='rectangular',
                                 single_grid_for_all_specimen=False):
        # Don't Change Values after grid-construction
        # TODO change this into sv / psv / calc class
        assert self.__v.isConstructed is False
        if offset is None:
            offset = [0.0]*dimension
        assert type(offset) is list
        assert type(max_v) in [float, int] and max_v > 0
        assert type(single_grid_for_all_specimen) is bool

        boundaries = [[-max_v + offset[i_d], max_v + offset[i_d]]
                      for i_d in range(dimension)]
        self.__v = bGrid.Grid(dimension,
                              boundaries,
                              d=dv,
                              n=number_of_velocities,
                              shape=shape)
        self.single_grid_for_all_specimen = single_grid_for_all_specimen

    def print(self):
        print("Time Grid:")
        if self.__t.isInitialized:
            # noinspection PyArgumentList
            self.__t.print()
        else:
            print("Not Initialized\n")
        print("Positional Grid:")
        if self.__p.isInitialized:
            # noinspection PyArgumentList
            self.__p.print()
        else:
            print("Not Initialized\n")
        print("Velocity Grid:")
        if self.__v.isInitialized:
            # noinspection PyArgumentList
            self.__v.print()
        else:
            print("Not Initialized\n")
        print("Specimen:")
        self.__s.print()

    # Combined PSV-Grid, for Calculations
    # sv_index = b_init.get_index_array_of_species(v_dim,
    #                                              s_n,
    #                                              v_n)
    # INITIALIZATION PARAMETERS
    # Todo: pack this into function
    # init_psv(p=[], s=[], rho, v0, temp)

    # #### COLLISION INVARIANTS ####
    # # Todo This is probably unnecessary
    # # Correlate to physical quantities: Mass, Momentum, Energy.
    # # They are invariant under application of the collision operator.
    # # For entry [i,j]:  i denotes the Density (as in DENSITY_SWITCH)
    # #                   j denotes the specimen
    #
    # # RHO is a multiplicative factor applied on whole density
    # # Correlates to MASS
    # RHO = np.ones((N_Dens, N_Sp), dtype=float)
    # # DRIFT sets the mean velocity
    # # Correlates to MOMENTUM
    # DRIFT = np.zeros((N_Dens, N_Sp, DIM_V), dtype=float)
    # # TEMP sets the variance of velocities
    # # Correlates to ENERGY
    # TEMP = np.ones((N_Dens, N_Sp), dtype=float)
    #
    # ''' DENSITY_SWITCH '''
    # # DENSITY_SWITCH controls how to create the initial Density u[t=0, ...]
    # # If DENSITY_SWITCH is a np.array with DENSITY_SWITCH[i] == k,
    # # Then u[t=0, x=i] will be initialized with of RHO[k],...
    # # If DENSITY_SWITCH is a string,
    # # then it denotes the address of a .npy-file
    # # which contains the initial density
    #
    # # Construct Initial Distribution based on Collision Invariants
    # DENSITY_SWITCH = np.zeros(tuple(N_X[0:3]), dtype=int)
    # # TODO: This should get some testing first
    # assert DIM_X is 1
    # DENSITY_SWITCH[N_X[-1] // 2:, :, :] = 1
    # DENSITY_SWITCH = DENSITY_SWITCH.reshape(N_X[-1])
    # # Read Initial Distribution from File
    # #   uncomment to generate file again
    # # DENSITY_SWITCH = cf_path + '_' + cf_name + "_initial_density.npy"

    # Animation Parameters '''

    animated_moments = None     # Which Moments are animated and in what order
    #     = [
    #     'Mass',
    #     'Mass Flow',
    #     'Momentum',
    #     'Momentum Flow',
    #     'Energy',
    #     'Energy Flow',
    #     #                  'COMPLETE'
    # ]

#############################################################################
#                            Maybe still necessary                          #
#############################################################################
    # # TODO all of this should be more elegant
    # # TODO Major step for complex mixtures!
    # if type(N_V) == int:
    #     # TODO: This only works for simple mixtures
    #     N_V = np.ones(N_Sp, dtype=int) * N_V
    # # Change Data types from lists to np.arrays, for simple indexing
    # # TODO do this more elegantly
    # N_V = np.array(N_V)
    #
    # #### Create Space Grid ####
    # X = b_init.get_space_array(DIM_X,
    #                            N_X,
    #                            MAX_X)
    #
    # #### Create Velocity Grid ####
    # # Make V-index array - marks beginning for velocities of each species
    #
    # # Create Velocity array
    # # contains all velocities of all species in a row
    # V = b_init.get_velocity_array(DIM_V,
    #                               N_Sp,
    #                               N_V,
    #                               MAX_V,
    #                               V_OFFSET,
    #                               spv_index)
    #
    # # Generate step sizes
    # # TODO for DIM_X != 1 this might lead to errors
    # assert DIM_X == 1
    # DX = X[1, 0] - X[0, 0]  # step size in X
    # # TODO change this into an assert, that checks for numerical stability
    # DT = 0.25 * DX / MAX_V  # step size in T

# test = BoltzmannPDE(10, 10, 10)
# test.initialize_position_grid(2, [[1, 2], [4, 5]], d=0.1)
# test.print()
