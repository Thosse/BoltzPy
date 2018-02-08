import boltzmann.configuration.configuration as b_cnf
# import boltzmann.configuration.initialization as b_ini
# import boltzmann.configuration.calculation as b_clc
# import boltzmann.configuration.animation as b_ani


class BoltzmannPDE:
    r"""
    Main class, is divided into four autonomous parts:

    *Configuration*
        * based solely on User Input
        * sets the Specimen-Parameters
        * generates the physical grids of T-Space, P-Space, and SV-Space.

    *Initialization*
        * based on physical grids and User Input
        * generates PSV-Grid
        * initializes V-distribution and sets flags for each P-Grid point
          (inner point/boundary point).
        * Generates collision-list and collision-weights
        * saves/loads conf-file to/from HDD
          (each init-operation adds an element to an instruction-vector)
          -> save/load-module

    *Calculation*
        * acts alternately on the PSV-Grid and a copy of it
        * contains several implementations for transport and collision steps
        * during calculations intermediate results are processed and written to HDD.

    *Animation*
        * reads the stored results on the HDD
        * generates animations of specified variables

    .. todo::
        - Remove classes from submodules -> replace by functions
        - Where to specify integration order?
        - Create separate module to save/load conf-files

    Attributes
    ----------
    __use_gpu : bool
        Decides computation device (GPU/CPU)
    __use_single_v_grid : bool
        Use equal velocity grids for all specimen (yes/no)
    config : :obj:Configuration
        Handles basic setup of:
            * Specimen parameters,
            * Time-Grid,
            * Position-Space-Grid,
            * Velocity-Space-Grid.

    """
    def __init__(self,
                 use_gpu=False,
                 use_single_v_grid=False):
        self.__use_gpu = use_gpu
        self.__use_single_v_grid = use_single_v_grid
        # Todo set fType depending on use_gpu, and submit to initialize
        if use_gpu:
            f_type = float
        else:
            f_type = float
        self.config = b_cnf.Configuration(f_type=f_type)

    def print(self):
        self.config.print()
