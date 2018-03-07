import boltzmann.configuration.configuration as b_cnf
import boltzmann.initialization.initialization as b_ini
import boltzmann.calculation.calculation as b_clc
# import boltzmann.configuration.animation as b_ani


class BoltzmannPDE:
    r"""Main Simulation Class, that
    inherits all functionalities from multiple Subclasses.

    Each Attribute or Method
    is inherited from one of the following Subclasses:

    :class:`~boltzmann.configuration.Configuration`
        * based solely on User Input
        * sets the Specimen-Parameters
        * configures and generates the Time and Position-Space grids
        * configures and generates Velocity-Space grids
          (Combined Velocity Grid for each Specimen)
        * generates Collisions (list and weights)

    :class:`~boltzmann.initialization.Initialization`
        * based on physical grids and User Input
        * initializes V-distribution in PSV Grid
        * sets flags for each P-Grid point (inner point/boundary point).
        * saves/loads conf-file to/from HDD
          (each init-operation adds an element to an instruction-vector)
          -> save/load-module

    :class:`~boltzmann.calculation.Calculation`
        * acts alternately on the PSV-Grid and a copy of it
        * contains several implementations
          for transport and collision steps
        * during calculations intermediate results are processed
          and written to HDD.

    *Animation*
        * reads the stored results on the HDD
        * generates animations of specified variables

    .. todo::
        - where to put moments for output(functions) (animation?),
          what data type for moments? array? list?
        - Ask Seb about properties
        - Ask Seb about exceptions
        - Ask Stefan about circular grids (collision generation, get_index...)
        - add method: b = boundaries
          b = [ G[0]*d, G[-1]*d]
          also for specimen (G[index[i]]*d[i], ...)
          IS this necessary?
        - read into numpys ufunc -> Speedup
        - time steps apply to calculation or animation?
        - add CUDA Support (PyTorch)
        - Create separate module to save/load conf-files
        - link configuration and initialization somehow.
          adding a new species, or changing P.dim,
          should delete the current Initialization
        - sphinx: use scipy theme instead of haiku
        - sphinx: bullet list in Attributes?
        - sphinx: how to make links in class attributes
          to other  classes and their attributes

    Attributes
    ----------
    config : :class:`~boltzmann.configuration.Configuration`
    init : :class:`~boltzmann.initialization.Initialization`
    """
    def __init__(self):
        self.cnf = b_cnf.Configuration()
        self.ini = b_ini.Initialization(self.cnf)
        # Todo Fix Initialization troubles (lots of check_integrity Errors)
        # self.clc = b_clc.Calculation(self.cnf, self.ini)

    def begin_initialization(self):
        self.ini = b_ini.Initialization(self.cnf)
        return

    # def begin_calculation(self, moments):
    #     self.clc = b_clc.Calculation(self.cnf, self.ini, moments)
