import boltzmann.configuration.configuration as b_cnf
import boltzmann.initialization.initialization as b_ini
# import boltzmann.configuration.calculation as b_clc
# import boltzmann.configuration.animation as b_ani

import numpy as np


class BoltzmannPDE(b_cnf.Configuration,
                   b_ini.Initialization):
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
    """
    def __init__(self):
        b_cnf.Configuration.__init__(self)
