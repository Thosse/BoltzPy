r"""
Calculation Submodule
=====================
The :py:class:`~boltzmann.calculation.Calculation`
submodule provides the following functionalities:

    * Generate the :py:class:`~boltzmann.calculation.Collisions`
      based on the velocity space grids
      (:py:class:`~boltzmann.configuration.SVGrid`)
    * Run the actual simulation
    * Set up functions to generate output and write it to the disk
      (:py:class:`~boltzmann.calculation.OutputFunction`)

"""

from .calculation import Calculation
from .output_function import OutputFunction
from .collisions import Collisions
