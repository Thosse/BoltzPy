r"""
Configuration Submodule
=======================
The :py:class:`~boltzmann.configuration.Configuration`
submodule provides the following functionalities:

 * Specify the simulated
   :py:class:`~boltzmann.configuration.Species`
 * Configure Time-:py:class:`~boltzmann.configuration.Grid`
   and Positional-Space-:py:class:`~boltzmann.configuration.Grid`
 * Configure Specimen-Velocity-Grids
   (:py:class:`~boltzmann.configuration.SVGrid`)

"""

from .configuration import Configuration
from .grid import Grid
from .species import Species
from .svgrid import SVGrid
