r"""
Configuration Submodule
=======================
The :py:class:`~boltzmann.configuration.Configuration`
submodule provides the following functionalities:

 * Specify the simulated
   :py:class:`~boltzmann.configuration.Species` /
   :py:class:`~boltzmann.configuration.Specimen`
 * Configure :py:class:`Time Grid <boltzmann.configuration.Grid>`
   and :py:class:`Positional-Space-Grid <boltzmann.configuration.Grid>`
 * Configure :py:class:`Specimen-Velocity-Grids
   <boltzmann.configuration.SVGrid>`
"""

from .configuration import Configuration
from .grid import Grid
from .species import Species
from .specimen import Specimen
from .svgrid import SVGrid
