r"""
Configuration Submodule
=======================
The Configuration submodule provides the following functionalities:

* Specify the simulated Specimen
* Configure Time-, Positional-Space-, and Velocity-Space-Grids
* Generate the Collisions, based on the Velocity-Space-Grids (SV-Grids)


Configuration Class
-------------------

 * :py:class:`boltzmann.configuration.Configuration`
 
Subclasses
----------

 * :py:class:`boltzmann.configuration.Species`
 * :py:class:`boltzmann.configuration.Grid`
 * :py:class:`boltzmann.configuration.SVGrid`
 * :py:class:`boltzmann.configuration.Collisions`

"""

from .configuration import Configuration
from .species import Species
from .grid import Grid
from .svgrid import SVGrid
from .collisions import Collisions
