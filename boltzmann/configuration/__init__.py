r"""
Configuration Submodule
=======================
The Configuration submodule provides functions to specify the general
parameters of the simulation:

* simulated Specimen
* Time-, Positional-Space-, and Velocity-Space-Grids.

It relies heavily on both the Grid and Species classes.

Classes
-------

 * :py:class:`boltzmann.configuration.Species`
 * :py:class:`boltzmann.configuration.Grid`

"""


from .species import Species
from .grid import Grid
