r"""
Configuration Submodule
=======================

This module incorporates the Calculation class and several subclasses.
The Calculation class contains all functionalities to set up the simulated specimen and create the Time-, Positional Space-, and Velocity Space-Grids.
It provides a framework for both the Grid and Species classes.

Classes
-------

 * :py:class:`boltzmann.configuration.Configuration`
 * :py:class:`boltzmann.configuration.Species`
 * :py:class:`boltzmann.configuration.Grid`

"""

from .configuration import Configuration
from .species import Species
from .grid import Grid

