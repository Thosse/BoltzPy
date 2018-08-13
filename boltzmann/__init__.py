r"""
BoltzPy
=======

The BoltzPy package is a solver for the boltzmann equation.

    * We uses discrete velocity models (DVM) to model rarefied gases.
    * The collision operator is deterministically approximated,
      see ADD SOURCES -> [Bernhoffpaper], [Brechtken, Sasse].
    * We support multiple specimen of varying masses.


Basic Structure of the Code:
----------------------------

 Any simulation is essentially an instance of 
 :py:class:`boltzmann.simulation.Simulation`.
 
 Almost all complex functionalities are delegated to one of the submodules.
 This class mainly acts as a mediator between these submodules.

Submodules:
-----------

 * :py:mod:`boltzmann.configuration`
 * :py:mod:`boltzmann.initialization`
 * :py:mod:`boltzmann.collisions`
 * :py:mod:`boltzmann.calculation`
 * :py:mod:`boltzmann.animation`
"""

from .simulation import Simulation

import boltzmann.configuration
import boltzmann.initialization
import boltzmann.collisions
import boltzmann.calculation
import boltzmann.animation
