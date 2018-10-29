r"""
BoltzPy
=======

The BoltzPy package is a solver for the boltzpy equation.

    * We uses discrete velocity models (DVM) to model rarefied gases.
    * The collision operator is deterministically approximated,
      see ADD SOURCES -> [Bernhoffpaper], [Brechtken, Sasse].
    * We support multiple specimen of varying masses.


Basic Structure of the Code:
----------------------------

 Any simulation is essentially an instance of 
 :py:class:`boltzpy.Simulation`.


Workflow
--------
 
 * Specify the simulated
   :py:class:`~boltzpy.Species` /
   :py:class:`~boltzpy.Specimen`
 * Configure :py:class:`Time Grid <boltzpy.Grid>`
   and :py:class:`Positional-Space-Grid <boltzpy.Grid>`
 * Configure :py:class:`Specimen-Velocity-Grids
   <boltzpy.SVGrid>`

Submodules:
-----------

 * :py:mod:`boltzpy.collisions`
 * :py:mod:`boltzpy.computation`
 * :py:mod:`boltzpy.animation`
"""

from boltzpy.simulation import Simulation
from boltzpy.specimen import Specimen
from boltzpy.species import Species
from boltzpy.svgrid import SVGrid
from boltzpy.grid import Grid
from boltzpy.rule import Rule
from boltzpy.data import Data

import boltzpy.collision_relations
import boltzpy.computation
import boltzpy.output
import boltzpy.animation
