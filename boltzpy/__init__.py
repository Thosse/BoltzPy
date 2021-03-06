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

 Any simulation is set up by creating
 a :py:class:`boltzpy.Simulation` instance.
 This instance can call its
 :py:meth:`~boltzpy.Simulation.run_computation` method
 to start the simulation and has methods to visualize the results as well.



Workflow
--------
 
 * Specify the simulated
   :py:class:`~boltzpy.Species` /
   :py:class:`~boltzpy.Specimen`
 * Configure :py:class:`Time Grid <boltzpy.Grid>`
   and :py:class:`Positional-Space-Grid <boltzpy.Grid>`
 * Configure :py:class:`Specimen-Velocity-Grids
   <boltzpy.SVGrid>`
"""
from boltzpy.BaseClass import BaseClass
from boltzpy.grid import Grid
from boltzpy.rule import Rule, InnerPointRule, ConstantPointRule, BoundaryPointRule
from boltzpy.scheme import Scheme
from boltzpy.geometry import Geometry
from boltzpy.simulation import Simulation
from boltzpy.species import Species
from boltzpy.specimen import Specimen
from boltzpy.svgrid import SVGrid
from boltzpy.collisions import Collisions
from boltzpy.data import Data
