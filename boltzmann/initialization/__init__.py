r"""
Initialization Submodule
========================
The :py:class:`~boltzmann.initialization.Initialization`
submodule provides a framework to:

* Set up a set of initialization
  :py:class:`~boltzmann.initialization.Rule`
  which specify the

  * initial states of inner points
  * Behaviour in the Simulation, e.g.

    * Normal Inner point
    * Input/output points with time conditions on v, rho,..
    * Boundary point with specified reflections

* Initialize the PSV-Grid on which the Calculation acts

"""

from .initialization import Initialization
from .rule import Rule
