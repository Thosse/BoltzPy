r"""
Initialization Submodule
========================
The Initialization class provides a framework to set up the

* initial states of inner points.
* boundary points and conditions
  (Input/output points with time conditions on v, rho,..;
  boundary point with specified reflections)

of the PSV-Grid.

Classes
-------

 * :py:class:`boltzmann.initialization.Initialization`
 * :py:class:`boltzmann.initialization.Instruction`
 * :py:class:`boltzmann.initialization.InnerPointRule`

"""

from .initialization import Initialization
from .initialization import Instruction
from .initialization import InnerPointRule
