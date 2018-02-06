r"""
BoltzPy
======================

The BoltzPy package is a solver for the boltzmann equation
to simulate rarefied gases with multiple specimen of
varying masses.
It's based on new results in Kinetic gas theory and on discrete velocity models.
ADD SOURCES -> [Bernhoffpaper], [Brechtken, Sasse].

Motivation:

Currently known and used algorithms assume all gases to be mono atomic.
We seek to develop, implement, test and compare our numerical approaches
for the simulation of the boltzmann equation for gas mixtures.

Features:

 * Basic Configuration

Planned:
 * Implementation of 1D geometries
 * Basic Initialization
 * Basic Calculation
 * Animation of several characteristic variables
 * Generate all possible collisions
 * Automatic reduction of collisions (BRECHTKEN/SASSE)
 * Plan and Implement (small-sized) config file
 * Implement 2D and 3D geometries
 * add more complex geometries to grid (setting and generation)
 * Simple GUI for configuration (DARIUS)

Classes:

The Main class, which provides a simple framework to write test cases as a
small python-script:

 * :py:class:`boltzmann.BoltzmannPDE`

A framework for testing all implemented methods is provided by:

 * :py:class:`boltzmann.BoltzmannTest`

Submodules:

The, mostly autonomic, submodules are:

 * :py:mod:`boltzmann.configuration`
 * :py:mod:`boltzmann.initialization`
 * :py:mod:`boltzmann.calculation`
 * :py:mod:`boltzmann.animation`
"""

from .BoltzmannPDE import BoltzmannPDE
from .BoltzmannTest import BoltzmannTest
import boltzmann.configuration
import boltzmann.initialization
import boltzmann.calculation
import boltzmann.animation

