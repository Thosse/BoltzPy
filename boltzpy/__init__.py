from boltzpy.BaseClass import BaseClass
from boltzpy.BaseModel import BaseModel
from boltzpy.grid import Grid
from boltzpy.CollisionModel import CollisionModel
from boltzpy.rules import BaseRule
from boltzpy.rules import InnerPointRule
from boltzpy.rules import ConstantPointRule
from boltzpy.rules import BoundaryPointRule
from boltzpy.rules import HomogeneousRule
from boltzpy.geometry import Geometry
from boltzpy.simulation import Simulation
import boltzpy.AnimatedFigure as Plot


# Define module constants
TEST_DIR = __file__[:-12] + "/test"
SIMULATION_DIR = __file__[:-21] + "/Simulations"
