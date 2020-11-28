import boltzpy as bp
from boltzpy.test_rule import RULES


###################################
#           Setup Cases           #
###################################
GEOMETRIES = dict()
GEOMETRIES["Geometry/2D_small"] = bp.Geometry(
    (10,),
    0.5,
    [RULES["2D_small/LeftConstant"],
     RULES["2D_small/Interior"],
     RULES["2D_small/RightBoundary"]])
GEOMETRIES["Geometry/equalMass"] = bp.Geometry(
    (10,),
    0.5,
    [RULES["equalMass/LeftBoundary"],
     RULES["equalMass/LeftInterior"],
     RULES["equalMass/RightInterior"],
     RULES["equalMass/RightBoundary"]])

