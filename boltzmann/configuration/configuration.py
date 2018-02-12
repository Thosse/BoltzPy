from . import grid as b_grd
from . import species as b_spc

import math
import numpy as np


def np_gcd(array_of_ints):
    gcd = array_of_ints[0]
    for new_number in array_of_ints[1:]:
        gcd = math.gcd(gcd, new_number)
    return gcd

