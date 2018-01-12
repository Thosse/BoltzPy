import math
import numpy as np


def np_gcd(array_of_ints):
    assert array_of_ints.dtype == np.int32
    gcd = array_of_ints[0]
    for new_number in array_of_ints[1:]:
        gcd = math.gcd(ret, new_number)
    return gcd


def get_close_int(real_number, precision=1e-6):
    close_int = int(math.floor(real_number))
    if math.fabs(close_int - real_number) < precision:
        return close_int
    elif math.fabs(close_int+1 - real_number) < precision:
        return close_int + 1
    else:
        assert False, "Number {} is not close to an integer".format(real_number)
