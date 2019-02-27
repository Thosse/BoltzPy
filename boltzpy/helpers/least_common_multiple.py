from math import gcd


def lcm(list_of_ints):
    result = list_of_ints[0]
    for i in list_of_ints[1:]:
        result = result * i // gcd(result, i)
    return result
