from .import grid as b_grd
from . import species as b_spc

import numpy as np


class Configuration:
    """Framework for Input of Simulation Parameters.
    Generates Physical Grids based on Parameters.

    Attributes:
        __s (:obj:Species):
            Contains all Parameters for Specimen Grid.
        __t (:obj:Grid):
            Contains all Parameters for Time Grid.
        __p (:obj:Grid):
            Contains all Parameters for Positional Space Grid
        __v (:obj:Grid):
            Contains all Parameters for Velocity Space.
        __fType (Type):
            Decides Data Type for Grids
    """
    def __init__(self, f_type=float):
        self.__s = b_spc.Species()
        self.__p = b_grd.Grid(f_type)
        self.__t = b_grd.Grid(f_type)
        self.__v = b_grd.Grid(f_type)
        self.__fType = f_type

    def set_t(self,
              max_time=None,
              step_size=None,
              number_time_steps=None):
        if max_time is None:
            boundaries = None
        else:
            boundaries = [0.0, max_time]
        self.__t.initialize(dim=1,
                            b=boundaries,
                            d=step_size,
                            n=number_time_steps,
                            shape='rectangular')

    def set_p(self,
              dimension,
              boundaries=None,
              step_size=None,
              total_grid_points=None,
              shape='rectangular'):
        self.__p.initialize(dim=dimension,
                            b=boundaries,
                            d=step_size,
                            n=total_grid_points,
                            shape=shape)

    def set_v(self,
              dimension,
              max_v=None,
              step_size=None,
              total_grid_points=None,
              offset=None,
              shape='rectangular'):
        if offset is None:
            offset = [0.0]*dimension
        assert type(offset) is list
        assert len(offset) == dimension
        assert type(max_v) in [float, int] and max_v > 0

        if max_v is None:
            b = None
        else:
            b = [-max_v, max_v]
        self.__v.initialize(dimension,
                            b,
                            d=step_size,
                            n=total_grid_points,
                            shape=shape)
        # Todo this is currently wrong
        offset = np.array(offset)
        self.__v.b += offset

    def add_specimen(self,
                     mass=1,
                     alpha_list=None,
                     name=None,
                     color=None):
        self.__s.add_specimen(mass, alpha_list, name, color)

    def print(self):
        print("Specimen:")
        self.__s.print()
        print("Time Grid:")
        self.__t.print()
        print("Positional Grid:")
        self.__p.print()
        print("Velocity Grid:")
        self.__v.print()
