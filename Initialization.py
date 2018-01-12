import Grid as bG
import Species as bSp


class Initialization:
    """Framework for Input of Simulation Parameters

    Attributes:
        __s (:obj:Species):
            Contains all Specimen Parameters
        __t (:obj:Grid):
            Contains all Time Parameters
        __p (:obj:Grid):
            Contains all Positional Space Parameters
        __v (:obj:Grid):
            Contains all Velocity Space Parameters
        __fType (Type):
            Decides Data Type for Grids
    """
    def __init__(self, f_type=float):
        self.__s = bSp.Species()
        self.__p = bG.Grid(f_type)
        self.__t = bG.Grid(f_type)
        self.__v = bG.Grid(f_type)
        self.__fType = f_type

    def time(self,
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

    def position_space(self,
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

    def velocity_space(self,
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

        # assert max_v is not None, "This is not specified so far"
        # b = [[-max_v + offset[i_d], max_v + offset[i_d]]
        #      for i_d in range(dimension)]
        if max_v is None:
            b = None
        else:
            b = [-max_v, max_v]
        self.__v.initialize(dimension,
                            b,
                            d=step_size,
                            n=total_grid_points,
                            shape=shape)
        offset = np.array(offset)
        self.b += offset

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
