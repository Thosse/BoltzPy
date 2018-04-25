import numpy as np
from matplotlib.colors import cnames as mpl_colors


class Specimen:
    """Encapsulates all data of a single simulated specimen."""
    def __init__(self,
                 name,
                 color,
                 mass,
                 collision_rate
                 ):
        self.name = name
        self.color = color
        self.mass = mass
        self.collision_rate = collision_rate
        self.check_integrity(self)
        return

    #####################################
    #            Verification           #
    #####################################
    @staticmethod
    def check_integrity(check_instance=None,
                        name=None,
                        color=None,
                        mass=None,
                        collision_rate=None):
        """Checks Integrity of Attributes.
        If parameter check_instance is set, the whole instance is checked.
        Otherwise only the specified parameters are checked.

        Parameters
        ----------
        check_instance : :obj:`Specimen`, optional
        name : :obj:`str`, optional
        color : :obj:`str`, optional
        mass : :obj:`int`, optional
        collision_rate : :obj:`np.ndarray` of :obj:`float`, optional
            Determines the collision probability between two specimen.
            Should be a row or column of :attr:`collision_rate_matrix`.
        """
        if check_instance is not None:
            assert type(check_instance) is Specimen
            self = check_instance
            self.check_integrity(name=self.name,
                                 color=self.color,
                                 mass=self.mass,
                                 collision_rate=self.collision_rate)
            return
        if name is not None:
            assert type(name) is str
        if color is not None:
            assert type(color) is str
            assert color in mpl_colors
        if mass is not None:
            assert type(mass) == int
            assert mass > 0
        if collision_rate is not None:
            assert type(collision_rate) is np.ndarray
            assert len(collision_rate.shape) == 1
            assert collision_rate.dtype == float
            # noinspection PyTypeChecker
            assert np.all(collision_rate >= 0)
        return

    def print(self, print_collision_rate=False):
        """Prints all Properties for Debugging."""
        print("Name of Specimen  = {}".format(self.name))
        print("Color of Specimen = {}".format(self.color))
        print("Mass of Specimen = {}".format(self.mass))
        if print_collision_rate:
            print("Collision-Factors = \n{}".format(self.collision_rate))
        return
