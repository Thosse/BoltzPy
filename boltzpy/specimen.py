import numpy as np
import boltzpy.constants as b_const


# Todo how to reference class attributes in numpy style?
# Todo break line in multi attribute docstring
class Specimen:
    """Contains the data of a single simulated specimen.

    This class is only a data structure for the
    :class:`~boltzpy.Species` class.
    Its only used to encapsulate data
    and assert its integrity, if necessary.


    Attributes
    ----------
    name : :obj:`str`
        Name of the Specimen, mainly used for the legend in the animation.
        Must be unique.
    mass : :obj:`int`
        Mass of the Specimen.
        Strongly influences the size of the Specimens
        :class:`Velocity Grid <boltzpy.SVGrid>`.
    collision_rate : :obj:`~numpy.array` [:obj:`float`]
        Determines the collision probability between two specimen.
        Is a row and a column of
        :attr:`Species.collision_rates <boltzpy.Species>`.
    color : :obj:`str`
        Color of the Specimen in the animation.
        Must be an element of :const:`~boltzpy.constants.SUPP_COLORS`.
    """
    def __init__(self,
                 name,
                 mass,
                 collision_rate,
                 color):
        self.check_parameters(name=name,
                              mass=mass,
                              collision_rate=collision_rate,
                              color=color)
        self.name = name
        self.color = color
        self.mass = mass
        if isinstance(collision_rate, list):
            collision_rate = np.array(collision_rate)
        self.collision_rate = collision_rate
        self.check_integrity()
        return

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self, complete_check=True):
        """Sanity Check.
        Asserts all conditions in :meth:`check_parameters`.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all attributes must be set (not None).
            If False, then unassigned attributes are ignored.
        """
        self.check_parameters(name=self.name,
                              color=self.color,
                              mass=self.mass,
                              collision_rate=self.collision_rate,
                              complete_check=complete_check)
        return

    @staticmethod
    def check_parameters(name=None,
                         mass=None,
                         collision_rate=None,
                         color=None,
                         complete_check=False):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        name : :obj:`str`, optional
        color : :obj:`str`, optional
        mass : :obj:`int`, optional
        collision_rate : :obj:`~numpy.array` [:obj:`float`], optional
            Determines the collision probability between two specimen.
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        """
        # For complete check, assert that all parameters are assigned
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all([param is not None for param in locals().values()])

        # check all parameters, if set
        if name is not None:
            assert isinstance(name, str)
            assert len(name) > 0

        if color is not None:
            assert isinstance(color, str)
            assert color in b_const.SUPP_COLORS

        if mass is not None:
            assert type(mass) == int
            assert mass > 0

        if collision_rate is not None:
            assert isinstance(collision_rate, np.ndarray)
            assert collision_rate.ndim == 1
            assert collision_rate.dtype == float
            assert np.all(collision_rate >= 0)
        return

    def __str__(self, write_collision_factors=True):
        """Converts the instance to a string, describing all attributes."""
        description = ''
        description += "Name = {}\n".format(self.name)
        description += "Color = {}\n".format(self.color)
        description += "Mass = {}".format(self.mass)
        if write_collision_factors:
            description += '\n'
            description += "Collision-Factors: \n\t{}".format(
                self.collision_rate)
        return description
