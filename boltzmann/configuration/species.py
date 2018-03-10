
import numpy as np


class Species:
    """A simple class encapsulating data about species to be simulated."""
    def __init__(self):
        self._n = 0
        self._mass = np.zeros(shape=(0,), dtype=int)
        self._alpha = np.zeros(shape=(0, 0), dtype=float)
        self._name = []
        self._color = []
        self.check_integrity()
        return

    @property
    def supported_colors(self):
        """:obj:`list` of :obj:`str`:
                List of all currently supported colors.
                Used in :class:`~boltzmann.animation.Animation`."""
        supported_colors = ['blue', 'red', 'green',
                            'yellow', 'black', 'brown',
                            'orange', 'pink', 'gray']
        return supported_colors

    @property
    def n(self):
        """:obj:int: Total number of specimen."""
        return self._n

    @property
    def mass(self):
        """:obj:`~np.ndarray` of :obj:`int`:
        Relative (not physical) mass of each specimen.
        Array of shape (:attr:`n`,).
        """
        return self._mass

    @property
    def alpha(self):
        """:obj:`~numpy.ndarray` of :obj:`float`:
        Describes Probability/Frequency of Collisions between 2 specimen.
        Array of shape (:attr:`n`,
        :attr:`n`).
        """
        return self._alpha

    @property
    def name(self):
        """:obj:`list` of :obj:`str`:
        List of Names of all Specimen
        (Used in :class:`~boltzmann.animation.Animation`).
        List of length :attr:`n`.
        """
        return self._name

    @property
    def color(self):
        """:obj:`list` of :obj:`str`:
        List of Colors of all Specimen
        (Used in :class:`~boltzmann.animation.Animation`).
        List of length :attr:`n`.
        """
        return self._color

    #####################################
    #           Configuration           #
    #####################################
    # Todo replace alpha_list, name and color by keyword argument
    # Todo clean up assert and default value block
    # Tod add documentation (esp. for keywords)
    def add_specimen(self,
                     mass=1,
                     alpha_list=None,
                     name=None,
                     color=None):
        # assign default values and/or sanity check
        # sanity check for mass
        assert type(mass) is int and mass > 0
        # Give default value to alpha list and sanity check
        if alpha_list is None:
            # Todo Check if this is a good default parameter
            alpha_list = [1] * (self.n+1)
        else:
            assert type(alpha_list) is list
            assert all([type(_) in [int, float] for _ in alpha_list])
            assert len(alpha_list) is (self.n+1)
        # Give default value to name and sanity check
        if name is None:
            name = 'Specimen_' + str(self.n)
        else:
            assert type(name) is str
        # Give default value to color and sanity check
        if color is None:
            _free_colors = [c for c in self.supported_colors
                            if c not in self.color]
            assert _free_colors is not [], "All Colors are used, add more."
            color = _free_colors[0]
        else:
            assert color in self.supported_colors, "Unsupported Color"
        # Actual Assignment of new values
        self._n += 1
        self._mass.resize(self.n)
        # Todo find read only properties of array
        # Todo (self.mass[-1] = mass also works)
        self._mass[-1] = mass
        # Add a row and a column to alpha
        _alpha = np.zeros(shape=(self.n, self.n),
                          dtype=float)
        _alpha[0:-1, 0:-1] = self.alpha
        self._alpha = _alpha
        # Todo find read only properties of array
        # Todo see mass
        self._alpha[-1, :] = np.array(alpha_list)
        self._alpha[:, -1] = np.array(alpha_list)
        self._name.append(name)
        self._color.append(color)
        self.check_integrity()

    def check_integrity(self):
        """Sanity Check"""
        assert type(self.n) is int
        assert self.n >= 0
        assert type(self.mass) is np.ndarray
        assert all(self.mass > 0)
        assert type(self.alpha) is np.ndarray
        assert self.alpha.shape == (self.n, self.n)
        assert type(self.name) is list
        assert len(self.name) is self.n
        assert all([type(name) is str for name in self.name])
        assert len(self.color) is self.n
        assert type(self.color) is list
        assert all([color in self.supported_colors
                    for color in self.color])

    def print(self):
        """Prints all Properties for Debugging Purposes"""
        print("Number of Specimen = {}".format(self.n))
        print("Names of Specimen  = {}".format(self.name))
        print("Masses of Specimen = {}".format(self.mass))
        print("Collision-Factor-Matrix = \n{}".format(self.alpha))
        print("Colors of Specimen = {}".format(self.color))
