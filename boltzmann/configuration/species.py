
import numpy as np


class Species:
    """A simple class encapsulating data about species to be simulated."""
    def __init__(self):
        self._n = 0
        self._mass = np.zeros(shape=(self._n,), dtype=int)
        self._alpha = np.zeros(shape=(self._n, self._n), dtype=float)
        self._names = []
        self._colors = []
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
    def names(self):
        """:obj:`list` of :obj:`str`:
        List of Names of all Specimen
        (Used in :class:`~boltzmann.animation.Animation`).
        List of length :attr:`n`.
        """
        return self._names

    @property
    def colors(self):
        """:obj:`list` of :obj:`str`:
        List of Colors of all Specimen
        (Used in :class:`~boltzmann.animation.Animation`).
        List of length :attr:`n`.
        """
        return self._colors

    #####################################
    #           Configuration           #
    #####################################
    def add_specimen(self,
                     mass=1,
                     **kwargs):
        """Adds a new specimen.

        Parameters
        ----------
        mass : int
        **kwargs : The keyword arguments are used to manually set attributes
            **alpha_list** set the new row and column of :attr:`alpha`

            **name** set the new element of :attr:`names`

            **color** set the new element of :attr:`colors`

            """
        # Handle Keywords
        assert type(mass) is int and mass > 0
        if 'name' in kwargs:
            assert type(kwargs['name']) is str
            name = kwargs['name']
        else:
            name = 'Specimen_' + str(self.n)
        if 'alpha_list' in kwargs:
            assert type(kwargs['alpha_list']) is list
            assert all([type(alpha) in [int, float]
                        and alpha >= 0
                        for alpha in kwargs['alpha_list']])
            assert len(kwargs['alpha_list']) is (self.n + 1)
            alpha_list = kwargs['alpha_list']
        else:
            alpha_list = [1] * (self.n+1)

        if 'color' in kwargs:
            assert kwargs['color'] in self.supported_colors
            color = kwargs['color']
        else:
            free_color_list = [c for c in self.supported_colors
                               if c not in self.colors]
            assert len(free_color_list) is not 0, "All Colors are used, " \
                                                  "add more."
            color = free_color_list[0]

        # Assign new values
        self._n += 1
        self._mass.resize(self.n)
        self._mass[-1] = mass
        # Add a row and a column to alpha
        _alpha = np.zeros(shape=(self.n, self.n),
                          dtype=float)
        _alpha[0:-1, 0:-1] = self.alpha
        self._alpha = _alpha
        self._alpha[-1, :] = np.array(alpha_list)
        self._alpha[:, -1] = np.array(alpha_list)
        self._names.append(name)
        self._colors.append(color)
        self.check_integrity()
        return

    def check_integrity(self):
        """Sanity Check. Checks Integrity of all Attributes"""
        assert type(self.n) is int
        assert self.n >= 0
        assert type(self.mass) is np.ndarray
        assert self.mass.shape == (self.n,)
        assert self.mass.dtype == int
        assert all(self.mass > 0)
        assert type(self.alpha) is np.ndarray
        assert self.alpha.shape == (self.n, self.n)
        assert self.alpha.dtype == float
        assert np.all(self.alpha >= 0)
        assert type(self.names) is list
        assert len(self.names) is self.n
        assert all([type(name) is str for name in self.names])
        assert len(self.colors) is self.n
        assert type(self.colors) is list
        assert all([color in self.supported_colors
                    for color in self.colors])

    def print(self):
        """Prints all Properties for Debugging Purposes"""
        print("Number of Specimen = {}".format(self.n))
        print("Names of Specimen  = {}".format(self.names))
        print("Masses of Specimen = {}".format(self.mass))
        print("Collision-Factor-Matrix = \n{}".format(self.alpha))
        print("Colors of Specimen = {}".format(self.colors))
        return
