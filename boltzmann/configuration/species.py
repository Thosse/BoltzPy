
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

    # Todo move into constant
    @property
    def supported_colors(self):
        """:obj:`list` of :obj:`str`:
                List of all currently supported colors.
                Used in :class:`~boltzmann.animation.Animation`."""
        supported_colors = ['blue', 'red', 'green',
                            'yellow', 'brown', 'gray',
                            'olive', 'purple', 'cyan',
                            'orange', 'pink', 'lime']
        return supported_colors

    @property
    def default_parameters(self):
        """:obj:`dict`: Contains all default values for :meth:`add_specimen`.
        """
        free_color_list = [color for color in self.supported_colors
                           if color not in self._colors]
        assert len(free_color_list) is not 0, "All Colors are used, " \
                                              "add more."
        color = next((color for color in self.supported_colors
                      if color not in self._colors),
                     'black')
        if color is 'black':
            raise UserWarning('All available colors are used up.'
                              'Any new specimen will be colored black.')
        def_par = {'mass': 1,
                   'name': 'Specimen_' + str(self.n),
                   'alpha_list': [1] * (self.n + 1),
                   'color': color}
        return def_par

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
                     mass=None,
                     name=None,
                     alpha_list=None,
                     color=None):
        """Adds a new specimen.

        Parameters
        ----------
        mass : int, optional
        alpha_list : :obj:`list` of :obj:`float`
            Row and column of :attr:`alpha` collision probability matrix
        name : :obj:`str`, optional
        color : :obj:`str`, optional set the new element of :attr:`colors`

            """
        # Handle Keywords
        if mass is None:
            mass = self.default_parameters['mass']
        if name is None:
            name = self.default_parameters['name']
        if alpha_list is None:
            alpha_list = self.default_parameters['alpha_list']
        else:
            assert type(alpha_list) is list
            assert all([type(alpha) in [int, float]
                        and alpha >= 0
                        for alpha in alpha_list])
            assert len(alpha_list) is (self.n + 1)
        if color is None:
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
        alpha = np.zeros(shape=(self.n, self.n),
                         dtype=float)
        alpha[0:-1, 0:-1] = self.alpha
        self._alpha = alpha
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
