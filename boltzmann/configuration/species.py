
from . import specimen as b_spm

import numpy as np


class Species:
    """Handles all :class:`Specimen` objects.
    Specimen can be accessed by either their index oder name."""
    def __init__(self):
        self._specimen_array = np.ndarray(0, dtype=b_spm.Specimen)
        self.collision_rate_matrix = np.zeros(shape=(0, 0), dtype=float)
        self.check_integrity()
        return

    def __getitem__(self, item):
        """:obj:`Species` instances can be indexed by both position (int)
        and name of :obj:`Specimen` (str)."""
        if type(item) is int:
            return self._specimen_array[item]
        elif type(item) is str:
            for s in self._specimen_array:
                if s.name == item:
                    return s
            return None
        else:
            message = "item must be either the index(int) or name(str)"
            raise TypeError(message)

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
        """:obj:`dict`: Default parameters for :meth:`add_specimen`.
        """
        used_colors = [s.color for s in self._specimen_array]
        color = next((color for color in self.supported_colors
                      if color not in used_colors),
                     'black')
        # TODO Raise Warning in this case?
        # if color is 'black':
        default_params = {
                          'name': 'Specimen_' + str(self.n),
                          'color': color,
                          'mass': 1,
                          'collision_rate': np.ones(self.n + 1)
                         }
        return default_params

    @property
    def n(self):
        """:obj:`int` :
        Total number of :class:`Specimen`.
        """
        return self._specimen_array.size

    @property
    def mass(self):
        """:obj:`np.ndarray` of :obj:`int` :
            Array of masses of all :class:`Specimen`.
            """
        mass = [s.mass for s in self._specimen_array]
        return np.array(mass)

    @property
    def colors(self):
        """:obj:`list` of :obj:`int` :
        List of colors of all :class:`Specimen`.
        """
        colors = [s.color for s in self._specimen_array]
        return colors

    @property
    def names(self):
        """:obj:`list` of :obj:`int` :
        List of names of all :class:`Specimen`.
        """
        names = [s.name for s in self._specimen_array]
        return names

    #####################################
    #           Configuration           #
    #####################################
    def add_specimen(self,
                     name=None,
                     color=None,
                     mass=None,
                     collision_rate=None):
        """Adds a new specimen.

        Parameters
        ----------
        name : :obj:`str`, optional
        color : :obj:`str`, optional
        mass : int, optional
        collision_rate : :obj:`list` of :obj:`float`, optional
            Determines the collision probability between two specimen.
            Row (and column) of :attr:`collision_rate_matrix`.
        """
        # Handle Keyword Arguments
        if name is None:
            name = self.default_parameters['name']
        else:
            b_spm.Specimen.check_integrity(name=name)
        if color is None:
            color = self.default_parameters['color']
        else:
            b_spm.Specimen.check_integrity(color=color)
        if mass is None:
            mass = self.default_parameters['mass']
        else:
            b_spm.Specimen.check_integrity(mass=mass)
        if collision_rate is None:
            collision_rate = self.default_parameters['collision_rate']
        else:
            if type(collision_rate) is list:
                collision_rate = np.array(collision_rate)
            b_spm.Specimen.check_integrity(collision_rate=collision_rate)

        # Add the new row/column to self.collision_rate_matrix
        tmp_col_rat_mat = np.zeros(shape=(self.n + 1, self.n + 1),
                                   dtype=float)
        tmp_col_rat_mat[0:-1, 0:-1] = self.collision_rate_matrix
        self.collision_rate_matrix = tmp_col_rat_mat
        self.collision_rate_matrix[-1, :] = np.array(collision_rate)
        self.collision_rate_matrix[:, -1] = np.array(collision_rate)
        self._relink_all_collision_rates()

        # Create and add new Specimen
        new_specimen = b_spm.Specimen(name,
                                      color,
                                      mass,
                                      self.collision_rate_matrix[-1, :])
        self._specimen_array = np.append(self._specimen_array,
                                         [new_specimen])
        self.check_integrity()
        return

    def _relink_all_collision_rates(self):
        for (i_s, s) in enumerate(self._specimen_array):
            s.collision_rate = self.collision_rate_matrix[i_s, :]
        return

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self):
        """Sanity Check. Checks Integrity of all Attributes"""
        assert type(self.n) is int
        assert self.n >= 0
        for (i_s, s) in enumerate(self._specimen_array):
            assert type(s) is b_spm.Specimen
            s.check_integrity(check_instance=s)
            col_rat_1 = s.collision_rate
            col_rat_2 = self.collision_rate_matrix[i_s, :]
            col_rat_3 = self.collision_rate_matrix[:, i_s]
            assert np.shares_memory(col_rat_1, col_rat_2)
            assert isinstance(col_rat_1, np.ndarray)
            assert np.all(col_rat_1 == col_rat_2)
            assert np.all(col_rat_1 == col_rat_3)
        assert self.collision_rate_matrix.shape == (self.n, self.n)
        return

    def print(self):
        """Prints all Properties for Debugging."""
        print("Number of Specimen = {}".format(self.n))
        for s in self._specimen_array:
            s.print()
        print("Collision-Factor-Matrix = \n"
              "{}".format(self.collision_rate_matrix))
        return
