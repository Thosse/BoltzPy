
from . import specimen as b_spm

import numpy as np
import h5py


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
            item = self._get_item_position(item)
            return self._specimen_array[item]
        else:
            message = "item must be either the index(int) or name(str)"
            raise TypeError(message)

    @property
    def default_parameters(self):
        """:obj:`dict`: Default parameters for :meth:`add_specimen`.
        """
        used_colors = [s.color for s in self._specimen_array]
        color = next((color for color in self.colors_supported
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
    def colors_supported(self):
        """:obj:`list` of :obj:`str`:
                List of all default colors."""
        supported_colors = ['blue', 'red', 'green',
                            'yellow', 'brown', 'gray',
                            'olive', 'purple', 'cyan',
                            'orange', 'pink', 'lime',
                            'black']
        return supported_colors

    @property
    def colors_unused(self):
        """:obj:`list` of :obj:`str`:
                        List of all default colors without any
                        :obj:`Specimen` of that color."""
        # black color should only used, if all other colors are used up
        used_colors = self.colors + ['black']
        unused_colors = [color for color in self.colors_supported
                         if color not in used_colors]
        return unused_colors

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
    def edit_specimen(self,
                      item,
                      name=None,
                      color=None,
                      mass=None,
                      collision_rate=None):
        """Edits attributes of a :obj:`Specimen`.

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or name of the :obj:`Specimen` to be edited
        name : :obj:`str`, optional
        color : :obj:`str`, optional
        mass : int, optional
        collision_rate : :obj:`list` of :obj:`float`, optional
            Determines the collision probability between two specimen.
            Row (and column) of :attr:`collision_rate_matrix`.
        """
        item = self._get_item_position(item)

        # Handle Keyword Arguments
        if name is not None:
            b_spm.Specimen.check_integrity(name=name)
            self[item].name = name
        if color is not None:
            b_spm.Specimen.check_integrity(color=color)
            self[item].color = color
        if mass is not None:
            b_spm.Specimen.check_integrity(mass=mass)
            self[item].mass = mass
        if collision_rate is not None:
            if type(collision_rate) is list:
                collision_rate = np.array(collision_rate)
            b_spm.Specimen.check_integrity(collision_rate=collision_rate)
            self.collision_rate_matrix[item, :] = collision_rate
            self.collision_rate_matrix[:, item] = collision_rate

        self.check_integrity()
        return

    def add_specimen(self,
                     name=None,
                     color=None,
                     mass=None,
                     collision_rate=None):
        """Adds a new :obj:`Specimen`.

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
        default = self.default_parameters

        # Add the new row/column to self.collision_rate_matrix
        tmp_col_rat_mat = np.zeros(shape=(self.n + 1, self.n + 1),
                                   dtype=float)
        tmp_col_rat_mat[0:-1, 0:-1] = self.collision_rate_matrix
        self.collision_rate_matrix = tmp_col_rat_mat
        self.collision_rate_matrix[-1, :] = default['collision_rate']
        self.collision_rate_matrix[:, -1] = default['collision_rate']
        self._relink_all_collision_rates()

        # Create and append default Specimen
        new_specimen = b_spm.Specimen(default['name'],
                                      default['color'],
                                      default['mass'],
                                      self.collision_rate_matrix[-1, :])
        self._specimen_array = np.append(self._specimen_array,
                                         [new_specimen])
        # edit new Specimen
        self.edit_specimen(-1,
                           name=name,
                           color=color,
                           mass=mass,
                           collision_rate=collision_rate)
        self.check_integrity()
        return

    def delete_specimen(self, item):
        """Removes a :obj:`Specimen`.

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or name of the :obj:`Specimen` to be removed.
        """
        item = self._get_item_position(item)
        self._specimen_array = np.delete(self._specimen_array, item)
        self.collision_rate_matrix = np.delete(self.collision_rate_matrix,
                                               item,
                                               axis=0)
        self.collision_rate_matrix = np.delete(self.collision_rate_matrix,
                                               item,
                                               axis=1)
        self._relink_all_collision_rates()
        return

    def _get_item_position(self, item):
        assert type(item) in [int, str]
        if type(item) is str:
            item = self.names.index(item)
        assert type(item) is int
        if item >= self.n:
            msg = 'index {} is out of bounds ' \
                  'for axis 0 ' \
                  'with size {}'.format(item, self.n)
            raise IndexError(msg)
        return item

    def _relink_all_collision_rates(self):
        """Resets the :attr:`Specimen.collision_rate` attributes
        for every single :obj:`Specimen`"""
        for (i_s, s) in enumerate(self._specimen_array):
            s.collision_rate = self.collision_rate_matrix[i_s, :]
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_file):
        """Creates and Returns a :obj:`Species` object,
        based on the parameters in the given file.

        Parameters
        ----------
        hdf5_file : h5py.File
            Opened HDF5 :obj:`Configuration` file.

        Returns
        -------
        :obj:`Species`
        """
        s = Species()
        # read data from file
        names = hdf5_file["Names"]
        colors = hdf5_file["Colors"].value
        masses = hdf5_file["Masses"].value
        col_rate = hdf5_file["Collision_Rate_Matrix"].value
        # Todo move into elementwise check_integrity
        assert len(names.shape) is 1 and len(col_rate.shape) is 2
        assert names.shape == colors.shape == masses.shape
        assert col_rate.shape == (names.size, names.size)
        # setup s iteratively
        for i in range(names.size):
            s.add_specimen(name=names[i],
                           color=colors[i],
                           mass=masses[i],
                           collision_rate=col_rate[i, 0:i+1])
        s.check_integrity()
        return s

    def save(self, hdf5_file):
        """Writes the parameters of the :obj:`Species` object
        to the given file.

        Parameters
        ----------
        hdf5_file : h5py.File
            Opened HDF5 :obj:`Configuration` file.
        """
        self.check_integrity()
        for key in hdf5_file.keys():
            del hdf5_file[key]
        # Set special data type for String-Arrays
        #  noinspection PyUnresolvedReferences
        h5py_string_type = h5py.special_dtype(vlen=str)
        # Write Attributes
        hdf5_file["Names"] = np.array(self.names,
                                      dtype=h5py_string_type)
        hdf5_file["Colors"] = np.array(self.colors,
                                       dtype=h5py_string_type)
        hdf5_file["Masses"] = self.mass
        hdf5_file["Collision_Rate_Matrix"] = self.collision_rate_matrix
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

    def print(self, print_collision_rates_separately=False):
        """Prints all Properties for Debugging."""
        print("Number of Specimen = {}".format(self.n))
        for s in self._specimen_array:
            s.print(print_collision_rates_separately)
        if not print_collision_rates_separately:
            print("Collision-Factor-Matrix = \n"
                  "{}".format(self.collision_rate_matrix))
        return
