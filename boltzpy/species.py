
import boltzpy.constants as b_const
import boltzpy.specimen as b_spm

import h5py
import numpy as np


class Species:
    """Array of all simulated :class:`Specimen`.

    ..todo::
        - add attribute parent/parent_array to specimen
        - add index attribute to specimen
        - move most integrity checks from species to specimen
        - move editing of collision_rate to specimen
        - if index is None, collision_rate is stored locally (for init)
        - if index is int, collision rate is read from matrix from parent
        - when adding specimen to species -> give index and add coll_rate


    Attributes
    ----------
    specimen_array : :obj:`~numpy.array` [:class:`Specimen`]
        Array of all simulated :class:`Specimen`.
    collision_rate_matrix : :obj:`~numpy.array` [:obj:`float`]
        Determines the collision probability between two specimen.
        The :attr:`Specimen.collision_rates <boltzpy.Specimen>`
        of the :class:`Specimen` are the rows and columns of this matrix.
    """
    def __init__(self):
        self.specimen_array = np.empty(shape=(0,),
                                       dtype=b_spm.Specimen)
        self.collision_rate_matrix = np.zeros(shape=(0, 0), dtype=float)
        self.check_integrity()
        return

    def __getitem__(self, item):
        """Get a :class:`Specimen` by index or name.

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or :attr:`Specimen.name`

        Returns
        -------
        :class:`Specimen`

        Raises
        ------
        IndexError
            *item* is an int and out of bounds.
        ValueError
            *item* is a string and no Specimen with name *item* exists.
        TypeError
            *name* is neither an :obj:`int` or a :obj:`str`
        """
        if isinstance(item, int):
            return self.specimen_array[item]
        if isinstance(item, str):
            item_idx = self.index(item)
            return self.specimen_array[item_idx]
        else:
            raise TypeError("item must be either an int or a str,"
                            "not a {}".format(type(item)))

    @property
    def size(self):
        """:obj:`int` :
        Total number of :class:`Specimen`.
        """
        return self.specimen_array.size

    @property
    def mass(self):
        """:obj:`~numpy.array` [:obj:`int`] :
            Array of masses of all :class:`Specimen`.
            """
        mass = np.array([s.mass for s in self.specimen_array], dtype=int)
        return mass

    @property
    def colors(self):
        """:obj:`list` [:obj:`str`] :
        List of colors of all :class:`Specimen`.
        """
        colors = [s.color for s in self.specimen_array]
        return colors

    @property
    def colors_unused(self):
        """:obj:`list` [:obj:`str`]:
        List of all default colors with no
        :obj:`Specimen` of that color.
        """
        # black color should only used, if all other colors are used up
        used_colors = self.colors + ['black']
        unused_colors = [color for color in b_const.SUPP_COLORS
                         if color not in used_colors]
        return unused_colors

    @property
    def names(self):
        """:obj:`list` [:obj:`str`] :
        List of names of all :class:`Specimen`.
        """
        names = [s.name for s in self.specimen_array]
        return names

    #####################################
    #           Configuration           #
    #####################################
    def index(self, name):
        """Find a :class:`Specimen`
        with :attr:`Specimen.name` == *name*
        in :attr:`specimen_array`
        and return its index.

        Parameters
        ----------
        name : :obj:`str`
            The searched for :attr:`Specimen.name`.

        Returns
        -------
        :obj:`int`

        Raises
        ------
        AssertionError
            *name* is not a :obj:`str`
        ValueError
            No :class:`Specimen` with :attr:`~Specimen.name` *name* exists.
        """
        assert isinstance(name, str)
        return self.names.index(name)

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
        collision_rate : :obj:`~numpy.array` [:obj:`float`], optional
            Determines the collision probability between two specimen.
            Row (and column) of :attr:`collision_rate_matrix`.
        """
        if isinstance(item, int):
            item_idx = item
        elif isinstance(item, str):
            item_idx = self.index(item)
        else:
            raise TypeError("item must be either an int or a str, "
                            "not a {}".format(type(item)))

        b_spm.Specimen.check_parameters(name=name,
                                        color=color,
                                        mass=mass,
                                        collision_rate=collision_rate)
        if name is not None:
            self[item_idx].name = name
        if color is not None:
            self[item_idx].color = color
        if mass is not None:
            self[item_idx].mass = mass
        if collision_rate is not None:
            if type(collision_rate) is list:
                collision_rate = np.array(collision_rate, dtype=float)
            assert collision_rate.size == self.size
            self.collision_rate_matrix[item_idx, :] = collision_rate
            self.collision_rate_matrix[:, item_idx] = collision_rate

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
        mass : :obj:`int`, optional
        collision_rate : :obj:`~numpy.array` [:obj:`float`], optional
            Determines the collision probability between two specimen.
            Row (and column) of :attr:`collision_rate_matrix`.
        """
        # Setup default parameters
        default_name = 'Specimen_' + str(self.size)
        default_color = next(iter(self.colors_unused), 'black')
        default_mass = 1
        default_collision_rate = np.ones(self.size + 1)

        # Add the new row/column to self.collision_rate_matrix
        tmp_col_rat_mat = np.zeros(shape=(self.size + 1, self.size + 1),
                                   dtype=float)
        tmp_col_rat_mat[0:-1, 0:-1] = self.collision_rate_matrix
        tmp_col_rat_mat[-1, :] = default_collision_rate
        tmp_col_rat_mat[:, -1] = default_collision_rate
        self.collision_rate_matrix = tmp_col_rat_mat
        self._relink_all_collision_rates()

        # Create and append default Specimen
        new_specimen = b_spm.Specimen(default_name,
                                      default_color,
                                      default_mass,
                                      self.collision_rate_matrix[-1, :])
        self.specimen_array = np.append(self.specimen_array,
                                        [new_specimen])
        # edit new Specimen
        self.edit_specimen(-1,
                           name=name,
                           color=color,
                           mass=mass,
                           collision_rate=collision_rate)
        self.check_integrity()
        return

    # Todo do unittest, to verify this actually works correctly
    def delete_specimen(self, item):
        """Removes a :obj:`Specimen`.

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or name of the :obj:`Specimen` to be removed.
        """
        if isinstance(item, int):
            index = item
        elif isinstance(item, str):
            index = self.index(item)
        else:
            raise TypeError

        self.specimen_array = np.delete(self.specimen_array, index)
        self.collision_rate_matrix = np.delete(self.collision_rate_matrix,
                                               index,
                                               axis=0)
        self.collision_rate_matrix = np.delete(self.collision_rate_matrix,
                                               index,
                                               axis=1)
        self._relink_all_collision_rates()
        return

    def _relink_all_collision_rates(self):
        """Resets each :class:`Specimens <Specimen>`
        :attr:`~Specimen.collision_rate`
        to a view of its respective row in the :attr:`collision_rate_matrix`
        for every single :obj:`Specimen`.
        Note that he :attr:`Specimen.collision_rate` attributes are pointers.
        """
        for (i_s, s) in enumerate(self.specimen_array):
            s.collision_rate = self.collision_rate_matrix[i_s, :]
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :obj:`Species` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group`

        Returns
        -------
        :class:`Species`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "Species"
        self = Species()

        try:
            # read data from file
            names = hdf5_group["Names"]
            colors = hdf5_group["Colors"].value
            masses = hdf5_group["Masses"].value
            col_rate = hdf5_group["Collision_Rate_Matrix"].value
            assert len(names.shape) is 1 and len(col_rate.shape) is 2
            assert names.shape == colors.shape == masses.shape
            assert col_rate.shape == (names.size, names.size)
            # setup s iteratively
            for i in range(names.size):
                self.add_specimen(name=names[i],
                                  color=colors[i],
                                  mass=masses[i],
                                  collision_rate=col_rate[i, 0:i+1])
        except KeyError:
            pass
        self.check_integrity()
        return self

    def save(self, hdf5_group):
        """Write the parameters of the :obj:`Species` instance
        into the given HDF5 group.


        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity()

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = "Species"

        # Define special data type for String-Arrays
        # noinspection PyUnresolvedReferences
        h5py_string_type = h5py.special_dtype(vlen=str)

        # Write Attributes
        hdf5_group["Names"] = np.array(self.names,
                                       dtype=h5py_string_type)
        hdf5_group["Colors"] = np.array(self.colors,
                                        dtype=h5py_string_type)
        hdf5_group["Masses"] = self.mass
        hdf5_group["Collision_Rate_Matrix"] = self.collision_rate_matrix
        return

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self):
        """Sanity Check. Checks Integrity of all Attributes"""
        assert type(self.size) is int
        assert self.size >= 0
        for (i_s, specimen) in enumerate(self.specimen_array):
            assert isinstance(specimen, b_spm.Specimen)
            specimen.check_integrity()
            # names must be unique
            assert self.names.count(specimen.name) == 1
            # collision rate is a link to a row of the collision matrix
            col_rat_1 = specimen.collision_rate
            col_rat_2 = self.collision_rate_matrix[i_s, :]
            col_rat_3 = self.collision_rate_matrix[:, i_s]
            assert np.shares_memory(col_rat_1, col_rat_2)
            assert isinstance(col_rat_1, np.ndarray)
            assert np.all(col_rat_1 == col_rat_2)
            assert np.all(col_rat_1 == col_rat_3)
        assert self.collision_rate_matrix.shape == (self.size, self.size)
        return

    def __str__(self):
        """Converts the instance to a string, describing all attributes."""
        description = "Number of Specimen = {}\n".format(self.size)
        for (i_s, specimen) in enumerate(self.specimen_array):
            description += 'Specimen_{}:'.format(i_s)
            description += '\n'
            description += '\t'
            description += specimen.__str__().replace('\n', '\n\t')
            description += '\n\n'
        description += "Collision-Factor-Matrix: \n"
        matrix_string = self.collision_rate_matrix.__str__()
        description += "\t" + matrix_string.replace('\n', '\n\t')
        return description
