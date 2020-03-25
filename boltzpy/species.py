import h5py
import numpy as np

import boltzpy as bp
import boltzpy.constants as bp_c


class Species(bp.BaseClass):
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
    specimen_arr : :obj:`~numpy.array` [:class:`Specimen`]
        Array of all simulated :class:`Specimen`.
    """

    def __init__(self):
        self.specimen_arr = np.empty(shape=(0,),
                                     dtype=bp.Specimen)
        self.check_integrity()
        return

    def __getitem__(self, item):
        """Get a :class:`Specimen` instance by index or name.

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or :attr:`Specimen.name`

        Returns
        -------
        spc : :class:`Specimen`
        """
        item_idx = self.index(item)
        return self.specimen_arr[item_idx]

    @property
    def size(self):
        """:obj:`int` :
        Total number of :class:`Specimen`.
        """
        return self.specimen_arr.size

    @property
    def names(self):
        """:obj:`list` [:obj:`str`] :
        List of names of all :class:`Specimen`.
        """
        names = [s.name for s in self.specimen_arr]
        return names

    @property
    def mass(self):
        """:obj:`~numpy.array` [:obj:`int`] :
            Array of masses of all :class:`Specimen`.
            """
        mass = np.array([s.mass for s in self.specimen_arr], dtype=int)
        return mass

    @property
    def colors(self):
        """:obj:`list` [:obj:`str`] :
        List of colors of all :class:`Specimen`.
        """
        colors = [s.color for s in self.specimen_arr]
        return colors

    @property
    def colors_unused(self):
        """:obj:`list` [:obj:`str`] :
        List of all default colors with no
        :obj:`Specimen` of that color.
        """
        # black color should only used, if all other colors are used up
        used_colors = self.colors + ['black']
        unused_colors = [color for color in bp_c.SUPP_COLORS
                         if color not in used_colors]
        return unused_colors

    @property
    def collision_rates(self):
        """:obj:`~numpy.array` [:obj:`float`] :
        A factor for the collision probability between two specimen.

        The :attr:`Specimen.collision_rates <boltzpy.Specimen>`
        of the :class:`Specimen` are the rows and columns
        of this symmetric matrix.
        """
        # fill collision_rates with -1.0,
        # this way uninitialized rows may be found
        collision_rates = np.full(shape=(self.size, self.size),
                                  fill_value=-1.0)
        for (s_idx, s) in enumerate(self.specimen_arr):
            collision_rates[s_idx, :] = s.collision_rate[:]
        return collision_rates

    @property
    def is_configured(self):
        """:obj:`bool` :
        True, if all necessary attributes of the instance are set.
        False Otherwise.
        """
        return self.size >= 1

    #####################################
    #           Configuration           #
    #####################################
    def index(self, item):
        """Find the index of a :class:`Specimen`
        by its :attr:`Specimen.name`.

        If *item* is an :obj:`int`, then assert *item* is a proper index.
        If *item* is a :obj:`str`, then search :attr:`specimen_arr`
        and return the index where :attr:`Specimen.name` == *item*.

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or the searched for :attr:`Specimen.name`.

        Returns
        -------
        index : :obj:`int`

        Raises
        ------
        err_idx : :obj:`IndexError`
            *item* is an int and out of bounds.
        err_val : :obj:`ValueError`
            *item* is a string and no Specimen with name *item* exists.
        err_type : :obj:`TypeError`
            *name* is neither an :obj:`int` or a :obj:`str`
        """
        if isinstance(item, int):
            n_specimen = self.size
            # allow negative/reverse indexing
            if -n_specimen <= item < n_specimen:
                return item
            else:
                msg = ("item must be in range(-self.size, self.size). "
                       "Given item {} is not in [{},...,{}]."
                       "".format(item, -n_specimen, n_specimen))
                raise IndexError(msg)
        if isinstance(item, str):
            return self.names.index(item)
        else:
            msg = ("item must be either an int or a str,"
                   "not a {}".format(type(item)))
            raise TypeError(msg)

    def add(self,
            name=None,
            mass=None,
            collision_rate=None,
            color=None):
        """Add a new :obj:`Specimen` to :attr:`specimen_arr`.

        If a parameter is None,
        then the :obj:`Specimens <Specimen>` corresponding attribute
        is set to a default value.

        Parameters
        ----------
        name : :obj:`str`, optional
        mass : :obj:`int`, optional
        collision_rate : :obj:`~numpy.array` [:obj:`float`], optional
            Determines the collision probability between two specimen.
            Row (and column) of :attr:`collision_rates`.
        color : :obj:`str`, optional
        """
        bp.Specimen.check_parameters(name=name,
                                     mass=mass,
                                     collision_rate=collision_rate,
                                     color=color)
        if name is None:
            name = 'Specimen_' + str(self.size)
        if mass is None:
            mass = 1
        if collision_rate is None:
            collision_rate = np.ones(shape=self.size + 1, dtype=float)
        if color is None:
            color = next(iter(self.colors_unused), 'black')

        s_new = bp.Specimen(name=name,
                            mass=mass,
                            color=color,
                            collision_rate=collision_rate)
        # adjust collision rates of already existing species
        for (s_idx, s) in enumerate(self.specimen_arr):
            assert isinstance(s, bp.Specimen)
            s.collision_rate = np.append(s.collision_rate,
                                         s_new.collision_rate[s_idx])
        # add new specimen
        self.specimen_arr = np.append(self.specimen_arr, [s_new])
        self.check_integrity()
        return

    def edit(self,
             item,
             new_name=None,
             new_mass=None,
             new_collision_rate=None,
             new_color=None):
        """Edit attributes of the :obj:`Specimen`, denoted by *item*.

        All attributes, where *new_<ATTRIBUTE>* is None,
        stay unchanged. They are NOT set to None.

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or name of the :obj:`Specimen` to be edited
        new_name : :obj:`str`, optional
        new_mass : :obj:`int`, optional
        new_collision_rate : :obj:`~numpy.array` [:obj:`float`], optional
        new_color : :obj:`str`, optional
        """
        (s_idx, s) = (self.index(item), self[item])
        bp.Specimen.check_parameters(name=new_name,
                                     mass=new_mass,
                                     collision_rate=new_collision_rate,
                                     color=new_color)
        if new_name is not None:
            s.name = new_name
        if new_color is not None:
            s.color = new_color
        if new_mass is not None:
            s.mass = new_mass
        if new_collision_rate is not None:
            assert new_collision_rate.size == self.size
            s.collision_rate = new_collision_rate
            # edit the other species collision rates,
            # since collision_rates must by symmetric
            for (s2_idx, s2) in enumerate(self.specimen_arr):
                s2.collision_rate[s_idx] = s.collision_rate[s2_idx]

        self.check_integrity()
        return

    def remove(self, item):
        """Removes a :obj:`Specimen`.

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or name of the :obj:`Specimen` to be removed.
        """
        item_idx = self.index(item)
        self.specimen_arr = np.delete(self.specimen_arr, item_idx)
        # remove items entry in in every remaining specimens collision rate
        for s in self.specimen_arr:
            s.collision_rate = np.delete(s.collision_rate, item_idx)
        self.check_integrity()
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
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Species`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "Species"
        self = Species()

        try:
            # read data from file
            names = hdf5_group["Names"]
            masses = hdf5_group["Masses"][()]
            col_rate = hdf5_group["collision_rates"][()]
            colors = hdf5_group["Colors"][()]
            assert names.ndim is 1
            assert col_rate.ndim is 2
            assert names.shape == colors.shape == masses.shape
            assert col_rate.shape == (names.size, names.size)
            # setup s iteratively
            for i in range(names.size):
                self.add(name=names[i],
                         mass=int(masses[i]),
                         collision_rate=col_rate[i, 0:i + 1],
                         color=colors[i])
        except KeyError:
            pass
        self.check_integrity()
        return self

    def save(self, hdf5_group):
        """Write the parameters of the :obj:`Species` instance
        into the given HDF5 group.


        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
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
        hdf5_group["Masses"] = self.mass
        hdf5_group["collision_rates"] = self.collision_rates
        hdf5_group["Colors"] = np.array(self.colors,
                                        dtype=h5py_string_type)
        return

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self):
        """Sanity Check. Checks Integrity of all Attributes"""
        assert isinstance(self.specimen_arr, np.ndarray)
        assert self.specimen_arr.ndim == 1
        assert self.specimen_arr.size == self.size
        for s in self.specimen_arr:
            assert isinstance(s, bp.Specimen)
            s.check_integrity()
            assert s.collision_rate.size == self.size

        assert type(self.size) is int
        assert self.size >= 0

        assert isinstance(self.names, list)
        assert all(isinstance(name, str) for name in self.names)
        assert all(self.names.count(name) == 1 for name in self.names)

        assert isinstance(self.mass, np.ndarray)
        assert self.mass.dtype == int
        assert self.mass.size == self.size
        assert all(m > 0 for m in self.mass)

        assert isinstance(self.colors, list)
        assert all(type(color) == str for color in self.colors)
        assert all(color in bp_c.SUPP_COLORS for color in self.colors)

        assert isinstance(self.colors_unused, list)
        assert all(type(color) == str for color in self.colors_unused)
        assert all(color in bp_c.SUPP_COLORS
                   for color in self.colors_unused)
        assert all(color not in self.colors_unused for color in self.colors)

        assert isinstance(self.collision_rates, np.ndarray)
        assert self.collision_rates.dtype == float
        assert self.collision_rates.ndim == 2
        assert self.collision_rates.shape == (self.size, self.size)
        for (specimen_idx, specimen) in enumerate(self.specimen_arr):
            # collision rates is a symmetric matrix
            s_col_rate = specimen.collision_rate
            row = self.collision_rates[specimen_idx, :]
            column = self.collision_rates[:, specimen_idx]
            assert isinstance(specimen.collision_rate, np.ndarray)
            assert np.all(s_col_rate == row)
            assert np.all(s_col_rate == column)
        return

    def __str__(self):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance."""
        description = "Number of Specimen = {}\n".format(self.size)
        for (i_s, specimen) in enumerate(self.specimen_arr):
            description += 'Specimen_{}:'.format(i_s)
            description += '\n'
            description += '\t'
            description += specimen.__str__().replace('\n', '\n\t')
            description += '\n\n'
        description += "Collision-Factor-Matrix: \n"
        matrix_string = self.collision_rates.__str__()
        description += "\t" + matrix_string.replace('\n', '\n\t')
        return description
