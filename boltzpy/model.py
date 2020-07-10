import numpy as np
import h5py

import boltzpy as bp


class Model(bp.BaseClass):
    r"""Manages the Velocity Grids of all
    :class:`~boltzpy.Species`.


    Note
    ----
    Just as in the :class:`Grid` class,
    the parameter :attr:`iMG` describes the
    position/physical values of all  Grid points.
    All entries must be viewed as multiples of :attr:`delta:

        :math:`pMG = iMG \cdot d`.

    Note that velocity grid points may occur in multiple
    :class:`Velocity Grids <boltzpy.Grid>`.
    Array of shape (:attr:`size`, :attr:`ndim`)

    Parameters
    ----------
    masses : :obj:`~numpy.array` [:obj:`int`]
        Denotes the masses of all specimen.
    shapes : :obj:`~numpy.array` [:obj:`int`]
        Denotes the shape of each :class:`boltzpy.Grid`.
    delta : :obj:`float`
        Internal step size (delta) of all :class:`Grids <boltzpy.Grid>`.
        This is NOT the physical distance between grid points.
    spacings : :obj:`~numpy.array` [:obj:`int`]
        Denotes the spacing of each :class:`boltzpy.Grid`.

    Attributes
    ----------
    ndim : :obj:`int`
        Dimensionality of all Velocity :class:`Grids <boltzpy.Grid>`.
    size : :obj:`int` :
        The total number of velocity grid points over all grids.
    specimen : :obj:`int` :
        The number of different specimen or velocity grids.
    index_offset : :obj:`~numpy.array` [:obj:`int`]
        Denotes the beginning of the respective velocity grid
        in the multi grid :attr:`iMG`.
    vGrids : :obj:`~numpy.array` [:class:`~boltzpy.Grid`]
        Array of all Velocity :class:`Grids <boltzpy.Grid>`.
    iMG : :obj:`~numpy.array` [:obj:`int`]
        The *integer Multi-Grid*.
        It is a concatenation of all
        Velocity integer Grids
        (:attr:`Grid.iG <boltzpy.Grid>`).
    """

    def __init__(self,
                 masses,
                 shapes,
                 delta,
                 spacings,
                 collision_factors,
                 algorithm_relations="all",
                 algorithm_weights="uniform"):
        assert isinstance(masses, (list, tuple, np.ndarray))
        self.masses = np.array(masses, dtype=int)
        assert self.masses.ndim == 1

        assert isinstance(shapes, (list, tuple, np.ndarray))
        self.shapes = np.array(shapes, dtype=int)
        assert self.shapes.ndim == 2

        self.delta = np.float(delta)

        assert isinstance(spacings, (list, tuple, np.ndarray))
        self.spacings = np.array(spacings, dtype=int)
        assert self.spacings.ndim == 1

        assert isinstance(collision_factors, (list, tuple, np.ndarray))
        self.collision_factors = np.array(collision_factors,
                                          dtype=float)
        assert self.collision_factors.ndim == 2

        self.algorithm_relations = str(algorithm_relations)
        self.algorithm_weights = str(algorithm_weights)

        self.ndim = self.shapes.shape[1]
        self.size = np.sum(np.prod(self.shapes, axis=1))
        self.specimen = self.masses.size

        # set up each Velocity Grid
        self.vGrids = np.array([bp.Grid(self.shapes[i],
                                        self.delta,
                                        self.spacings[i],
                                        is_centered=True)
                                for i in self.species],
                               dtype=bp.Grid)
        self.iMG = np.concatenate([G.iG for G in self.vGrids])
        self.index_offset = np.zeros(self.specimen + 1, dtype=int)
        for s in self.species:
            self.index_offset[s + 1:] += self.vGrids[s].size
        return

    # Todo properly vectorize
    @staticmethod
    def default_spacing(masses):
        # compute spacings by mass ratio
        lcm = int(np.lcm.reduce(masses))
        spacings = [2 * lcm // int(m) for m in masses]
        return spacings

    #####################################
    #           Properties              #
    #####################################
    @property
    def species(self):
        return np.arange(self.specimen)

    @property
    def index_range(self):
        # Todo remove soon
        result = np.zeros((self.specimen, 2), dtype=int)
        result[:, 0] = self.index_offset[0:self.specimen]
        result[:, 1] = self.index_offset[1:]
        return result

    @property
    def maximum_velocity(self):
        """:obj:`float`
        Maximum physical velocity for every sub grid."""
        return np.max(self.iMG * self.delta)

    @staticmethod
    def parameters():
        return {"masses",
                "shapes",
                "delta",
                "spacings",
                "collision_factors",
                "algorithm_relations",
                "algorithm_weights"}

    @staticmethod
    def attributes():
        attrs = Model.parameters()
        attrs.update({"ndim",
                      "size",
                      "specimen",
                      "index_offset",
                      "index_range",
                      "species",
                      "maximum_velocity"})
        return attrs

    #####################################
    #               Indexing            #
    #####################################
    def find_index(self,
                   index_of_specimen,
                   integer_value):
        """Find index of given grid_entry in :attr:`iMG`
        Returns None, if the value is not in the specified Grid.

        Parameters
        ----------
        index_of_specimen : :obj:`int`
        integer_value : :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        global_index : :obj:`int` of :obj:`None`

        """
        local_index = self.vGrids[index_of_specimen].get_idx(integer_value)
        if local_index < 0:
            return None
        else:
            index_offset = self.index_range[index_of_specimen, 0]
            global_index = index_offset + local_index
            assert np.all(self.iMG[global_index] == integer_value)
            return global_index

    def get_specimen(self, velocity_idx):
        """Get :class:`boltzpy.Specimen` index
        of given velocity in :attr:`iMG`.

        Parameters
        ----------
        velocity_idx : :obj:`int`

        Returns
        -------
        index : :obj:`int`

        Raises
        ------
        err_idx : :obj:`IndexError`
            If *velocity_idx* is out of the range of
            :attr:`Model.iMG`.
        """
        for (i, [beg, end]) in enumerate(self.index_range):
            if beg <= velocity_idx < end:
                return i
        msg = 'The given index ({}) points out of the boundaries of ' \
              'iMG, the concatenated Velocity Grid.'.format(velocity_idx)
        raise IndexError(msg)

    #####################################
    #           Visualization           #
    #####################################
    #: :obj:`list` [:obj:`dict`]:
    #: Default plot_styles for :meth::`plot`
    plot_styles = [{"marker": 'o', "color": "r", "facecolors": 'none'},
                   {"marker": 's', "color": "b", "facecolors": 'none'},
                   {"marker": 'x', "color": "black"},
                   {"marker": 'D', "color": "green", "facecolors": "none"}]

    def plot(self, plot_object=None):
        """Plot the Grid using matplotlib.

        Parameters
        ----------
        plot_object : TODO Figure? matplotlib.pyplot?
        """
        show_plot_directly = plot_object is None
        if plot_object is None:
            # Choose standard pyplot
            import matplotlib.pyplot as plt
            plot_object = plt
        # Plot Grids as scatter plot
        for (idx_G, G) in enumerate(self.vGrids):
            G.plot(plot_object, **(self.plot_styles[idx_G]))
        if show_plot_directly:
            plot_object.show()
        return plot_object

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`Model` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Model`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "Model"

        parameters = dict()
        for param in Model.parameters():
            parameters[param] = hdf5_group[param][()]

        self = Model(**parameters)
        return self

    def save(self, hdf5_group, write_all=False):
        """Write the main parameters of the :class:`Model` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        write_all : :obj:`bool`
            If True, write all attributes and properties to the file,
            even the unnecessary ones. Useful for testing,
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity()

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = self.__class__.__name__
        # write attributes to file
        if write_all:
            for attr in Model.attributes():
                hdf5_group[attr] = self.__getattribute__(attr)
        else:
            for attr in Model.parameters():
                hdf5_group[attr] = self.__getattribute__(attr)

        # check that the class can be reconstructed from the save
        other = Model.load(hdf5_group)
        assert self == other
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check."""
        assert isinstance(self.ndim, int)
        assert self.ndim in {2, 3}

        assert isinstance(self.maximum_velocity, float)
        assert self.maximum_velocity > 0

        assert isinstance(self.delta, float)
        assert self.delta > 0

        assert isinstance(self.shapes, np.ndarray)
        assert self.shapes.shape[0] == self.specimen
        assert np.array_equal(self.shapes,
                              np.array([G.shape for G in self.vGrids]))
        assert len({shape.ndim for shape in self.shapes}) <= 1, (
            "All Grids must have the same dimension.\n"
            "Given dimensions = "
            "{}".format([shape.shape[1] for shape in self.shapes]))
        assert all(len(set(shape)) == 1 for shape in self.shapes), (
            "All Velocity Grids must be squares.\n"
            "Given shapes = "
            "{}".format(self.shapes))

        assert isinstance(self.spacings, np.ndarray)
        assert self.spacings.shape[0] == self.specimen
        assert np.array_equal(self.spacings,
                              np.array([G.spacing for G in self.vGrids]))

        assert isinstance(self.vGrids, np.ndarray)
        assert self.vGrids.size == self.specimen
        assert self.vGrids.ndim == 1
        assert self.vGrids.dtype == 'object'
        for G in self.vGrids:
            isinstance(G, bp.Grid)
            G.check_integrity()

        assert isinstance(self.index_range, np.ndarray)
        assert self.index_range.dtype == int
        assert self.index_range.ndim == 2
        assert self.specimen == self.index_range.shape[0]
        assert self.index_range.shape[1] == 2
        assert np.all(self.index_range >= 0)
        assert np.all(self.index_range[1:, 0] == self.index_range[0:-1, 1])
        assert np.all(self.index_range[:, 0] < self.index_range[:, 1])
        assert np.array_equal(self.index_offset[1:] - self.index_offset[:-1],
                              np.array([G.size for G in self.vGrids]))

        assert isinstance(self.iMG, np.ndarray)
        assert self.iMG.dtype == int
        assert self.iMG.ndim == 2
        assert self.iMG.shape[0] == self.index_range[-1, 1]

        assert self.algorithm_relations in {"all", "fast"}
        assert self.algorithm_weights in {"uniform"}
        return

    def __str__(self,
                write_physical_grid=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance.
        """
        description = ''
        description += "Dimension = {}\n".format(self.ndim)
        description += "Total Size = {}\n".format(self.size)

        if self.vGrids is not None:
            for (idx_G, vGrid) in enumerate(self.vGrids):
                description += 'Specimen_{idx}:\n\t'.format(idx=idx_G)
                grid_str = vGrid.__str__(write_physical_grid)
                description += grid_str.replace('\n', '\n\t')
                description += '\n'
        return description[:-1]
