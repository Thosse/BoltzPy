import numpy as np
import h5py
from scipy.sparse import csr_matrix
from time import process_time

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
                 collision_relations=None,
                 collision_weights=None,
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

        # setup collisions
        if collision_relations is None:
            self.collision_relations = self.compute_relations()
        else:
            self.collision_relations = np.array(collision_relations)
        if collision_weights is None:
            self.collision_weights = self.compute_weights()
        else:
            self.collision_weights = np.array(collision_weights)

        # create collision_matrix
        col_mat = np.zeros((self.size, self.collision_weights.size), dtype=float)
        for [r, rel] in enumerate(self.collision_relations):
            weight = self.collision_weights[r]
            col_mat[rel, r] = weight * np.array([-1, 1, -1, 1])
        self.collision_matrix = csr_matrix(col_mat)
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

    @property
    def collisions(self):
        return self.collision_weights.size

    @property
    def collision_invariants(self):
        rank = np.linalg.matrix_rank(self.collision_matrix.toarray())
        maximum_rank = np.min([self.size, self.collision_weights.size])
        return maximum_rank - rank

    @staticmethod
    def parameters():
        return {"masses",
                "shapes",
                "delta",
                "spacings",
                "collision_factors",
                "collision_relations",
                "collision_weights",
                "algorithm_relations",
                "algorithm_weights"}

    @staticmethod
    def attributes():
        attrs = Model.parameters()
        attrs.update({"ndim",
                      "size",
                      "collision_matrix",
                      "specimen",
                      "index_offset",
                      "index_range",
                      "species",
                      "maximum_velocity",
                      "collision_invariants"})
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

    def get_spc(self, indices):
        """Get the specimen of given indices of :attr:`iMG`.

        Parameters
        ----------
        indices : :obj:`~numpy.array` [ :obj:`int` ]

        Returns
        -------
        species : :obj:`~numpy.array` [ :obj:`int` ]

        Raises
        ------
        err_idx : :obj:`IndexError`
            If *velocity_idx* is out of the range of
            :attr:`Model.iMG`.
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        species = np.full(indices.shape, -1, dtype=int)
        for s in self.species:
            offset = self.index_offset[s]
            species = np.where(indices >= offset, s, species)
        species = np.where(indices >= self.index_offset[-1], -1, species)
        assert np.all(species >= 0)
        return species

    ################################################
    #        Sorting and Ordering Collisions       #
    ################################################
    @staticmethod
    def key_index(relations):
        """"Sorts a set of relations,
        such that each relations indices are in ascending"""
        return np.sort(relations, axis=-1)

    def key_species(self, relations):
        species = self.get_spc(relations)
        return np.sort(species, axis=-1)

    def key_area(self, relations):
        (v0, v1, w0, w1) = self.iMG[relations.transpose()]
        # in 2D cross returns only the z component -> use absolute
        result = np.zeros(relations.shape[:-1] + (2,), dtype=float)
        if self.ndim == 2:
            area_1 = np.abs(np.cross(v1 - v0, w1 - v0))
            area_2 = np.abs(np.cross(w1 - w0, w1 - v0))
        # in 3D cross returns the vector -> use norm
        elif self.ndim == 3:
            area_1 = np.linalg.norm(np.cross(v1 - v0, w1 - v0), axis=-1)
            area_2 = np.linalg.norm(np.cross(w1 - w0, w1 - v0), axis=-1)
        else:
            raise NotImplementedError
        result[..., 0] = 0.5 * (area_1 + area_2)

        result[..., 1] = (np.linalg.norm(v1 - v0, axis=-1)
                          + np.linalg.norm(w1 - w0, axis=-1)
                          + 2 * np.linalg.norm(v0 - w1, axis=-1))
        return result

    def key_angle(self, relations):
        (v0, v1, w0, w1) = self.iMG[relations.transpose()]
        dv = v1 - v0
        gcd = np.gcd.reduce(dv.transpose())
        if self.ndim == 2:
            gcd = gcd[..., np.newaxis]
        elif self.ndim == 2:
            gcd = gcd[..., np.newaxis, np.newaxis]
        else:
            raise NotImplementedError
        angles = dv // gcd
        #
        angles = np.sort(np.abs(angles), axis=-1)
        return angles

    # Todo switch oder of key_function an relations, give group a default key=species
    def group(self, key_function, relations=None):
        if relations is None:
            relations = self.collision_relations
        assert relations.ndim > 1

        grouped = dict()
        keys = key_function(relations)
        # unique values _ are not needed
        unique_keys = np.unique(keys, axis=0)
        for key in unique_keys:
            pos = np.where(np.all(keys == key, axis=-1))
            grouped[tuple(key)] = relations[pos]
        return grouped

    def filter(self, key_function=None, relations=None):
        if key_function is None:
            key_function = self.key_index
        if relations is None:
            rels = self.collision_relations
        else:
            rels = relations

        keys = key_function(rels)
        # unique values _ are not needed
        _, positions = np.unique(keys, return_index=True, axis=0)
        if relations is None:
            self.collision_relations = self.collision_relations[positions]
            self.collision_weights = self.collision_weights[positions]
            return
        else:
            return relations[positions]

    def sort(self, key_function=None, relations=None):
        if key_function is None:
            key_function = self.key_index
        if relations is None:
            rels = self.collision_relations
        else:
            rels = relations

        keys = key_function(rels)
        # lexsort sorts columns, thus we transpose first
        # flip keys, as the last key (row) has highest priority
        positions = np.lexsort(np.flip(keys.transpose(), axis=0), axis=0)
        if relations is None:
            self.collision_relations = self.collision_relations[positions]
            self.collision_weights = self.collision_weights[positions]
            return None
        else:
            return relations[positions]

    ##################################
    #           Collisions           #
    ##################################
    def compute_weights(self, relations=None):
        """Generates and returns the :attr:`collision_weights`,
        based on the given relations and :attr:`algorithm_weights`
        """
        if relations is None:
            relations = self.collision_relations
        species = self.key_species(relations)[:, 1:3]
        coll_factors = self.collision_factors[species[..., 0], species[..., 1]]
        if self.algorithm_weights == "uniform":
            weights = np.ones(coll_factors.size)
        elif self.algorithm_weights == "area":
            weights = self.key_area(relations)[..., 0]
        else:
            raise NotImplementedError
        return weights * coll_factors

    def compute_relations(self):
        """Generates and returns the :attr:`collision_relations`."""
        print('Generating Collision Array...')
        tic = process_time()
        # collect collisions in the following lists
        relations = []

        """The velocities are named in the following way:
        1. v* and w* are velocities of the first/second specimen, respectively
        2. v0 or w0 denotes the velocity before the collision
           v1 or w1 denotes the velocity after the collision
        """
        # choose function for local collisions
        if self.algorithm_relations == 'all':
            coll_func = bp.Collisions.complete
        elif self.algorithm_relations == 'fast':
            coll_func = bp.Collisions.convergent
        else:
            raise NotImplementedError(
                'Unsupported Selection Scheme: '
                '{}'.format(self.algorithm_relations)
            )

        # Iterate over Specimen pairs
        for s0 in self.species:
            for s1 in np.arange(s0, self.specimen):
                # Todo first generate and store all base_relations
                # Todo then transfer them for all velocities
                # group grid[0] points by distance to grid[2]
                group = self.vGrids[s1].group(self.vGrids[s0].iG)
                for key in group.keys():
                    # only generate colliding velocities(colvels)
                    # for a representative v0 of its group,
                    representant = group[key][0]
                    repr_colvels = coll_func(
                        [self.vGrids[i].extension(2) for i in [s0, s0, s1, s1]],
                        self.masses[[s0, s0, s1, s1]],
                        representant)
                    # Get relations for other class elements by shifting
                    for v0 in group[key]:
                        # shift extended colvels
                        new_colvels = repr_colvels + (v0 - representant)
                        # get indices
                        new_rels = np.zeros(new_colvels.shape[0:2], dtype=int)
                        for i in range(4):
                            s = [s0, s0, s1, s1][i]
                            new_rels[:, i] = self.vGrids[s].get_idx(new_colvels[:, i, :])
                            new_rels[:, i] += self.index_offset[s]

                        # remove out-of-bounds or useless collisions
                        choice = np.where(
                            # must be in the grid
                            np.all(new_rels >= self.index_offset[[s0, s0, s1, s1]], axis=1)
                            # must be effective
                            & (new_rels[..., 0] != new_rels[..., 3])
                            & (new_rels[..., 0] != new_rels[..., 1])
                        )
                        # Add chosen Relations/Weights to the list
                        assert np.array_equal(
                                new_colvels[choice],
                                self.iMG[new_rels[choice]])
                        relations.extend(new_rels[choice])
                    # relations += new_rels
                    # weights += new_weights
        relations = np.array(relations, dtype=int)
        # remove redundant collisions
        relations = self.filter(self.key_index,
                                relations)
        # sort collisions for better comparability
        relations = self.sort(self.key_index,
                              relations)
        toc = process_time()
        print('Time taken =  {t} seconds\n'
              'Total Number of Collisions = {n}\n'
              ''.format(t=round(toc - tic, 3),
                        n=self.size))
        return relations

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
        return Model(**parameters)

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
        attributes = self.attributes() if write_all else self.parameters()
        for attr in attributes:
            value = self.__getattribute__(attr)
            if isinstance(value, csr_matrix):
                value = value.toarray()
            hdf5_group[attr] = value

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
