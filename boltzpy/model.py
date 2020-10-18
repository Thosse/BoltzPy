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
    def get_idx(self,
                species,
                velocities):
        """Find index of given grid_entry in :attr:`iMG`
        Returns None, if the value is not in the specified Grid.

        Parameters
        ----------
        species : :obj:`~numpy.array` [:obj:`int`]
        velocities : :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        global_index : :obj:`int` of :obj:`None`

        """
        # reshape arrays for vectorization
        species = np.array(species)
        assert species.ndim in [0, 1], "species must be 0d or 1d array"
        # species must be 1d array
        species = np.array(species, ndmin=1)
        assert velocities.ndim in [1, 2, 3], "velocities must be 1d, 2d, or 3d array"
        # reshape the result/indices at the end
        shape_indices = velocities.shape[:-1]
        # reshape velocities into 3d
        n_vels = velocities.size // (species.size * self.ndim)
        shape_velocities = (n_vels, species.size, self.ndim)
        velocities = velocities.reshape(shape_velocities)

        assert species.ndim == 1 and velocities.ndim == 3
        indices = np.zeros(velocities.shape[0:2], dtype=int)
        for col, s in enumerate(species):
            local_indices = self.vGrids[s].get_idx(velocities[:, col, :])
            indices[..., col] = np.where(local_indices >= 0,
                                         local_indices + self.index_offset[s],
                                         -1)

        # noinspection PyUnreachableCode
        if __debug__:
            pos = np.where(np.all(indices >= 0, axis=-1))
            assert np.all(self.iMG[indices[pos]] == velocities[pos])
        return indices.reshape(shape_indices)

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

    def group(self, relations=None, key_function=None):
        rels = self.collision_relations if relations is None else relations
        assert rels.ndim == 2
        if key_function is None:
            key_function = self.key_species

        grouped = dict()
        keys = key_function(rels)
        unique_keys = np.unique(keys, axis=0)
        for key in unique_keys:
            pos = np.where(np.all(keys == key, axis=-1))
            grouped[tuple(key)] = rels[pos]
        return grouped

    def filter(self, relations=None, key_function=None):
        rels = self.collision_relations if relations is None else relations
        assert rels.ndim == 2
        if key_function is None:
            key_function = self.key_index

        keys = key_function(rels)
        # unique values _ are not needed
        _, positions = np.unique(keys, return_index=True, axis=0)
        if relations is None:
            self.collision_relations = self.collision_relations[positions]
            self.collision_weights = self.collision_weights[positions]
            return
        else:
            return relations[positions]

    def sort(self, relations=None, key_function=None):
        rels = self.collision_relations if relations is None else relations
        assert rels.ndim == 2
        if key_function is None:
            key_function = self.key_index

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
            coll_func = Model.get_colvels
        elif self.algorithm_relations == 'naive':
            coll_func = Model.get_colvels_naive
        elif self.algorithm_relations == 'convergent':
            coll_func = Model.get_colvels_convergent
        else:
            raise NotImplementedError(
                'Unsupported Selection Scheme: '
                '{}'.format(self.algorithm_relations)
            )

        # Iterate over Specimen pairs
        for s0 in self.species:
            for s1 in np.arange(s0, self.specimen):

                # group grid[0] points by distance to grid[2]
                group = bp.Grid.group(self.vGrids[s0].iG,
                                      self.vGrids[s1].key_distance)
                # generate relations for a representative of each group
                repr = {key: group[key][0] for key in group.keys()}
                repr_rels = dict()
                for key in group.keys():
                    # generate collisions in extended grids
                    # allows transfering the results without losing collisions
                    repr_rels[key] = coll_func(
                        [self.vGrids[i].extension(2) for i in [s0, s0, s1, s1]],
                        self.masses[[s0, s0, s1, s1]],
                        repr[key])
                for key, velocities in group.items():
                    # Get relations for other class elements by shifting
                    for v0 in velocities:
                        # shift extended colvels
                        new_colvels = repr_rels[key] + (v0 - repr[key])
                        # get indices
                        new_rels = self.get_idx([s0, s0, s1, s1],
                                                 new_colvels)

                        # remove out-of-bounds or useless collisions
                        choice = np.where(
                            # must be in the grid
                            np.all(new_rels >= 0, axis=1)
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
        relations = self.filter(relations, self.key_index)
        # sort collisions for better comparability
        relations = self.sort(relations, self.key_index)
        toc = process_time()
        print('Time taken =  {t} seconds\n'
              'Total Number of Collisions = {n}\n'
              ''.format(t=round(toc - tic, 3),
                        n=relations.shape[0]))
        return relations

    @staticmethod
    def get_colvels_naive(grids, masses, v0):
        # store results in list ov colliding velocities (colvels)
        colvels = []
        # iterate over all v1 (post collision of v0)
        for v1 in grids[1].iG:
            # ignore v=(a, a, * , *)
            # calculate Velocity (index) difference
            diff_v = v1 - v0
            for w0 in grids[2].iG:
                # Calculate w1, using the momentum invariance
                assert all((diff_v * masses[0]) % masses[2] == 0)
                diff_w = -diff_v * masses[0] // masses[2]
                w1 = w0 + diff_w
                if w1 not in grids[3]:
                    continue
                colvel = np.array([v0, v1, w0, w1], ndmin=3)
                # check if its a proper Collision
                if not Model.is_collision(colvel, masses):
                    continue
                # Collision is accepted -> Add to List
                colvels.append(colvel)
        return np.concatenate(colvels, axis=0)

    @staticmethod
    def get_colvels(grids,
                    masses,
                    v0):
        # store results in list ov colliding velocities (colvels)
        colvels = []
        for v1 in grids[1].iG:
            dv = v1 - v0
            if np.any((dv * masses[0]) % masses[2] != 0):
                continue
            dw = -(dv * masses[0]) // masses[2]
            # find starting point for w0, projected on axis( v0 -> v1 )
            w0_proj = v0 + dv // 2 - dw // 2
            w0 = grids[2].hyperplane(w0_proj, dv)
            # Calculate w1, using the momentum invariance
            w1 = w0 + dw
            # remove w0/w1 if w1 is out of the grid
            pos = np.where(grids[3].get_idx(w1) >= 0)
            w0 = w0[pos]
            w1 = w1[pos]
            # construct colliding velocities array (colvels)
            local_colvels = np.empty((w0.shape[0], 4, w0.shape[1]), dtype=int)
            local_colvels[:, 0] = v0
            local_colvels[:, 1] = v1
            local_colvels[:, 2] = w0
            local_colvels[:, 3] = w1
            # remove collisions that don't fulfill all conditions
            choice = np.where(Model.is_collision(local_colvels, masses))
            colvels.append(local_colvels[choice])
        return np.concatenate(colvels, axis=0)

    @staticmethod
    def get_colvels_convergent(grids, masses, v0):
        # angles = np.array([[1, 0], [1, 1], [0, 1], [-1, 1],
        #                    [-1, 0], [-1, -1], [0, -1], [1, -1]])
        # Todo This is sufficient, until real weights are used
        angles = np.array([[1, -1], [1, 0], [1, 1], [0, 1]])
        # store results in lists
        colvels = []    # colliding velocities
        # iterate over the given angles
        for axis_x in angles:
            # get y axis by rotating x axis 90°
            axis_y = np.array([[0, -1], [1, 0]]) @ axis_x
            assert np.dot(axis_x, axis_y) == 0, (
                "axis_x and axis_y must be orthogonal"
            )
            # choose v1 from the grid points on the x-axis (through v0)
            # just in positive direction because of symmetry and to avoid v1=v0
            for v1 in grids[1].line(v0,
                                    grids[1].spacing * axis_x,
                                    range(1, grids[1].shape[0])):
                diff_v = v1 - v0
                diff_w = diff_v * masses[0] // masses[2]
                # find starting point for w0,
                w0_projected_on_axis_x = v0 + diff_v // 2 + diff_w // 2
                w0_start = next(grids[2].line(w0_projected_on_axis_x,
                                              axis_y,
                                              range(- grids[2].spacing,
                                                    grids[2].spacing)),
                                None)
                if w0_start is None:
                    continue

                # find all other collisions along axis_y
                for w0 in grids[2].line(w0_start,
                                        grids[2].spacing * axis_y,
                                        range(-grids[2].shape[0],
                                              grids[2].shape[0])):
                    w1 = w0 - diff_w
                    # skip, if w1 is not in the grid (can be out of bounds)
                    if np.array(w1) not in grids[3]:
                        continue
                    colvel = np.array([v0, v1, w0, w1], ndmin=3)
                    # check if its a proper Collision
                    if not Model.is_collision(colvel, masses):
                        continue
                    # Collision is accepted -> Add to List
                    colvels.append([v0, v1, w0, w1])
        colvels = np.array(colvels)
        # Todo assert colvels.size != 0
        return colvels

    @staticmethod
    def is_collision(colvels,
                     masses):
        v0 = colvels[:, 0]
        v1 = colvels[:, 1]
        w0 = colvels[:, 2]
        w1 = colvels[:, 3]
        cond = np.empty((3, colvels.shape[0]))
        # Ignore Collisions without changes in velocities
        cond[0] = np.any(np.logical_or(v0 != v1, w0 != w1), axis=-1)
        # Invariance of momentum
        cond[1] = np.all(masses[0] * (v1 - v0) == masses[2] * (w0 - w1), axis=-1)
        # Invariance of energy
        cond[2] = np.all(np.sum(masses[0] * (v0 ** 2 - v1 ** 2), axis=-1)
                         == np.sum(masses[2] * (w1 ** 2 - w0 ** 2), axis=-1))
        return np.all(cond, axis=0)

    #####################################
    #           Visualization           #
    #####################################
    #: :obj:`list` [:obj:`dict`]:
    #: Default plot_styles for :meth::`plot`
    plot_styles = [{"marker": 'o', "alpha": 0.5, "s": 9},
                   {"marker": 'x', "s": 16},
                   {"marker": 's', "alpha": 0.5, "s": 9},
                   {"marker": 'D', "alpha": 0.5, "s": 9}]

    def plot(self, 
             relations=None,
             species=None):
        """Plot the Grid using matplotlib."""
        relations = [] if relations is None else np.array(relations, ndmin=2)
        species = self.species if species is None else np.array(species, ndmin=1)
        # setup ax for plot
        import matplotlib.pyplot as plt
        projection = "3d" if self.ndim == 3 else None
        ax = plt.figure().add_subplot(projection=projection)
        # Plot Grids as scatter plot
        for s in species:
            self.vGrids[s].plot(ax, **(self.plot_styles[s]))
        # plot collisions
        for r in relations:
            # repeat the collision to "close" the rectangle / trapezoid
            rels = np.tile(r, 2)
            # transpose the velocities, for easy unpacking
            vels = (self.iMG[rels] * self.delta).transpose()
            ax.plot(*vels, color="gray", linewidth=1)
        plt.show()
        return

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

        assert self.algorithm_relations in {"all", "convergent", "naive"}
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