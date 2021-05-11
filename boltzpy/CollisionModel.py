import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import boltzpy as bp


class CollisionModel(bp.BaseModel):
    r"""Manages the Velocity Grids of all
    :class:`~boltzpy.Species`.

    Parameters
    ----------
    masses : :obj:`~numpy.array` [:obj:`int`]
        Denotes the masses of all specimen.
    shapes : :obj:`~numpy.array` [:obj:`int`]
        Denotes the shape of each :class:`boltzpy.Grid`.
    base_delta : :obj:`float`
        Internal step size (delta) of all :class:`Grids <boltzpy.Grid>`.
        This is NOT the physical distance between grid points.
    spacings : :obj:`~numpy.array` [:obj:`int`]
        Denotes the spacing of each :class:`boltzpy.Grid`.
    """

    def __init__(self,
                 masses,
                 shapes,
                 base_delta=1.0,
                 spacings=None,
                 collision_factors=None,
                 collision_relations=None,
                 collision_weights=None,
                 setup_collision_matrix=True,
                 **kwargs):
        bp.BaseModel.__init__(self,
                              masses,
                              shapes,
                              base_delta,
                              spacings)
        if collision_factors is None:
            self.collision_factors = np.ones((self.nspc, self.nspc), dtype=float)
        else:
            assert isinstance(collision_factors, (list, tuple, np.ndarray))
            self.collision_factors = np.array(collision_factors, dtype=float)
        assert self.collision_factors.ndim == 2

        # setup collisions
        if collision_relations is None:
            self.collision_relations = self.cmp_relations()
        else:
            self.collision_relations = np.array(collision_relations)
        if collision_weights is None:
            self.collision_weights = self.cmp_weights()
        else:
            self.collision_weights = np.array(collision_weights)

        # set up sparse collision_matrix
        self.collision_matrix = csr_matrix((self.nvels, self.collision_weights.size))
        if setup_collision_matrix:
            self.update_collisions(self.collision_relations, self.collision_weights)
        CollisionModel.check_integrity(self)
        return

    @staticmethod
    def parameters():
        params = bp.BaseModel.parameters()
        params.update({"collision_factors",
                       "collision_relations",
                       "collision_weights"})
        return params

    @staticmethod
    def attributes():
        attrs = CollisionModel.parameters()
        attrs.update(bp.BaseModel.attributes())
        attrs.update({"ncols",
                      "collision_invariants"})
        return attrs

    #####################################
    #           Properties              #
    #####################################
    @property
    def ncols(self):
        """:obj:`int` :
        The total number of collision relations."""
        assert self.collision_relations.shape == (self.collision_weights.size, 4)
        return self.collision_weights.size

    @property
    def collision_invariants(self):
        """:obj:`int` :
        The number of collision invariants of this model."""
        rank = np.linalg.matrix_rank(self.collision_matrix.toarray())
        maximum_rank = np.min([self.nvels, self.ncols])
        return maximum_rank - rank

    def is_invariant(self, relations=None, operations=None):
        """:obj:`bool` :
        Returns True if the collisions, more precisely i_vels[relations],
        are invariant under the given operations."""
        # set default parameters
        if relations is None:
            relations = self.collision_relations
        relations = np.array(relations, copy=False, ndmin=2, dtype=int)
        assert relations.shape[1:] == (4,)

        if operations is None:
            # by default check both reflections and permutations
            operations = np.concatenate((self.reflection_matrices,
                                         self.permutation_matrices),
                                        axis=0)
        operations = np.array(operations, copy=False, ndmin=3)
        assert operations.shape[1:] == (self.ndim, self.ndim)
        assert operations.dtype == int

        # group by species, necessary to call get_idx
        key_spc = self.key_species(relations)
        grp_spc = self.group(key_spc, relations)
        for spc, rels in grp_spc.items():
            spc = np.array(spc)
            cols = self.i_vels[rels]
            # apply operations, all simultaneously
            op_cols = np.einsum("abc,dec->adeb", operations, cols)
            # reshape into (N, 4, ndim) shape
            op_cols = op_cols.reshape((-1, 4, self.ndim))
            # get indices
            op_rels = self.get_idx(spc, op_cols)
            # all op_rels must be grid points
            if np.any(op_rels == -1):
                return False
            # filter out redundant collisions
            op_rels = self.filter(self.key_index(op_rels), op_rels)
            # number of relations should not have changed (if invariant)
            if op_rels.shape[0] != rels.shape[0]:
                return False
        return True

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
        (v0, v1, w0, w1) = self.i_vels[relations.transpose()]
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
        (v0, v1, w0, w1) = self.i_vels[relations.transpose()]
        dv = v1 - v0
        # compute gcd of each velocity difference
        gcd = np.gcd.reduce(dv, axis=-1)
        # normalize with gcd to get the direction
        gcd = gcd[..., np.newaxis]
        angles = dv // gcd
        # sort each directions, to merge symmetries
        angles = np.sort(np.abs(angles), axis=-1)
        return angles

    @staticmethod
    def group(group_keys, values=None, as_dict=True, sort_key=None):
        """Create Partitions of positions (indices) with equal keys.

        Parameters
        ----------
        group_keys : :obj:`~numpy.array` [:obj:`int`] or :obj:`tuple’
            If a tuple is given, then the keys are merged (concatenated).
        values : :obj:`~numpy.array` [:obj:`int`] or :obj:`tuple’, optional
            If None, then the grouped positions are returned.
            If not None, the values are grouped and returned.
            relations.shape[0] must match keys.shape[0]
        as_dict : :obj:`bool`, optional
            If False, returns an array of arrays, instead of a dictionary.
        sort_key : :obj:`~numpy.array` [:obj:`int`], optional
            If not None, then equal group_keys are sorted by the sort_key.
            If None, equal group_keys are sorted randomly.
        """
        # merge group_keys
        if type(group_keys) in {tuple, list}:
            # allow 1D keys for concatenation
            group_keys = [key if key.ndim == 2 else key[:, np.newaxis]
                          for key in group_keys]
            group_keys = np.concatenate(group_keys, axis=1)
        assert isinstance(group_keys, np.ndarray)
        assert group_keys.ndim == 2

        # construct sort_key
        if sort_key is None:
            sort_key = group_keys
        else:
            # allow 1D keys for concatenation
            sort_key = sort_key if sort_key.ndim == 2 else sort_key[:, np.newaxis]
            sort_key = np.concatenate((group_keys, sort_key), axis=1)

        # get lexicographic order of sort_key
        positions = np.lexsort(np.flip(sort_key.transpose(), axis=0))
        del sort_key
        # sort group_keys by this order
        group_keys = group_keys[positions]
        # find first occurrence of each key in sorted array
        # pos determines the splitting points below
        key, pos = np.unique(group_keys, axis=0, return_index=True)
        del group_keys

        # define results as a list, for any given values
        if values is None:
            # if no values are given, then return the groupes positions
            results = [positions]
        elif isinstance(values, np.ndarray):
            results = [values[positions]]
        elif type(values) in {tuple, list}:
            results = [val[positions] for val in values]
        else:
            raise TypeError

        # split each result into array slices
        # all segments point to the elements with the same key
        for v, val in enumerate(results):
            # pos[0]==0 must be removed for split() to work correctly
            results[v] = np.split(val, pos[1:])
            # convert to a dictionary, if wanted
            if as_dict:
                results[v] = {tuple(k): val for k, val in zip(key, results[v])}

        # return result or tuple of results
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    @staticmethod
    def filter(keys, values=None):
        """Filter out elements with redundant keys.

        Parameters
        ----------
        keys : :obj:`~numpy.array` [:obj:`int`]
        values : :obj:`~numpy.array` [:obj:`int`] or :obj:`tuple’, optional
            If None, then the filtered positions are returned.
            If not None, the filtered valuesand returned.
            relations.shape[0] must match keys.shape[0]
        """
        assert keys.ndim == 2
        # ignore unique values ([0]), only take first positions ([1]=
        positions = np.unique(keys, return_index=True, axis=0)[1]

        # define results as a list, for any given values
        if values is None:
            # if no values are given, then return the groupes positions
            results = [positions]
        elif isinstance(values, np.ndarray):
            results = [values[positions]]
        elif type(values) in {tuple, list}:
            results = [val[positions] for val in values]
        else:
            raise TypeError

        # return result or tuple of results
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    @staticmethod
    def sort(keys, relations=None):
        """Sort keys (or relations) lexicographically.
        Returns a sorting permutation (if no relations are given)
        or the permuted/sorted relations (if relations are given).

        Parameters
        ----------
        keys : :obj:`~numpy.array` [:obj:`int`]
        relations : :obj:`~numpy.array` [:obj:`int`], optional
            If not None, then the sorted relations at are returned.
            relations.shape[0] must match keys.shape[0]
        """
        assert keys.ndim == 2

        # lexsort sorts columns, thus we transpose first
        # flip keys, as the last key (row) has highest priority
        positions = np.lexsort(np.flip(keys.transpose(), axis=0))
        # return sorted relations, if given
        if relations is not None:
            return relations[positions]
        # otherwise return ordering positions (technically a permutation)
        else:
            return positions

    # todo choose: given list of indices (of collision relations)
    #  or relations(what to do with the weights?) -> merge and filter redundants
    ##################################
    #           Collisions           #
    ##################################
    def update_collisions(self, relations, weights):
        """Updates the collision_relations, collision_weights
        and computes a new sparse collision_matrix.

        Parameters
        ----------
        relations : :obj:`~numpy.array` [:obj:`int`]
            A 2d Array. Each relation (row) has 4 integers,
            that point at velocities of the model.
        weights : :obj:`~numpy.array` [:obj:`float`]
            A 1d array, of numerical weights for each relation.
        """
        if relations.shape != (weights.size, 4) or weights.ndim != 1:
            raise ValueError
        if np.any(np.logical_or(relations < 0, relations >= self.nvels)):
            raise ValueError
        # update attributes
        self.collision_relations = relations
        self.collision_weights = weights

        # Filter out any duplicates, Weights are NOT added
        # This might lead to unpredictable behaviour,
        # if a relation is given twice with different weights
        relations, weights = self.filter(self.key_index(self.collision_relations),
                                         (relations, weights))
        self.collision_relations = relations
        self.collision_weights = weights

        # set up as lil_matrix, allows fast changes to sparse structure
        col_mat = lil_matrix((self.nvels, weights.size), dtype=float)
        _range = np.arange(relations.shape[0])
        for s, sign in enumerate([-1, 1, -1, 1]):
            col_mat[relations[:, s], _range] = weights * sign
        del _range

        # adjust changes to different grid spacings
        # this is necessary for invariance of moments
        # multiply with min(dv) keep the order of magnitude of the weights
        eq_factor = np.min(self.dv) / self.get_array(self.dv)**self.ndim
        for v in range(self.nvels):
            col_mat[v] *= eq_factor[v]

        # convert to csr_matrix, for fast arithmetics
        self.collision_matrix = col_mat.tocsr()
        return

    def cmp_weights(self, relations=None, scheme="uniform"):
        """Generates and returns the :attr:`collision_weights`,
        based on the given relations and :attr:`algorithm_weights`
        """
        if relations is None:
            relations = self.collision_relations
        species = self.key_species(relations)[:, 1:3]
        coll_factors = self.collision_factors[species[..., 0], species[..., 1]]
        if scheme == "uniform":
            return coll_factors
        elif scheme == "area":
            return coll_factors * self.key_area(relations)[..., 0]
        else:
            raise NotImplementedError

    def _get_extended_grids(self, *species):
        """Generate Grids with extended shapes for faster collision generation.

        These Grids must be larger than their "bases",
        to cover all possible shifts of collisions.
        The first Grids shape must double,
        the second grids shape must be large enough to
        cover the largest possible distance between the two grids.

         Parameters
        ----------
        species : :obj:`int`
            Must be 2 specimen.
        """
        species = np.array(species, copy=False, dtype=int)
        assert species.shape == (2,)
        assert np.all(species < self.nspc)

        # get maximum velocity (last velocities of each subgrid) without creating them
        max_vels = self.i_vels[self._idx_offset[species + 1] - 1]
        spacings = self.spacings[species]

        # compute (minimal) shapes for extended grids
        ext_shapes = np.zeros((2, self.ndim), dtype=int)
        # first shape must be doubled
        ext_shapes[0] = 2 * self.shapes[species[0]]
        # second shape must cover the maximal distance between grid points
        max_distance = np.sum(max_vels, axis=0)
        ext_shapes[1] = np.ceil(2 * max_distance / spacings[1] + 1)

        # retain even/odd shapes
        old_shapes = self.shapes[species]
        ext_shapes += (np.array(old_shapes) % 2) != (ext_shapes % 2)
        assert np.array_equal(old_shapes % 2, ext_shapes % 2)

        # generate extended grid, with same parameters but new shape
        ext_grids = [bp.Grid(shape, self.base_delta, spacing, True)
                     for shape, spacing in zip(ext_shapes, spacings)]
        return ext_grids

    # todo speedup idea: just compute largest intraspecies collisions -> reuse for others
    def cmp_relations(self, group_by="partitioned_distance"):
        """Computes the :attr:`collision_relations`.

         Parameters
        ----------
        groupy_by : :obj:`str`
            Switches between algorithms.
            Determines if and how the grids are partitioned into equivalence classes.
        """
        assert group_by in {"distance",
                            "partitioned_distance",
                            "None"}
        # collect collisions in a lists
        relations = []
        # Collisions are computed for every pair of specimen
        species_pairs = np.array([(s0, s1) for s0 in self.species
                                  for s1 in range(s0, self.nspc)])
        # pre initialize each species' velocity grid
        grids = self.subgrids()
        # generate symmetry matrices, for partitioned_distances
        sym_mat = self.symmetry_matrices

        # Compute collisions iteratively, for each pair
        for s0, s1 in species_pairs:
            masses = self.masses[[s0, s1]]

            # partition grid of first specimen into partitions
            # collisions can be shifted (and rotated) in each partition
            # this saves computation time.
            # The collisions must be found in larger grids
            # to find all possible collisions
            # No partitioning at all
            if group_by == "None":
                # imitate grouping, to fit into the remaining algorithm
                grp = grids[s0].iG[:, np.newaxis]
                extended_grids = np.array([grids[s0], grids[s1]], dtype=object)
                # no symmetries are used
                grp_sym = None
            # partition grids[s0] by distance
            # if s0 == s1, this is equivalent to partitioned distance (but faster)
            elif group_by == "distance" or s0 == s1:
                # partition based on distance to next grid point
                grp_keys = grids[s1].key_distance(grids[s0].iG)
                norm = bp.Grid.key_norm(grids[s0].iG)
                grp = self.group(grp_keys, grids[s0].iG, as_dict=False, sort_key=norm)
                del norm, grp_keys
                # no symmetries are used
                grp_sym = None
                # compute representative colliding velocities in extended grids
                extended_grids = self._get_extended_grids(s0, s1)
            # partition grids[s0] by distance and rotation
            else:
                # group based on distances, rotated into 0 <= x <= y <= z
                grp_keys = grids[s1].key_partitioned_distance(grids[s0].iG)
                norm = bp.Grid.key_norm(grids[s0].iG)
                # group both grp and grp_sym in one go (and with same order)
                grp, grp_sym = self.group(grp_keys[..., :-1],
                                          (grids[s0].iG, grp_keys[:, -1]),
                                          as_dict=False,
                                          sort_key=norm)
                # compute representative colliding velocities in extended grids
                extended_grids = self._get_extended_grids(s0, s1)

            # compute collision relations for each partition
            for p, partition in enumerate(grp):
                # choose representative velocity
                repr_vel = partition[0]
                # generate collision velocities for representative
                repr_colvels = self.get_colvels(extended_grids, masses, repr_vel)

                # to reflect / rotate repr_colvels into default symmetry region
                # multiply with transposed matrix
                if grp_sym is not None:
                    repr_colvels = np.einsum("ji, nkj->nki",
                                             sym_mat[grp_sym[p][0]],
                                             repr_colvels - repr_vel)
                # shift to zero for other partition elements
                else:
                    repr_colvels -= repr_vel
                # compute partitions collision relations, based on repr_colvels
                for pos, v0 in enumerate(partition):
                    # shift repr_colvels onto v0
                    if grp_sym is None:
                        new_colvels = repr_colvels + v0
                    # rotate and shift repr_colvels onto v0
                    else:
                        new_colvels = np.einsum("ij, nkj->nki",
                                                sym_mat[grp_sym[p][pos]],
                                                repr_colvels)
                        new_colvels += v0
                    # get indices
                    new_rels = self.get_idx([s0, s0, s1, s1], new_colvels)
                    # remove out-of-bounds or useless collisions
                    choice = np.where(
                        # must be in the grid
                        np.all(new_rels >= 0, axis=1)
                        # must be effective
                        & (new_rels[..., 0] != new_rels[..., 3])
                        & (new_rels[..., 0] != new_rels[..., 1])
                    )
                    # add relations to list
                    relations.extend(new_rels[choice])
        # convert list into array
        relations = np.array(relations, dtype=int)
        # remove redundant collisions
        relations = self.filter(self.key_index(relations), relations)
        # sort collisions for better comparability
        relations = self.sort(self.key_index(relations), relations)
        return relations

    @staticmethod
    def get_colvels(grids,
                    masses,
                    v0):
        # store results in list ov colliding velocities (colvels)
        colvels = []
        for v1 in grids[0].iG:
            dv = v1 - v0
            if np.any((dv * masses[0]) % masses[1] != 0):
                continue
            dw = -(dv * masses[0]) // masses[1]
            # find starting point for w0, projected on axis( v0 -> v1 )
            w0_proj = v0 + dv // 2 - dw // 2
            w0 = grids[1].hyperplane(w0_proj, dv)
            # Calculate w1, using the momentum invariance
            w1 = w0 + dw
            # remove w0/w1 if w1 is out of the grid
            pos = np.where(grids[1].get_idx(w1) >= 0)
            w0 = w0[pos]
            w1 = w1[pos]
            # construct colliding velocities array (colvels)
            local_colvels = np.empty((w0.shape[0], 4, w0.shape[1]), dtype=int)
            local_colvels[:, 0] = v0
            local_colvels[:, 1] = v1
            local_colvels[:, 2] = w0
            local_colvels[:, 3] = w1
            # remove collisions that don't fulfill all conditions
            choice = np.where(CollisionModel.is_collision(local_colvels, masses[[0,0, 1,1]]))
            colvels.append(local_colvels[choice])
        return np.concatenate(colvels, axis=0)

    @staticmethod
    def is_collision(colvels,
                     masses):
        colvels = colvels.reshape((-1, 4, colvels.shape[-1]))
        v0 = colvels[:, 0]
        v1 = colvels[:, 1]
        w0 = colvels[:, 2]
        w1 = colvels[:, 3]
        cond = np.empty((3, colvels.shape[0]))
        # Ignore Collisions without changes in velocities
        cond[0] = np.any(v0 != v1, axis=-1)
        # Invariance of momentum
        cond[1] = np.all(masses[0] * (v1 - v0) == masses[2] * (w0 - w1), axis=-1)
        # Invariance of energy
        cond[2] = np.all(np.sum(masses[0] * (v0 ** 2 - v1 ** 2), axis=-1)
                         == np.sum(masses[2] * (w1 ** 2 - w0 ** 2), axis=-1))
        return np.all(cond, axis=0)

    def collision_operator(self, state):
        """Computes J[f,f],
        with J[f,f] being the collision operator at all given points.
        These points are the ones specified in state.

        Note that this is the collision of all species.
        Collisions of species i with species j are not implemented.
        """
        shape = state.shape
        size = np.prod(shape[:-1], dtype=int)
        state = state.reshape((size, self.nvels))
        assert state.ndim == 2
        result = np.empty(state.shape, dtype=float)
        for p in range(state.shape[0]):
            u_c0 = state[p, self.collision_relations[:, 0]]
            u_c1 = state[p, self.collision_relations[:, 1]]
            u_c2 = state[p, self.collision_relations[:, 2]]
            u_c3 = state[p, self.collision_relations[:, 3]]
            col_factor = (np.multiply(u_c0, u_c2) - np.multiply(u_c1, u_c3))
            result[p] = self.collision_matrix.dot(col_factor)
        return result.reshape(shape)

    #####################################
    #           Coefficients            #
    #####################################
    def cmp_viscosity(self,
                      number_densities,
                      temperature,
                      dt,
                      maxiter=100000,
                      directions=None,
                      animate=False,
                      animate_filename=None):
        # Maxwellian must be centered in 0,
        # since this computation relies heavily on symmetry,
        mean_velocities = np.zeros((self.nspc, self.ndim))
        # all species should have equal temperature, but not a must!
        temperature = np.full(self.nspc, temperature, dtype=float)

        # initialize homogeneous rule, that handles the computation
        rule = bp.HomogeneousRule(number_densities=number_densities,
                                  mean_velocities=mean_velocities,
                                  temperatures=temperature,
                                  **self.__dict__)

        # set up source moment function (stress)
        if directions is None:
            directions = np.eye(self.ndim)[0:2]
        else:
            directions = np.array(directions, dtype=float, copy=False)
            assert directions.shape == (2, self.ndim)
            norm = np.linalg.norm(directions, axis=1).reshape(2, 1)
            directions /= norm

        # use only the first mean_velocity, otherwise a (2, nvels) array is created
        mom_func = self.mf_stress(mean_velocities[0], directions, orthogonalize=True)
        # set up source term
        rule.source_term = mom_func * rule.initial_state
        # check, that source term is orthogonal on all moments
        rule.check_integrity()

        # compute viscosity
        assert dt > 0 and maxiter > 0
        result = rule.compute(dt,
                              maxiter=maxiter,
                              animate=animate,
                              animate_filename=animate_filename)
        # compute viscosity as scalar product
        viscosity = np.sum(result[-1] * mom_func)
        normalize = np.sum(mom_func**2 * rule.initial_state)
        return viscosity / normalize

    def cmp_heat_transfer(self,
                          number_densities,
                          temperature,
                          dt,
                          maxiter=100000,
                          direction=None,
                          animate=False,
                          animate_filename=None):
        # Maxwellian must be centered in 0,
        # since this computation relies heavily on symmetry,
        mean_velocities = np.zeros((self.nspc, self.ndim))
        # all species should have equal temperature, but not a must!
        temperature = np.array(temperature, dtype=float)
        if temperature.size == 1:
            temperature = np.full((self.nspc,), temperature)
        assert temperature.shape == (self.nspc,)
        # initialize homogeneous rule, that handles the computation
        rule = bp.HomogeneousRule(number_densities=number_densities,
                                  mean_velocities=mean_velocities,
                                  temperatures=temperature,
                                  **self.__dict__)

        # set up source moment function (stress)
        if direction is None:
            directions = np.eye(self.ndim)[0]
        else:
            directions = np.array(direction, dtype=float)
            assert directions.shape == (self.ndim,)
            norm = np.linalg.norm(directions)
            directions /= norm
        # use only the first mean_velocity, otherwise a (2, nvels) array is created
        mom_func = self.mf_heat_flow(mean_velocities[0],
                                     directions,
                                     orthogonalize_state=rule.initial_state)
        # set up source term
        rule.source_term = mom_func * rule.initial_state
        # check, that source term is orthogonal on all moments
        rule.check_integrity()

        # compute viscosity
        assert dt > 0 and maxiter > 0
        result = rule.compute(dt,
                              maxiter=maxiter,
                              animate=animate,
                              animate_filename=animate_filename)
        # compute viscosity as scalar product
        heat_transfer = np.sum(result[-1] * mom_func)
        normalize = np.sum(mom_func**2 * rule.initial_state)
        return heat_transfer / normalize

    #####################################
    #           Visualization           #
    #####################################
    #: :obj:`list` [:obj:`dict`]:
    #: Default plot_styles for :meth::`plot`
    plot_styles = [{"marker": 'o', "alpha": 0.5, "s": 50},
                   {"marker": 'x', "alpha": 0.9, "s": 100},
                   {"marker": 's', "alpha": 0.5, "s": 50},
                   {"marker": 'D', "alpha": 0.5, "s": 50}]

    def plot_collisions(self,
                        relations=None,
                        species=None,
                        plot_object=None,
                        **kwargs):
        """Plot the Grid using matplotlib."""
        relations = [] if relations is None else np.array(relations, ndmin=2)
        species = self.species if species is None else np.array(species, ndmin=1)

        # setup ax for plot
        if plot_object is None:
            import matplotlib.pyplot as plt
            projection = "3d" if self.ndim == 3 else None
            ax = plt.figure().add_subplot(projection=projection)
        else:
            ax = plot_object

        # Plot Grids as scatter plot
        for s in species:
            self.subgrids(s).plot(ax, **(self.plot_styles[s]))

        # plot collisions
        for r in relations:
            # repeat the collision to "close" the rectangle / trapezoid
            rels = np.tile(r, 2)
            # transpose the velocities, for easy unpacking
            vels = (self.vels[rels]).transpose()
            ax.plot(*vels, color="gray", linewidth=0.2)

        # set tick values on axes, None = auto choice of matplotlib
        if "xticks" in kwargs.keys():
            ax.set_xticks(kwargs["xticks"])
        if "yticks" in kwargs.keys():
            ax.set_yticks(kwargs["yticks"])
        if "zticks" in kwargs.keys() and self.ndim == 3:
            ax.set_zticks(kwargs["zticks"])
        # when plotting collisions, keep equal aspect ratio of the axes
        if "aspect" in kwargs.keys():
            ax.set_aspect(kwargs['aspect'])
        else:
            ax.set_aspect('equal')

        if plot_object is None:
            # noinspection PyUnboundLocalVariable
            plt.show()
        else:
            return plot_object

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check."""
        bp.BaseModel.check_integrity(self)
        assert np.all(self.spacings % 2 == 0), (
            "For the vectorized collision generation scheme all spacings must even. "
            "It does not return the full set otherwise.\n"
            "Consider doubling the spacing and halving the base_delta.")
        return
