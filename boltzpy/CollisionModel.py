import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import boltzpy as bp
import h5py


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
        # collision factors can be 0 or 2 dimensional
        if collision_factors is None:
            self.collision_factors = np.ones((self.nspc, self.nspc), dtype=float)
        else:
            assert isinstance(collision_factors, (list, tuple, np.ndarray))
            self.collision_factors = np.array(collision_factors, dtype=float)
        if self.collision_factors.size == 1:
            self.collision_factors = np.full((self.nspc, self.nspc), self.collision_factors)
        assert self.collision_factors.shape == (self.nspc, self.nspc)

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

    #####################################
    #           Properties              #
    #####################################
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
        are invariant under the given matrix operations."""
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

    def submodel(self, s):
        """Returns a Copy of the current model, that is restricted to a set of specimen.

        Parameters
        ----------
        s : :obj:`int` (or list/array)

        Returns
        -------
        model : :class:`~boltzpy.CollisionModel` or :obj:`~numpy.array` [:class:`~boltzpy.Grid`]
            Collision model restricted to the given species.
        """
        s = np.array(s, ndmin=1, copy=False)
        assert s.shape == (len(set(s)),), "all entries must be unique"

        # reduce collisions, keep only collisions between the given species
        spc = self.key_species(self.collision_relations)
        grp = self.group(spc, as_dict=True)
        chosen_keys = [key for key in grp.keys()
                       if all(k in s for k in key)]
        choice = np.concatenate([grp[key] for key in chosen_keys])
        new_rels = np.array(self.collision_relations[choice], copy=False)
        new_weights = np.array(self.collision_weights[choice], copy=False)

        # undo global indices, obtain local indices
        spc = self.key_species(new_rels)
        new_rels -= self._idx_offset[spc]

        # compute index_offsets in new grids (note: rearrangements are possible)
        new_pos = [list(s).index(i) if i in s else s.size
                   for i in self.species]
        sizes = np.prod(self.shapes[s], axis=-1)
        new_offsets = np.array([np.sum(sizes[:pos]) for pos in new_pos])
        new_rels += new_offsets[spc]

        # construct model
        submodel = CollisionModel(masses=self.masses[s],
                                  shapes=self.shapes[s],
                                  base_delta=self.base_delta,
                                  spacings=self.spacings[s],
                                  collision_factors=self.collision_factors[s][:, s],
                                  collision_relations=new_rels,
                                  collision_weights=new_weights)
        return submodel

    ################################################
    #        Sorting and Ordering Collisions       #
    ################################################
    @staticmethod
    def _nd_lexsort(array, copy=True, axis=-2, partition_position=None):
        # This is an in-place operation!,
        # do no change to parameter objects, if not specifically allowed
        if copy:
            array = np.copy(array)

        # numpys lexsort does not really work in 3d
        assert array.ndim >= 3, "_nd_lexsort is intended for ndim >= 3"
        # instead: create a view as a structured array
        # where the last axis is viewed as (named) tuple
        # and used for lexicographic sorting
        assert array.shape[-1] > 0
        assert axis != -1 and axis != array.ndim - 1
        struc_array = array.view("i8" + (array.shape[-1] - 1) * ", i8")

        # give a priority order for the elements along the last axis
        # ("f0", "f1",...) is the default naming for the named tuple
        order = tuple(["f%i" % i for i in range(array.shape[-1])])
        if partition_position is None:
            # sort struc_array based on the priority order
            # this directly applies to the original array!
            struc_array.sort(order=order, axis=axis)
        else:
            assert isinstance(partition_position, int)
            assert abs(partition_position) < array.shape[axis] - 1
            # quicker version of sort, after which
            # only the partition_position-th element is at the right position
            struc_array.partition(partition_position,
                                  order=order, axis=axis)
        if copy is True:
            return array
        else:
            return

    @staticmethod
    def key_index(relations):
        """"Sorts a set of relations,
        such that each relations indices are in ascending"""
        return np.sort(relations, axis=-1)

    def key_species(self, relations):
        """"Determines the species involved in each collision.
        Returns the sorted species."""
        species = self.get_spc(relations)
        return np.sort(species, axis=-1)

    def key_shape(self, relations, use_norm=False):
        """"Describes the geometric shape of the collision trapezoid.

        Keep squares of width an height, as the root is slow and no integer."""
        # 3D models require two angles: length and height
        # in 2D models both angles are always equal
        # thus the second angle is omitted

        keys = np.empty((relations.shape[0], 2, self.ndim),
                        dtype=int)

        # determine the width vectors of the trapezoid
        # compute both options and store temporarily in keys
        colvels = self.i_vels[relations]
        keys[:, 0] = colvels[:, 1] - colvels[:, 0]     # = v_1 - v_0
        keys[:, 1] = colvels[:, 3] - colvels[:, 2]     # = v_3 - v_2
        # compute norm of width vectors
        norm = np.sum(keys ** 2, axis=-1)
        switch = (norm[:, 0] > norm[:, 1])
        # pick width vector of smaller length as first component
        keys[:, 0, :] = np.where(switch[:, None],
                                 keys[:, 1],
                                 keys[:, 0])

        # compute height vector by adding vector segments
        height_weights = np.array([-0.5, -0.5, 0.5, 0.5])[None, :, None]
        heights = np.sum(height_weights * colvels, axis=1)
        keys[:, 1, :] = heights

        # take sorted absolutes of length and height
        # to merge all grid symmetries
        keys = np.sort(np.abs(keys), axis=-1)

        # lexicographically sort length and height for intraspecies collisions
        # select keys of intraspecies collisions
        key_spc = self.key_species(relations)[:, 1:3]
        is_intra = key_spc[:, 0] == key_spc[:, 1]
        intra_angles = keys[is_intra]     # Do not remove!
        del key_spc
        # sort angles in intraspecies keys
        self._nd_lexsort(intra_angles, axis=1, copy=False)
        keys[is_intra] = intra_angles

        if use_norm:
            # ignore orientation information, and replace vecors by their norms
            # do NOT take square root, to keep integer values (it is also faster)
            keys = np.sum(keys ** 2, axis=-1)
        else:
            # concatenate sorted length and height vectors in keys
            keys.resize((relations.shape[0], np.prod(keys.shape[1:])))
        return keys

    def key_center_of_gravity(self, relations, use_norm=False):
        """"Computes the center of gravity, except for the division by the masses.
        For a key function and fix masses it is equivalent to the center of gravity.
        """
        # get colvels for center of gravity
        colvels = self.i_vels[relations[:, ::2]]
        # get masses for center of gravity
        masses = self.get_array(self.masses)[relations[:, ::2]]
        # compute center of gravity
        cog = np.sum(masses[..., None] * colvels,
                     axis=1)
        if use_norm:
            return np.sum(cog**2, axis=1)
        else:
            return np.sort(np.abs(cog), axis=1)

    def key_orbit(self, relations, reduce=True):
        """"Determines an unique id for the collisions orbit.

        The algorithm
            1. compute the orbits of all collisions as colliding velocities
            2. get indices of the colliding velocities, specieswise
            3. sort orbit relations, for permutation invariance
        """
        # currently, this algorithm only works in square/cubic grids
        assert self.is_cubic_grid

        # 1. compute the orbits of all collisions as colliding velocities
        # get colliding velocities from indices
        colvels = self.i_vels[relations]
        # get all elements of the orbit
        orbvels = np.einsum("sij, ckj -> cski",
                            self.symmetry_matrices,
                            colvels)
        del colvels

        # 2. get indices of all orbit collisions, specieswise
        # group colvels by species, entries are indices of relations
        grp = self.group(self.key_species(relations))
        # get relations from colliding velocities
        keys = np.empty(orbvels.shape[:3], dtype=int)
        for spc, idx in grp.items():
            keys[idx] = self.get_idx(spc, orbvels[idx])
        del orbvels

        # 3. sort orbkeys, for permutation invariance
        # sort indices in each relation of every orbit
        keys.sort(axis=2)
        # sort the orbit elements of each original collision (axis=1)
        pos = 0 if reduce else None
        self._nd_lexsort(keys, axis=1, copy=False, partition_position=pos)
        if reduce:
            return keys[:, 0, ...]
        else:
            return keys

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
        # 3D models require two angles: length and height
        # in 2D models both angles are always equal
        # thus the second angle is omitted
        key_components = 1 if self.ndim == 2 else 2
        keys = np.empty((relations.shape[0], key_components, self.ndim),
                        dtype=int)

        # determine the width of the trapezoid
        colvels = self.i_vels[relations]
        widths = colvels[:, 1] - colvels[:, 0]     # = v_1 - v_0
        keys[:, 0, :] = widths

        # 3D models additionally require the height vector
        if self.ndim != 2:
            # compute height vector by adding vector segments
            height_weights = np.array([-0.5, -0.5, 0.5, 0.5])[None, :, None]
            heights = np.sum(height_weights * colvels, axis=1)
            keys[:, 1, :] = heights

        # compute gcd of length and height
        gcd = np.gcd.reduce(keys, axis=-1)
        # gcd can be zero (height == 0), in this case divide by 1
        gcd[gcd == 0] = 1
        # normalize with gcd to get the direction
        keys = keys // gcd[..., np.newaxis]

        # take sorted absolutes of length and height
        # to merge all grid symmetries
        keys = np.sort(np.abs(keys), axis=-1)

        # lexicographically sort length and height for intraspecies collisions
        if self.ndim != 2:
            # select keys of intraspecies collisions
            key_spc = self.key_species(relations)[:, 1:3]
            is_intra = key_spc[:, 0] == key_spc[:, 1]
            intra_angles = keys[is_intra]     # Do not remove!
            del key_spc
            # sort angles in intraspecies keys
            self._nd_lexsort(intra_angles, axis=1, copy=False)
            keys[is_intra] = intra_angles

        # concatenate sorted length and height vectors in keys
        keys.resize((relations.shape[0], np.prod(keys.shape[1:])))
        return keys

    def key_energy_transfer(self, relations, as_bool=True):
        assert relations.ndim == 2
        # find interspecies collisions
        key_spc = self.key_species(relations)[:, 1:3]
        is_inter_col = key_spc[:, 0] != key_spc[:, 1]
        inter_rels = relations[is_inter_col]
        # define energy transfer as zero for intraspecies collisions
        energy_transfer = np.zeros(relations.shape[0], dtype=float)
        # compute transfer for interspecies collisions
        (v0, v1) = self.i_vels[inter_rels[:, :2].transpose()]
        energy_transfer[is_inter_col] = np.abs(np.sum(v0**2 - v1**2, axis=-1))
        if as_bool:
            return energy_transfer > 0
        else:
            energy_transfer[is_inter_col] = (
                self.masses[key_spc[is_inter_col, 0]]
                * np.sqrt(energy_transfer[is_inter_col])
            )
            return energy_transfer

    def key_is_normality_collision(self, relations, group_by="shape"):
        """Returns 1, for all collisions required for normality (based on shape).

        The collisions required for normality are
            - the smallest collisions along the grid axes
            - the smalles collisions along the 2-axis-diagonals

        Parameters
        ----------
        relations : :obj:`~numpy.array` [:obj:`int`]
        group_by : :obj:`str`, optional
            Allows to mark more general groups, like "angles"
            as required for normality.
        """
        assert relations.ndim == 2
        is_normality_collision = np.zeros((relations.shape[0], 1),
                                          dtype=int)
        species = self.key_species(relations)[:, 1:3]
        shapes = self.key_shape(relations)
        grp = self.group((species, shapes))
        # find normality collisions based on species and shape
        for key, positions in grp.items():
            key = np.array(key, dtype=int)
            # only consider intraspecies collisions
            if key[0] != key[1]:
                continue
            # only consider quadratic collisions
            length = key[2: 2 + self.ndim]
            width = key[2 + self.ndim: 2 + 2*self.ndim]
            if np.any(length != width):
                continue
            # only consider smallest possible squares
            # parallel to grid axes or along 2-axis-diagonals
            if group_by in ["shape", "orbit"]:
                spacing = self.spacings[key[0]]
            elif group_by == "angle":
                spacing = 1
            else:
                raise ValueError
            parallel = np.zeros(self.ndim, dtype=int)
            parallel[-1] = spacing
            diagonal = np.zeros(self.ndim, dtype=int)
            diagonal[-2:] = spacing
            if np.all(length == parallel) or np.all(length == diagonal):
                is_normality_collision[positions] = 1
            else:
                continue
        return is_normality_collision

    @staticmethod
    def merge_keys(*keys):
        """Merge multiple key arrays by elementwise concatenation

        Parameters
        ----------
        keys : :obj:`tuple`[ :obj:`~numpy.array` [:obj:`int`] ]
            If no values are given, then group the position indices.
            Otherwise all values are simultaneously grouped
            and returned as a tuple.
            Every values.shape[0] must match keys.shape[0]

        """
        assert len(keys) >= 1, "At least as single key must be given"
        # allow 1D keys for concatenation
        keys = [key if key.ndim == 2
                else key[:, np.newaxis]
                for key in keys]
        assert all(key.ndim == 2 for key in keys)
        assert all(key.shape[0] == keys[0].shape[0] for key in keys)
        # merge keys, by elementwise concatenation
        merged_keys = np.concatenate(keys, axis=1)
        assert isinstance(merged_keys, np.ndarray)
        assert merged_keys.ndim == 2, "merged keys must be a 2D array"
        return merged_keys

    @staticmethod
    def group(group_keys, *values, as_dict=True, sort_key=None):
        """Create Partitions of values (or indices) with equal keys.

        This method collects all values, that correspond to an equal key.
        Mathematically, this is similar to taking the quotient space.

        Parameters
        ----------
        group_keys : :obj:`~numpy.array` [:obj:`int`], :obj:`tuple`, or :obj:`list`
            If a tuple or list is given, then the keys are merged (concatenated).
        values : :obj:`list`[ :obj:`~numpy.array` [:obj:`int`] ], optional
            If no values are given, then group the position indices.
            Otherwise all values are simultaneously grouped
            and returned as a tuple.
            Every values.shape[0] must match keys.shape[0]
        as_dict : :obj:`bool`, optional
            If False, returns an array of arrays.
            Otherwise, by default, a dictionary.
        sort_key : :obj:`~numpy.array` [:obj:`int`], optional
            Enforces a specific order between elements in the same group.
            If not None, then equal group_keys are sorted by the sort_key.
            Otherwise, equal group_keys are sorted randomly.
        """
        # concatenate multiple group_keys into a single key_array
        if type(group_keys) in {tuple, list}:
            group_keys = CollisionModel.merge_keys(*group_keys)

        # obtain sorting order
        sort_key = (group_keys if sort_key is None
                    else CollisionModel.merge_keys(group_keys, sort_key))
        positions = CollisionModel.sort(sort_key)
        del sort_key

        # apply order to group_keys and values
        group_keys = group_keys[positions]
        if len(values) == 0:
            values = [positions]
        else:
            # convert the values to arrays, if necessary
            values = [np.array(val, copy=False) for val in values]
            # all values must  have a matching number of elements
            assert all(val.shape[0] == group_keys.shape[0]
                       for val in values)
            # apply reordering
            values = [val[positions] for val in values]
        del positions

        # find first occurrence of each key in sorted array
        # pos determines the splitting points below
        key, pos = np.unique(group_keys, axis=0, return_index=True)
        del group_keys

        # split each result into array slices
        # all segments point to the elements with the same key
        for v, val in enumerate(values):
            # pos[0]==0 must be removed for split() to work correctly
            # otherwise it creates an empty array as first entry
            values[v] = np.split(val, pos[1:])
            # convert to a dictionary, if wanted
            if as_dict:
                values[v] = {tuple(k): partition
                             for k, partition in zip(key, values[v])}

        # return result or tuple of results
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    @staticmethod
    def filter(keys, *values):
        """Filter out elements with redundant keys.

        Parameters
        ----------
        keys : :obj:`~numpy.array` [:obj:`int`]
        values : :obj:`list`[ :obj:`~numpy.array` [:obj:`int`] ], optional
            If no values are given, then filter the position indices.
            Otherwise all values are simultaneously filtered
            and returned as a tuple.
            Every values.shape[0] must match keys.shape[0]
        """
        assert keys.ndim == 2, "keys must be a 2D array"
        # using return_index=True returns a tuple of
        #   [0] the unique values (keys)
        #   [1] the position of first occurrence
        # we only require the position here
        positions = np.unique(keys, return_index=True, axis=0)[1]

        # return filtered positions or values
        if len(values) == 0:
            return positions
        elif len(values) == 1:
            return values[0][positions]
        else:
            # convert the values to arrays, if necessary
            values = [np.array(val, copy=False) for val in values]
            # all values must  have a matching number of elements
            assert all(val.shape[0] == keys.shape[0]
                       for val in values)
            return tuple([val[positions] for val in values])

    @staticmethod
    def sort(keys, *values):
        """Sort keys (or values based on keys) lexicographically.

        Returns sorted indices (if no values are given)
        or the sorted values (if values are given).

        Parameters
        ----------
        keys : :obj:`~numpy.array` [:obj:`int`]
        values : :obj:`list`[ :obj:`~numpy.array` [:obj:`int`] ], optional
            If no values are given, then sort the positions indices.
            Otherwise all values are simultaneously sorted
            and returned as a tuple.
            Every values.shape[0] must match keys.shape[0]
        """
        assert keys.ndim == 2, "keys must be a 2D array"
        # lexsort sorts the columns, not the rows
        # thus we transpose first, to sort the rows
        keys = keys.transpose()
        # lexsort assigns the last key element (row) the highest priority
        # thus we flip the keys
        # Since we transposed before, we flip along axis 0
        keys = np.flip(keys, axis=0)
        positions = np.lexsort(keys)

        # return sorted values, if given
        if len(values) == 0:
            return positions
        elif len(values) == 1:
            return values[0][positions]
        else:
            # convert the values to arrays, if necessary
            values = [np.array(val, copy=False) for val in values]
            # all values must  have a matching number of elements
            assert all(val.shape[0] == keys.shape[0]
                       for val in values)
            return tuple([val[positions] for val in values])

    # todo: save collision arrays directly to a file,
    #   then only store used_collision_relations in the ram
    #   this reduces memory demands and allows safe weight adjustments
    #   without removing any (currently) unused collisions
    #   It also allows to compute the collisions in chunks,
    #   avoiding the out of memory errors in large grids
    ##################################
    #           Collisions           #
    ##################################
    def update_collisions(self,
                          relations=None,
                          weights=None):
        """Updates the collision_relations, collision_weights
        and computes a new sparse collision_matrix.

        Parameters
        ----------
        relations : :obj:`~numpy.array` [:obj:`int`], optional
            A 2d Array. Each relation (row) has 4 integers,
            that point at velocities of the model.
        weights : :obj:`~numpy.array` [:obj:`float`], optional
            A 1d array, of numerical weights for each relation.
        """
        if relations is None:
            relations = self.collision_relations
        if weights is None:
            weights = self.collision_weights
        if weights.ndim != 1:
            raise ValueError
        if relations.shape != (weights.size, 4):
            raise ValueError
        if not relations.dtype == int:
            raise TypeError
        if np.any((relations < 0) | (relations >= self.nvels)):
            raise ValueError

        # update attributes
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

    def remove_collisions(self, indices=None,
                          probability=None,
                          key_function=None,
                          update_collision_matrix=True,
                          only_set_weight_to_zero=False):
        """Remove collisions either by index or a probability array.

        Parameters
        ----------
        indices : :obj:`~numpy.array` [:obj:`int`], optional
        probability : :obj:`~numpy.array` [:obj:`int`], optional
            Entries must be non negative integers.
            Values do **not** need to sum up to 1.
        key_function : :obj:`function`, optional
            Used to partition the collisions into groups.
            Instead of removeing a single emelents,
            its whole group is removed.
            By default :meth:`key_index` is used.
        update_collision_matrix : :obj:`bool`, optional
            If True (default), then the collision matrix
            is updated after removal. Otherwise not.
        only_set_weight_to_zero : :obj:`bool`, optional
            If False (default), both then remove from both
             the collision relations and weights arrays.
        """
        # remove either by index or probability
        assert (indices is None) != (probability is None)
        # negative collision weights are used to mark collision removal
        assert np.all(self.collision_weights >= 0)
        # if specified find a single collision based on probability
        if probability is not None:
            assert probability.shape == (self.ncols,)
            assert probability.dtype == float
            assert np.all(probability >= 0)
            # enforce a probability distribution, with altering the original
            sum = probability.sum()
            if not np.isclose(sum, 1):
                probability = probability / probability.sum()
            # select a random collision, by it position
            indices = np.random.choice(self.ncols, p=probability)
            indices = np.array([indices])

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == int
        assert np.all((0 <= indices) & (indices < self.ncols))
        # key_function is used to remove whole collision groups
        if key_function is None:
            key_function = self.key_index
        assert callable(key_function)
        grp = self.group(key_function(self.collision_relations))
        # mark specified collision groups for removal
        for idx in indices:
            col = self.collision_relations[idx]
            key = key_function(col.reshape((1, 4)))
            key = tuple(key.reshape(4))
            positions = grp[key]
            # mark removal by a negative collision weight
            self.collision_weights[positions] = -np.inf

        # process markd collisions
        if only_set_weight_to_zero:     # Set weights of marked collisions to 0
            self.collision_weights = np.where(self.collision_weights >= 0,
                                              self.collision_weights,
                                              0)
        else:                           # remove marked collisions
            keep = np.where(self.collision_weights >= 0)[0]
            self.collision_relations = self.collision_relations[keep]
            self.collision_weights = self.collision_weights[keep]
        if update_collision_matrix:
            self.update_collisions()
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

    def _get_extended_grids(self, species, groups=None):
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
        groups : :obj:`list`
            The partitions of the first velocity grid.
        """
        species = np.array(species, copy=False, dtype=int)
        assert species.shape == (2,)
        assert np.all(species < self.nspc)
        if isinstance(groups, dict):
            groups = list(groups.values())

        # get maximum velocity (last velocities of each subgrid) without calling subgrids
        max_vels = self.max_i_vels[species]
        spacings = self.spacings[species]

        # to find all possible collisions with this scheme,
        # the extended  grids must cover the maximum range of the original grids,
        # starting from each representative.
        # Find max values of representants, otherwise assume the worst (max_vel)
        if groups is not None:
            max_grp = np.max(np.abs([g[0] for g in groups]))
        else:
            max_grp = max_vels[0]
        # necessary max velocities of extended grids
        ext_max_vel = max_grp[None] + max_vels[None, 0] + max_vels
        # compute (minimal) shapes for extended grids (might need a +1)
        ext_shapes = np.ceil((2 * ext_max_vel) / spacings[:, None])
        # np.ceil returns integer valued floats
        ext_shapes = np.array(ext_shapes, dtype=int)
        # retain even/odd shapes
        old_shapes = self.shapes[species]
        ext_shapes += (np.array(old_shapes) % 2) != (ext_shapes % 2)
        assert np.array_equal(old_shapes % 2, ext_shapes % 2)

        # generate extended grid, with same parameters but new shape
        ext_grids = [bp.Grid(shape, self.base_delta, spacing, True)
                     for shape, spacing in zip(ext_shapes, spacings)]
        return ext_grids

    # todo speedup idea: just compute largest intraspecies collisions -> reuse for others
    def cmp_relations(self, group_by=None):
        """Computes the :attr:`collision_relations`.

         Parameters
        ----------
        groupy_by : :obj:`str`
            Switches between algorithms.
            Determines if and how the grids are partitioned into equivalence classes.
        """
        if group_by is not None:
            assert group_by in {"distance",
                                "sorted_distance",
                                "None",
                                "norm_and_sorted_distance"}
        elif self.is_cubic_grid:
            group_by = "norm_and_sorted_distance"
        else:
            group_by = "None"
        # grouping algorithms require cubic grids
        if group_by in {"sorted_distance", "norm_and_sorted_distance"}:
            assert self.is_cubic_grid

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
                # no symmetries are used
                grp_sym = None
                # no extended grids necessary
                extended_grids = np.array([grids[s0], grids[s1]], dtype=object)
                # no cutoff with max_distances necessary
                max_distance = None
            elif group_by == "norm_and_sorted_distance":
                # group based on sorted velocity components and sorted distance
                # sort_dist = grids[s1].key_sorted_distance(grids[s0].iG)[..., :-1]
                sort_vel = np.sort(np.abs(grids[s0].iG), axis=-1)
                sym_vel = grids[s1].key_symmetry_group(grids[s0].iG)
                # group both grp and grp_sym in one go (and with same order)
                grp, grp_sym = self.group(sort_vel,
                                          grids[s0].iG,
                                          sym_vel,
                                          as_dict=False)
                del sort_vel, sym_vel
                # no extended grids necessary
                extended_grids = np.array([grids[s0], grids[s1]], dtype=object)
                # no cutoff with max_distances necessary
                max_distance = None
            # partition grids[s0] by distance
            # if s0 == s1, this is equivalent to partitioned distance (but faster)
            elif group_by == "distance" or s0 == s1:
                # partition based on distance to next grid point
                dist = grids[s1].key_distance(grids[s0].iG)
                norm = bp.Grid.key_norm(grids[s0].iG)
                grp = self.group(dist, grids[s0].iG, as_dict=False, sort_key=norm)
                del norm, dist
                # no symmetries are used
                grp_sym = None
                # compute representative colliding velocities in extended grids
                extended_grids = self._get_extended_grids((s0, s1), grp)
                # cutoff unnecessary velocity with max_distances,
                # when computing the reference colvels
                max_distance = self.max_i_vels[None, s0] + self.max_i_vels[[s0, s1]]
            # partition grids[s0] by distance and rotation
            elif group_by == "sorted_distance":
                # group based on distances, rotated into 0 <= x <= y <= z
                sort_dist = grids[s1].key_sorted_distance(grids[s0].iG)
                norm = bp.Grid.key_norm(grids[s0].iG)
                # group both grp and grp_sym in one go (and with same order)
                grp, grp_sym = self.group(sort_dist[..., :-1],
                                          grids[s0].iG,
                                          sort_dist[:, -1],
                                          as_dict=False, sort_key=norm)
                del norm, sort_dist
                # compute representative colliding velocities in extended grids
                extended_grids = self._get_extended_grids((s0, s1), grp)
                # cutoff unnecessary velocity with max_distances,
                # when computing the reference colvels
                max_distance = self.max_i_vels[None, s0] + self.max_i_vels[[s0, s1]]
            else:
                raise ValueError

            # compute collision relations for each partition
            for p, partition in enumerate(grp):
                # choose representative velocity
                repr_vel = partition[0]
                # generate collision velocities for representative
                repr_colvels = self.get_colvels([s0, s1],
                                                repr_vel,
                                                extended_grids,
                                                max_distance)
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

    def get_colvels(self, species, v0, grids, max_dist=None):
        if max_dist is None:
            values = [G.iG for G in grids]
        else:
            values = []
            for i in [0, 1]:
                pos = np.where(np.all(np.abs(grids[i].iG - v0) <= max_dist[i], axis=-1))
                values.append(grids[i].iG[pos])
        masses = self.masses[species]
        spacings = self.spacings[species]
        # store results in list ov colliding velocities (colvels)
        colvels = []

        # pre filter useless elements of values[0]
        dv = values[0] - v0
        dw = -(dv * masses[0]) // masses[1]
        useful = np.where(np.all((dv * masses[0]) % (2 * masses[1]) == 0, axis=-1)
                          & np.all(dw % spacings[1] == 0, axis=-1))
        values[0] = values[0][useful]
        del dv, dw

        # find colvels in prefiltered values
        for v1 in values[0]:
            dv = v1 - v0
            dw = -(dv * masses[0]) // masses[1]
            # find starting point for w0, projected on axis( v0 -> v1 )
            w0_proj = v0 + dv // 2 - dw // 2
            w0 = grids[1].hyperplane(w0_proj, dv, values[1])
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
            choice = np.where(CollisionModel.is_collision(local_colvels,
                                                          masses[[0, 0, 1, 1]]))
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
                      normalize=True,
                      hdf5_group=None):
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

        # use only the first mean_velocity, otherwise a (2, nvels) array is created
        mom_func = self.mf_stress(mean_velocities[0], directions, orthogonalize=True)
        # set up source term
        rule.source_term = mom_func * rule.initial_state
        # check, that source term is orthogonal on all moments
        rule.check_integrity()
        # compute viscosity
        assert dt > 0 and maxiter > 0
        inverse_source_term = rule.compute(dt, maxiter=maxiter, hdf5_group=hdf5_group)
        viscosity = np.sum(inverse_source_term[-1] * mom_func)

        if normalize:
            viscosity = viscosity / np.sum(mom_func**2 * rule.initial_state)
        return viscosity

    def cmp_heat_transfer(self,
                          number_densities,
                          temperature,
                          dt,
                          maxiter=100000,
                          direction=None,
                          normalize=True):
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

        # use only the first mean_velocity, otherwise a (2, nvels) array is created
        mom_func = self.mf_heat_flow(mean_velocities[0],
                                     direction,
                                     orthogonalize_state=rule.initial_state)
        # set up source term
        rule.source_term = mom_func * rule.initial_state
        # check, that source term is orthogonal on all moments
        rule.check_integrity()
        # compute heat transfer
        assert dt > 0 and maxiter > 0
        inverse_source_term = rule.compute(dt, maxiter=maxiter)
        heat_transfer = np.sum(inverse_source_term[-1] * mom_func)

        if normalize:
            heat_transfer = heat_transfer / np.sum(mom_func**2 * rule.initial_state)
        return heat_transfer

    #####################################
    #           Visualization           #
    #####################################
    #: :obj:`list` [:obj:`dict`]:
    #: Default plot_styles for :meth::`plot`
    plot_styles = [{"marker": 'o', "alpha": 0.5, "s": 50},
                   {"marker": 'x', "alpha": 0.9, "s": 50},
                   {"marker": 's', "alpha": 0.5, "s": 50},
                   {"marker": 'D', "alpha": 0.5, "s": 50}]

    def plot_collisions(self,
                        relations=None,
                        species=None,
                        plot_object=None,
                        lw=0.2,
                        color="gray",
                        zorder=4,
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
            ax.plot(*vels,
                    color=color,
                    linewidth=lw,
                    zorder=zorder)

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
        elif self.ndim == 3:    # Axed3D does not support equal aspect ratio
            ax.set_aspect('auto')
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
