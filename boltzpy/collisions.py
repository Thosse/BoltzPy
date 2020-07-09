
import numpy as np
from scipy.sparse import csr_matrix
from time import time
import h5py

import boltzpy as bp


class Collision(bp.BaseClass):
    r"""Encapsulates the :attr:`relation` and :attr:`weight`
    of any single Collision.

    Attributes
    ----------
    relation : :obj:`~numpy.array` [:obj:`int`]
        Contains the indices of the colliding velocities.
        Any relation consists of 4 indices in the following order
        :math:`\left[ v_0, v_1, w_0, w_1\right]`,
        where  :math:`v_0, w_0` index the pre
        and :math:`v_1, w_1` index the post collision velocities.
    weight : :obj:`float`
        Contains the numeric integration weights
        of the respective collision in :attr:`relations`.
    """
    def __init__(self, relation, weight=1.0):
        if isinstance(relation, list):
            relation = np.array(relation, dtype=int)
        assert isinstance(relation, np.ndarray)
        assert relation.dtype == int
        assert relation.size == 4
        self.relation = relation
        self.weight = float(weight)

    #####################################
    #           Visualization           #
    #####################################
    def plot(self,
             model,
             plot_object=None):
        indices = list(self.relation) + [self.relation[0]]
        quadrangle = model.iMG[indices] * model.delta
        x_vals = quadrangle[..., 0]
        y_vals = quadrangle[..., 1]
        plot_object.plot(x_vals, y_vals, c="gray")
        return plot_object


# Todo move this class into model(velocity_grid, collisions)
class Collisions(bp.BaseClass):
    r"""Generates and encapsulates the collision :attr:`relations`
    and :attr:`weights`.

    .. todo::
        - check integrity (non neg weights,
          no multiple occurrences, physical correctness)
        - **Add Stefan's Generation-Scheme**
        - can both the transport and the collisions
          be implemented as interpolations? -> GPU Speed-UP
        - @generate: replace for loops by numpy.apply_along_axis
          (this probably needs several additional functions).

    Attributes
    ----------
    relations : :obj:`~numpy.array` [:obj:`int`]
        Contains the active collisions.
        Each collision is a 4-tuple of indices in :attr:`model.iMG`
        and is in the form
        :math:`\left[ v_0, v_1, w_0, w_1\right]`,
        where  :math:`v_0, w_0` are the pre
        and :math:`v_1, w_1` are the post collision velocities.
    weights : :obj:`~numpy.array` [:obj:`float`]
        Contains the numeric integration weights
        of the respective collision in :attr:`relations`.
    """
    def __init__(self, relations=None, weights=None):
        if relations is None:
            self.relations = np.empty((0, 4), dtype=int)
        else:
            self.relations = np.array(relations, dtype=int)
        if weights is None:
            self.weights = np.empty((0,), dtype=float)
        else:
            self.weights = np.array(weights, dtype=float)
        assert self.relations.shape[0] == self.weights.size
        assert self.relations.shape[1] == 4
        assert self.relations.ndim == 2
        return

    # Any single relation can be permutated without changing its effect
    INTRASPECIES_PERMUTATION = np.array([[0, 1, 2, 3], [1, 2, 3, 0],
                                         [2, 3, 0, 1], [3, 0, 1, 2],
                                         [3, 2, 1, 0], [0, 3, 2, 1],
                                         [1, 0, 3, 2], [2, 1, 0, 3]],
                                        dtype=int)

    INTERSPECIES_PERMUTATION = np.array([[0, 1, 2, 3],
                                         [2, 3, 0, 1],
                                         [3, 2, 1, 0],
                                         [1, 0, 3, 2]],
                                        dtype=int)

    @property
    def list(self):
        return ([*self.relations[i], self.weights[i]] for i in range(self.size))

    @property
    def size(self):
        """:obj:`int` : Total number of active collisions."""
        assert self.relations.shape[0] == self.weights.size
        assert self.relations.shape[1] == 4
        return self.relations.shape[0]

    # Todo Remove this, should be deprecated
    @property
    def is_set_up(self):
        """:obj:`bool` :
        True, if the instance is completely set up
        and ready to call :meth:`~Simulation.run_computation`.
        False Otherwise.
        """
        return self.size > 0

    def issubset(self, other):
        """Checks if self.relations are a subset of other.relations.
        Weights are not checked and may differ.

        Parameters
        ----------
        other : :class:`Collisions`

        Returns
        -------
        :obj:`bool`
        """
        assert isinstance(other, Collisions)
        # group/filter by index
        grp_self = self.group(mode="index")
        grp_other = other.group(mode="index")
        # assert that each keys has only a single value
        assert all([len(value) == 1 for value in grp_self.values()])
        assert all([len(value) == 1 for value in grp_other.values()])
        # use set of keys to check for subset relationship
        set_self = set(grp_self.keys())
        set_other = set(grp_other.keys())
        return set_self.issubset(set_other)

    #####################################
    #        Sorting and Ordering       #
    #####################################
    @staticmethod
    def key_index(collision_tuple):
        sorted_indices = np.sort(collision_tuple[0:4])
        return tuple(sorted_indices)

    @staticmethod
    def key_species(collision_tuple, model):
        indices = [model.get_specimen(col_idx)
                   for col_idx in collision_tuple[0:4]]
        return tuple(sorted(indices))

    @staticmethod
    def key_area(collision_tuple, model):
        [v0, v1, w0, w1] = model.iMG[collision_tuple[0:4]]
        area_1 = np.linalg.norm(np.cross(v1 - v0, w1 - v0))
        area_2 = np.linalg.norm(np.cross(w1 - w0, w1 - v0))
        area = 0.5 * (area_1 + area_2)
        circumference = (np.linalg.norm(v1 - v0)
                         + np.linalg.norm(w1 - w0)
                         + 2 * np.linalg.norm(v0 - w1))
        return area, circumference

    @staticmethod
    def key_angle(collision_tuple, model, merge_similar_angles=True):
        (v0, v1, w0, w1) = model.iMG[collision_tuple[0:4]]
        dv = v1 - v0
        angle = dv // np.gcd.reduce(dv)
        if merge_similar_angles:
            angle = sorted(np.abs(angle))
        return tuple(angle)

    @staticmethod
    def key(mode, model):
        if mode == "index":
            return Collisions.key_index
        elif mode == "area":
            assert model is not None
            return lambda x: Collisions.key_area(x, model)
        elif mode == "angle":
            assert model is not None
            return lambda x: Collisions.key_angle(
                x,
                model)
        elif mode == "species":
            assert model is not None
            return lambda x: Collisions.key_species(x, model)
        else:
            msg = ('Unsupported Parameter:\n\t'
                   'mode = ' + '{}'.format(mode))
            raise NotImplementedError(msg)

    def group(self,
              model=None,
              mode="index"):
        key_func = Collisions.key(mode, model)
        collisions = self.list
        grouped_collisions = dict()
        for coll in collisions:
            key = key_func(coll)
            if key in grouped_collisions.keys():
                grouped_collisions[key].append(coll)
            else:
                grouped_collisions[key] = [coll]
        return grouped_collisions

    def filter(self):
        grouped_collisions = self.group(mode="index")
        filtered_colls = [group[0] for group in grouped_collisions.values()]
        self.relations = np.array([coll[0:4] for coll in filtered_colls],
                                  dtype=int)
        self.weights = np.array([coll[4] for coll in filtered_colls],
                                dtype=float)
        return

    def sort(self,
             model=None,
             mode='index'):
        key_func = Collisions.key(mode, model)
        collisions = self.list
        sorted_colls = sorted(collisions, key=key_func)
        self.relations = np.array([coll[0:4] for coll in sorted_colls],
                                  dtype=int)
        self.weights = np.array([coll[4] for coll in sorted_colls],
                                dtype=float)
        return

    #####################################
    #           Configuration           #
    #####################################
    @staticmethod
    def is_collision(velocities,
                     masses):
        (v0, v1, w0, w1) = velocities
        # Ignore Collisions without changes in velocities
        if np.all(v0 == v1) and np.all(w0 == w1):
            return False
        # Invariance of momentum
        if not np.array_equal(masses[0] * (v1 - v0),
                              masses[2] * (w0 - w1)):
            return False
        # Invariance of energy
        energy_0 = np.sum(masses[0] * v0 ** 2 + masses[2] * w0 ** 2)
        energy_1 = np.sum(masses[1] * v1 ** 2 + masses[3] * w1 ** 2)
        if energy_0 != energy_1:
            return False
        # Accept this Collision
        return True

    def setup(self, model):
        """Generates the :attr:`relations` and :attr:`weights`.

        Parameters
        ----------
        model : :class:`Model`
        """
        assert isinstance(model, bp.Model)

        print('Generating Collision Array...')
        time_beg = time()
        # collect collisions in the following lists
        relations = []
        weights = []

        """The velocities are named in the following way:
        1. v* and w* are velocities of the first/second specimen, respectively
        2. v0 or w0 denotes the velocity before the collision
           v1 or w1 denotes the velocity after the collision
        """
        # choose function for local collisions
        if model.algorithm_relations == 'all':
            coll_func = Collisions.complete
        elif model.algorithm_relations == 'fast':
            coll_func = Collisions.convergent
        else:
            raise NotImplementedError(
                'Unsupported Selection Scheme: '
                '{}'.format(model.algorithm_relations)
            )

        grids = np.empty((4,), dtype=object)
        # use larger grids to shift within equivalence classes
        extended_grids = np.empty((4,), dtype=object)
        species = np.zeros(4, dtype=int)
        masses = np.zeros(4, dtype=int)
        # Iterate over Specimen pairs
        for (idx_v, grid_v) in enumerate(model.vGrids):
            grids[0:2] = grid_v
            # Todo make extended grid a property?
            extended_shape = (2 * grids[0].shape[0] - grids[0].shape[0] % 2,
                              2 * grids[0].shape[0] - grids[0].shape[0] % 2)
            extended_grids[0:2] = bp.Grid(extended_shape,
                                          grids[0].physical_spacing,
                                          grids[0].spacing,
                                          grids[0].is_centered)
            species[0:2] = idx_v
            for (idx_w, grid_w) in enumerate(model.vGrids):
                grids[2:4] = grid_w
                extended_shape = (2 * grids[2].shape[0] - grids[2].shape[0] % 2,
                                  2 * grids[2].shape[0] - grids[2].shape[0] % 2)
                extended_grids[2:4] = bp.Grid(extended_shape,
                                              grids[2].physical_spacing,
                                              grids[2].spacing,
                                              grids[2].is_centered)
                species[2:4] = idx_w
                masses[:] = model.masses[species]
                # Todo rename into collision_factor
                collision_rate = model.collision_factors[idx_v, idx_w]
                index_offset = np.array(model.index_range[species, 0])
                # Todo may be bad for differing weights
                # skip already computed combinations
                if idx_w < idx_v:
                    continue
                # group grid[0] points by distance to grid[2]
                for equivalence_class in grids[2].group(grids[0].iG).values():
                    # only generate colliding velocities(colvels)
                    # for a representative v0 of its group,
                    v0_repr = equivalence_class[0]
                    [repr_colvels, extended_weights] = coll_func(
                        extended_grids,
                        masses,
                        v0_repr,
                        collision_rate)
                    # Get relations for other class elements by shifting
                    for v0 in equivalence_class:
                        # shift extended colvels
                        new_colvels = repr_colvels + (v0 - v0_repr)
                        # get indices
                        new_rels = np.zeros(new_colvels.shape[0:2], dtype=int)
                        for i in range(4):
                            new_rels[:, i] = grids[i].get_idx(new_colvels[:, i, :])
                        new_rels += index_offset

                        # remove out-of-bounds or useless collisions
                        choice = np.where(
                            # must be in the grid
                            np.all(new_rels >= index_offset, axis=1)
                            # must be effective
                            & (new_rels[..., 0] != new_rels[..., 3])
                            & (new_rels[..., 0] != new_rels[..., 1])
                        )
                        # Add chosen Relations/Weights to the list
                        assert np.array_equal(
                                new_colvels[choice],
                                model.iMG[new_rels[choice]])
                        relations.extend(new_rels[choice])
                        weights.extend(extended_weights[choice])
                    # relations += new_rels
                    # weights += new_weights
        self.relations = np.array(relations, dtype=int)
        self.weights = np.array(weights, dtype=float)
        # remove redundant collisions
        self.filter()
        # sort collisions for better comparability
        self.sort()
        time_end = time()
        print('Time taken =  {t} seconds\n'
              'Total Number of Collisions = {n}\n'
              ''.format(t=round(time_end - time_beg, 3),
                        n=self.size))
        self.check_integrity()
        return

    ##############################################
    #       Collision Generation Functions       #
    ##############################################
    @staticmethod
    def complete(grids,
                 masses,
                 v0,
                 collision_rate):
        # store results in lists
        colvels = []     # colliding velocities
        weights = []
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
                # check if its a proper Collision
                if not Collisions.is_collision([v0, v1, w0, w1],
                                               masses):
                    continue
                # Collision is accepted -> Add to List
                colvels.append([v0, v1, w0, w1])
                weights.append(collision_rate)
        assert len(colvels) == len(weights)
        colvels = np.array(colvels)
        weights = np.array(weights)
        return [colvels, weights]

    @staticmethod
    def convergent(grids,
                   masses,
                   v0,
                   collision_rate):
        # angles = np.array([[1, 0], [1, 1], [0, 1], [-1, 1],
        #                    [-1, 0], [-1, -1], [0, -1], [1, -1]])
        # Todo This is sufficient, until real weights are used
        angles = np.array([[1, -1], [1, 0], [1, 1], [0, 1]])
        # store results in lists
        colvels = []    # colliding velocities
        weights = []
        # iterate over the given angles
        for axis_x in angles:
            # get y axis by rotating x axis 90°
            axis_y = np.array([[0, -1], [1, 0]])@axis_x
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
                    # check if its a proper Collision
                    if not Collisions.is_collision([v0, v1, w0, w1],
                                                   masses):
                        continue
                    # Collision is accepted -> Add to List
                    colvels.append([v0, v1, w0, w1])
                    weights.append(collision_rate)
        assert len(colvels) == len(weights)
        colvels = np.array(colvels)
        weights = np.array(weights)
        # Todo assert colvels.size != 0
        return [colvels, weights]

    def generate_collision_matrix(self, dt):
        # Size of complete velocity grid
        rows = np.max(self.relations.flatten()) + 1
        # Number of different collisions
        columns = self.size
        col_matrix = np.zeros(shape=(rows, columns),
                              dtype=float)
        for [i_col, col] in enumerate(self.relations):
            """Negative sign for pre-collision velocities
            => necessary for stability
               v[i]*v[j] - v[k]*v[l] is used as collision term
               => v'[*] = ... - X*u[*]"""
            # Todo multiplication with dt -> move out of matrix
            col_weight = dt * self.weights[i_col]
            col_matrix[col, i_col] = [-1, 1, -1, 1]
            col_matrix[col, i_col] *= col_weight
        col_mat = csr_matrix(col_matrix)
        return col_mat

    @property
    def number_of_collision_invariants(self):
        """:obj:`int` :
        Determines the number of collision invariants
        by computing the rank of the resulting matrix.
        """
        mat = self.generate_collision_matrix(1)
        rank = np.linalg.matrix_rank(mat.toarray())
        size_of_velocity_space = mat.shape[0]
        return size_of_velocity_space - rank

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`Collisions` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Collisions`
        """
        assert isinstance(hdf5_group, h5py.Group)
        # Todo move back in, with hashes
        # assert hdf5_group.attrs["class"] == "Collisions"
        self = Collisions()

        # read attributes from file
        try:
            self.relations = hdf5_group["Relations"][()]
        except KeyError:
            self.relations = None
        try:
            self.weights = hdf5_group["Weights"][()]
        except KeyError:
            self.weights = None

        self.check_integrity()
        return self

    def save(self, hdf5_group):
        """Write the main parameters of the :obj:`Collisions` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        # Todo Create hashes of parameters as attribute -> save & compare
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity()

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = "Collisions"

        # write all set attributes to file
        if self.relations is not None:
            hdf5_group["Relations"] = self.relations
        if self.weights is not None:
            hdf5_group["Weights"] = self.weights

        # check that the class can be reconstructed from the save
        other = Collisions.load(hdf5_group)
        assert self == other
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self, context=None):
        """Sanity Check"""
        if context is not None:
            assert isinstance(context, bp.Simulation)
        if self.relations is not None or self.weights is not None:
            assert self.relations is not None and self.weights is not None
            assert isinstance(self.relations, np.ndarray)
            assert isinstance(self.weights, np.ndarray)
            assert self.relations.dtype == int
            assert self.weights.dtype == float
            assert self.relations.ndim == 2
            assert self.weights.ndim == 1
            assert self.relations.shape == (self.weights.size, 4)
            for col in self.relations:
                # assert col[0] < col[1]
                # assert col[0] < col[2]
                if context is not None:
                    model = context.model
                    di_0 = (model.iMG[col[1]] - model.iMG[col[0]])
                    di_1 = (model.iMG[col[3]] - model.iMG[col[2]])
                    assert all(np.array(di_1 + di_0) == 0)
                    s = [model.get_specimen(v_col) for v_col in col]
                    assert s[0] == s[1] and s[2] == s[3]
                    # Todo add conserves energy check
            assert all(w > 0 for w in self.weights.flatten())
        return


# Todo move this into model class
#####################################
#           Visualization           #
#####################################
def plot(model,
         collisions,
         iterative=True,
         plot_object=None):
    assert isinstance(model, bp.Model)
    assert model.specimen <= len(model.plot_styles)

    # make sure its a list of Collisions,
    # this allows to plot lists of relations
    if not all(isinstance(coll, Collision) for coll in collisions):
        collisions = [coll if isinstance(coll, Collision)
                      else Collision(coll)
                      for coll in collisions]

    show_plot_directly = plot_object is None
    if plot_object is None:
        # Choose standard pyplot
        import matplotlib.pyplot as plt
        plot_object = plt

    # show all Collisions together
    for coll in collisions:
        coll.plot(model=model,
                  plot_object=plot_object)
    model.plot(plot_object)
    if show_plot_directly:
        plot_object.show()

    # show each element one by one
    if iterative:
        for coll in collisions:
            print("Relation:\n\t",
                  str(coll.relation))
            print("Velocities:\n\t",
                  str(model.iMG[coll.relation]).replace('\n', '\n\t'))
            plot_object.close()
            coll.plot(model=model,
                      plot_object=plot_object)
            # plot Grid on top of collision
            model.plot(plot_object=plot_object)
            plot_object.show()
    return plot_object
