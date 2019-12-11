
import numpy as np
from scipy.sparse import csr_matrix
from time import time
import h5py

import boltzpy as bp


class Collisions:
    r"""Generates and encapsulates the collision :attr:`relations`
    and :attr:`weights`.

    .. todo::
        - check integrity (non neg weights,
          no multiple occurrences, physical correctness)
        - plot method - visualization of collisions (opt. param: svgrid, species)
        - add load / save method
        - **Add Stefan's Generation-Scheme**
        - can both the transport and the collisions
          be implemented as interpolations? -> GPU Speed-UP
        - count collisions for each pair of specimen? Useful?
          This allows to do collision steps specieswise, but also leads to more matrices...
        - Check if its faster to switch v[0, 1] and v[1, 0]?
        - @generate: replace for loops by numpy.apply_along_axis
          (this probably needs several additional functions).

    Attributes
    ----------
    relations : :obj:`~numpy.array` [:obj:`int`]
        Contains the active collisions.
        Each collision is a 4-tuple of indices in :attr:`sv.iMG`
        and is in the form
        :math:`\left[ v_0, v_1, w_0, w_1\right]`,
        where *_0, *_1 are the pre and post collision velocities, respectively.
    weights : :obj:`~numpy.array` [:obj:`float`]
        Contains the numeric integration weights
        of the respective collision in :attr:`relations`.
    """
    def __init__(self):
        self.relations = None
        self.weights = None
        return

    @property
    def size(self):
        """:obj:`int` : Total number of active collisions."""
        if self.relations is not None:
            return self.relations.shape[0]
        else:
            return 0

    @property
    def is_set_up(self):
        """:obj:`bool` :
        True, if the instance is completely set up
        and ready to call :meth:`~Simulation.run_computation`.
        False Otherwise.
        """
        is_set_up = self.relations is not None and self.weights is not None
        return is_set_up

    #####################################
    #           Configuration           #
    #####################################
    def setup(self, scheme, svgrid, species):
        """Generates the :attr:`relations` and :attr:`weights`.

        Parameters
        ----------
        scheme : :class:`Scheme`
        svgrid : :class:`SVGrid`
        species : :class:`Species`
        """
        assert isinstance(scheme, bp.Scheme)
        assert isinstance(svgrid, bp.SVGrid)
        assert isinstance(species, bp.Species)
        print('Generating Collision Array...')
        time_beg = time()
        if scheme.Collisions_Generation == 'UniformComplete':
            # Todo this should just as well work for other forms
            # Todo it just depends on the find_index
            self._generate_collisions_complete(svgrid, species)
        else:
            msg = ('Unsupported Selection Scheme:'
                   + '{}'.format(scheme.Collisions_Generation))
            raise NotImplementedError(msg)
        time_end = time()
        print('Time taken =  {t} seconds\n'
              'Total Number of Collisions = {n}\n'
              ''.format(t=round(time_end - time_beg, 3),
                        n=self.size))
        self.check_integrity()
        return

    def _generate_collisions_complete(self,
                                      svgrid,
                                      species):
        """Generate all possible, non-useless collisions."""
        if any(form != 'rectangular' for form in svgrid.forms):
            msg = 'Unsupported SVGrid forms: {f}'.format(f=svgrid.forms)
            raise NotImplementedError(msg)
        # collect collisions in the following lists
        collisions = []
        weights = []

        """The velocities are named in the following way:
        1. v* and w* are velocities of the first/second specimen, respectively
        2. v0 or w0 denotes the velocity before the collision
           v1 or w1 denotes the velocity after the collision
        """
        # Choose first Specimen
        for (idx_spc_v, grid_v) in enumerate(svgrid.vGrids):
            index_offset_v = svgrid.index_range[idx_spc_v, 0]
            mass_v = species.mass[idx_spc_v]
            # Choose second Specimen
            for (idx_spc_w, grid_w) in enumerate(svgrid.vGrids):
                index_offset_w = svgrid.index_range[idx_spc_w, 0]
                mass_w = species.mass[idx_spc_w]
                if idx_spc_w < idx_spc_v:
                    # we already checked this combination
                    continue
                """Iterate over all possible velocity combinations
                check whether they are proper collisions
                if yes, then store them in the list"""
                for (loc_v0, v0) in enumerate(grid_v.iG):
                    # global index in self.iMG
                    index_v0 = index_offset_v + loc_v0
                    assert np.all(v0 == svgrid.iMG[index_v0])
                    # we choose idx_v0 < idx_v1 to ignore v=(a, a, * , *)
                    # and ignore repeating collisions
                    for (loc_v1, v1) in enumerate(grid_v.iG[loc_v0 + 1:]):
                        # global index in self.iMG
                        index_v1 = index_v0 + 1 + loc_v1
                        assert np.all(v1 == svgrid.iMG[index_v1])
                        # calculate Velocity (index) difference
                        diff_v = v1 - v0
                        for (loc_w0, w0) in enumerate(grid_w.iG):
                            # global index in self.iMG
                            index_w0 = index_offset_w + loc_w0
                            assert np.all(w0 == svgrid.iMG[index_w0])
                            # Calculate w1, using the momentum invariance
                            diff_w = -diff_v * mass_v // mass_w
                            w1 = w0 + diff_w
                            # find the global index of w1, if its in the grid
                            index_w1 = svgrid.find_index(idx_spc_w, w1)
                            if index_w1 is None:
                                continue
                            # check if its a proper Collision
                            new_col_idx = [index_v0,
                                           index_v1,
                                           index_w0,
                                           index_w1]
                            new_col_val = np.array([v0, v1, w0, w1],
                                                   dtype=int)
                            is_a_collision = self.check_single_collision(
                                new_col_idx,
                                new_col_val,
                                [mass_v, mass_w])
                            if not is_a_collision:
                                continue
                            # Collision is accepted -> Add to List
                            collisions.append(new_col_idx)
                            new_weight = species.collision_rates[idx_spc_v,
                                                                 idx_spc_w]
                            weights.append(new_weight)
        assert len(collisions) == len(weights)
        self.relations = np.array(collisions, dtype=int)
        self.weights = np.array(weights, dtype=float)
        return

    @staticmethod
    def check_single_collision(collision_indices,
                               collision_velocities,
                               masses):
        """Check whether the Collision Candidate fulfills all necessary
        Conditions.

        Parameters
        ----------
        collision_indices : :obj:`list` [:obj:`int`]
            Indices of the colliding velocities in the SV-Grid.
            Array of shape=(4).
        collision_velocities : :obj:`~numpy.array` [:obj:`int`]
            Colliding velocities in the SV-Grid
            in multitudes of :attr:`SVGrid.delta`.
            Array of shape=(4, :attr:`SVGrid.ndim`).
        masses : array(int)
            Step sizes of the Specimens Velocity-Grids
            Array of shape=(2,).

        Returns
        -------
        bool
            True if collision fulfills all conditions, False otherwise.
        """
        # Abbreviation

        v = collision_velocities[0:2]
        w = collision_velocities[2:4]
        m = masses
        # Value not found in Grid
        if collision_indices[3] is None:
            return False
        # Ignore collisions that were already found
        if collision_indices[3] < collision_indices[0]:
            # Todo maybe add ...[2] < ...[0] as well
            return False
        # Ignore v=(X,b,b,X) for same species
        # as such collisions have no effect
        if np.all(collision_indices[1] == collision_indices[2]):
            return False
        # Ignore Collisions with no initial velocity difference
        if np.all(v[0] == w[0]):
            return False
        # invariance of momentum
        if not np.all(m[0] * (v[1] - v[0]) == m[1] * (w[0] - w[1])):
            return False
        # invariance of energy
        energy_0 = np.sum(m[0] * v[0] ** 2 + m[1] * w[0] ** 2)
        energy_1 = np.sum(m[0] * v[1] ** 2 + m[1] * w[1] ** 2)
        if energy_0 != energy_1:
            return False
        # Accept this Collision
        return True

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
        if self.relations is None:
            return None
        else:
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
                assert col[0] < col[1]
                assert col[0] < col[2]
                if context is not None:
                    sv = context.sv
                    di_0 = (sv.iMG[col[1]] - sv.iMG[col[0]])
                    di_1 = (sv.iMG[col[3]] - sv.iMG[col[2]])
                    assert all(np.array(di_1 + di_0) == 0)
                    s = [sv.get_specimen(v_col) for v_col in col]
                    assert s[0] == s[1] and s[2] == s[3]
                    # Todo add conserves energy check
            assert all(w > 0 for w in self.weights.flatten())
        return