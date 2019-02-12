
import numpy as np
from scipy.sparse import csr_matrix
from time import time
import h5py

import boltzpy as bp


# Todo Add Collision Scheme "Free_Flow"
# Todo rework this, using new SVGrid
class Collisions:
    r"""Generates and encapsulates the collision :attr:`relations`
    and :attr:`weights`.

    .. todo::
        - check integrity (non neg weights,
          no multiple occurrences, physical correctness)
        - print method - visualization of collisions
        - add load / save method
        - **Add Stefan's Generation-Scheme**
        - can both the transport and the collisions
          be implemented as interpolations? -> GPU Speed-Up
        - How to sort the arrays for maximum efficiency?

        - count collisions for each pair of specimen? Useful?
        - only for implemented index_difference:
          choose v[2] out of smaller grid
          which only contains possible values
        - Check if its faster to switch v[0, 1] and v[1, 0]?
        - @generate: replace for loops by numpy.apply_along_axis
          (this probably needs several additional functions).

    Attributes
    ----------
    relations : :obj:`~numpy.array` [:obj:`int`]
        Contains the active collisions.
        Each collision is a 4-tuple of indices in :attr:`sv.iG`
        and is in the form
        :math:`\left[ v_{s_1}^{pre}, v_{s_1}^{post},
        v_{s_2}^{pre}, v_{s_2}^{post}
        \right]`.
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
        True, if the instance is completely set up and ready to call :meth:`~Simulation.run_computation`.
        False Otherwise.
        """
        return self.relations is not None and self.weights is not None

    #####################################
    #           Configuration           #
    #####################################

    def setup(self, scheme, svgrid, species):
        """Generates the :obj:`collision_arr`,
        based on the :attr:`~boltzpy.Scheme`.
        """
        assert isinstance(scheme, bp.Scheme)
        assert isinstance(svgrid, bp.SVGrid)
        assert isinstance(species, bp.Species)
        print('Generating Collision Array...', end='')
        time_beg = time()
        if scheme.Collisions_Generation == 'UniformComplete':
            self._generate_collisions_complete(svgrid, species)
        else:
            msg = ('Unsupported Selection Scheme:'
                   + '{}'.format(scheme.Collisions_Generation))
            raise NotImplementedError(msg)
        time_end = time()
        print('Done\n'
              'Time taken =  {t} seconds\n'
              'Total Number of Collisions = {n}\n'
              ''.format(t=round(time_end - time_beg, 3),
                        n=self.size))
        self.check_integrity()
        return

        # Todo Simplify - Looks horrible
        # noinspection PyAssignmentToLoopOrWithParameter

    def _generate_collisions_complete(self, svgrid, species):
        """Generate all possible, non-useless collisions."""
        if svgrid.form != 'rectangular':
            msg = 'Unsupported SVGrid form: {f}'.format(f=svgrid.form)
            raise NotImplementedError(msg)
        # collect collisions in these lists
        col_arr = []
        weight_arr = []

        # Abbreviations
        sv = svgrid
        mass = sv.masses
        n_spc = mass.size

        # Each collision is an array of 4 Velocity-Indices
        # Ordered as: [[v_pre_s0, v_post_s0],
        #              [v_pre_s1, v_post_s1]]
        v = np.zeros((2, 2), dtype=int)
        # physical velocities (in multiples of d[s])
        # indexed as in v:  pv[i, j] = sv.iMG[ v[i, j] ]
        pv = np.zeros((2, 2, sv.dim), dtype=int)

        # For the colliding specimen we keep track of
        # specimen-indices
        s = np.zeros((2,), dtype=int)
        # masses
        m = np.zeros((2,), dtype=int)
        # step sizes
        d = np.zeros((2,), dtype=float)
        # the slices in the sv_grid, slc[spc, :] = [start, end+1]
        slc = np.zeros((2, 2), dtype=int)

        for s[0] in range(n_spc):
            m[0] = mass[s[0]]
            d[0] = sv.vGrids[s[0]].d
            slc[0] = sv.idx_range(s[0])
            for s[1] in range(s[0], n_spc):
                m[1] = mass[s[1]]
                d[1] = sv.vGrids[s[1]].d
                slc[1] = sv.idx_range(s[1])
                # v[0, 0] = v_pre_s0
                for v[0, 0] in range(slc[0, 0], slc[0, 1]):
                    pv[0, 0] = sv.iMG[v[0, 0]]
                    # v[0, 1] = v_post_s0
                    for v[0, 1] in range(v[0, 0] + 1, slc[0, 1]):
                        # due to range, ignores v=(a,a,X,X) (no effect)
                        pv[0, 1] = sv.iMG[v[0, 1]]
                        # Velocity difference (dimensional, multiples of d[1])
                        # dpv_1 = - dpv_0 = pv[0, 0] - pv[0, 1]
                        dpv_1 = pv[0, 0] - pv[0, 1]
                        # v[1, 0] = v_pre_s1
                        for v[1, 0] in range(slc[1, 0], slc[1, 1]):
                            pv[1, 0] = sv.iMG[v[1, 0]]
                            # Calculate
                            pv[1, 1] = pv[1, 0] + dpv_1
                            # get index ov v[1, 1]
                            try:
                                v[1, 1] = sv.get_index(s[1], pv[1, 1])
                            except ValueError:
                                continue
                            # if v[1, 1] < v[0, 0], then the collision was
                            # already found before
                            if v[1, 1] < v[0, 0]:
                                continue

                            # Check if v fulfills all conditions
                            if not self._is_collision(d, v, pv):
                                continue

                            # Collision is accepted -> Add to List
                            col_arr.append(v.flatten())
                            weight = species.collision_rates[s[0], s[1]]
                            weight_arr.append(weight)
        assert len(col_arr) == len(weight_arr)
        self.relations = np.array(col_arr, dtype=int)
        self.weights = np.array(weight_arr, dtype=float)
        return

    @staticmethod
    def _is_collision(d, v, pv):
        """Check whether the Collision Candidate fulfills all necessary
        Conditions.

        Parameters
        ----------
        d : array(float)
            Step sizes of the Specimens Velocity-Grids
            Array of shape=(2,).
        v : array(int)
            Indices of the colliding velocities in the SV-Grid.
            Array of shape=(2,2).
        pv : array(int)
            Physical Velocities, as multiples of their respective step size d.
            Array of shape=(2,2, sv.dim).

        Returns
        -------
        bool
            True if collision fulfills all conditions, False otherwise.
        """
        # Ignore v=(X,b,b,X) (only for s[0]=s[1], has no effect)
        if v[0, 1] == v[1, 0]:
            return False
        # Ignore Collisions with no initial velocity difference
        elif np.allclose(pv[0, 0] * d[0], pv[1, 0] * d[1]):
            return False
        # Ignore Collisions not fulfilling law of conservation of energy
        elif not Collisions._meets_energy_conservation(d, pv):
            return False
        # Accept this Collision
        else:
            return True

    # Todo Check if its pv or iG - d * pv makes no sense
    @staticmethod
    def _meets_energy_conservation(d, pv):
        """Checks if the collision denoted by pv
        fulfills energy conservation

        Parameters
        ----------
        d : array(float)
            Step sizes of the Specimens Velocity-Grids
        pv : array(int)
            Indices of the colliding velocities in the Complete SV-Grid

        Returns
        -------
        bool
            True if collision fulfills energy conservation law,
            False otherwise.
        """
        assert isinstance(pv, np.ndarray)
        assert isinstance(d, np.ndarray)
        # TODO Add nicely to Docstring
        # Energy Conservation law:
        #    m[0]*(d[0]*pv[0, 0])**2 + m[1]*(d[1]*pv[1, 0])**2
        # == m[0]*(d[0]*pv[0, 1])**2 + m[1]*(d[1]*pv[1, 1])**2
        # m[i]*d[i] = CONST for all Specimen
        # => can be canceled in Equation:
        #    d[0]*pv[0, 0]**2 + d[1]*pv[1, 0]**2
        # == d[0]*pv[0, 1]**2 + d[1]*pv[1, 1]**2
        pre_energy = d[0] * pv[0, 0] ** 2 + d[1] * pv[1, 0] ** 2
        assert isinstance(pre_energy, np.ndarray)
        pre_energy = pre_energy.sum()
        post_energy = d[0] * pv[0, 1] ** 2 + d[1] * pv[1, 1] ** 2
        assert isinstance(post_energy, np.ndarray)
        post_energy = post_energy.sum()
        return np.allclose(pre_energy, post_energy)

    def generate_collision_matrix(self, dt):
        # Size of complete velocity grid
        rows = np.max(self.relations.flatten()) + 1
        # Number of different collisions
        columns = self.size
        col_matrix = np.zeros(shape=(rows, columns),
                              dtype=float)
        for [i_col, col] in enumerate(self.relations):
            # Negative sign for pre-collision velocities
            # => necessary for stability
            #   v[i]*v[j] - v[k]*v[l] is used as collision term
            #   => v'[*] = ... - X*u[*]
            # Todo multiplication with dt -> move out of matrix
            col_weight = dt * self.weights[i_col]
            col_matrix[col, i_col] = [-1, 1, -1, 1]
            col_matrix[col, i_col] *= col_weight
        col_mat = csr_matrix(col_matrix)
        return col_mat

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
            self.relations = hdf5_group["Relations"].value
        except KeyError:
            self.relations = None
        try:
            self.weights = hdf5_group["Weights"].value
        except KeyError:
            self.weights = None
        # Todo read Scheme parameters from relations (save as attributes)

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

# Todo Keep for testing
    # @staticmethod
    # def Test_is_v_in_col_arr(v, col_arr):
    #     """Check if :obj:`v` or an equivalent permutation of it
    #     is already in :obj:`col_arr`"""
    #     # Todo speed up
    #     v = v.flatten()
    #     # set of permutation, which are checked:
    #     col_permutations = np.zeros((4, 4))
    #     col_permutations[0] = np.copy(v)
    #     col_permutations[1] = np.array([v[1], v[0], v[3], v[2]])
    #     col_permutations[2] = np.array([v[2], v[3], v[0], v[1]])
    #     col_permutations[3] = np.array([v[3], v[2], v[1], v[0]])
    #     for old_col in col_arr:
    #         if np.any(np.all(col_permutations == old_col, axis=1)):
    #             print("Permutations = \n{}".format(col_permutations))
    #             print("Found Col = \n{}".format(old_col))
    #             return True
    #     return False
