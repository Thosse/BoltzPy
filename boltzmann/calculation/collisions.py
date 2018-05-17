
import numpy as np
from scipy.sparse import csr_matrix

from time import time


class Collisions:
    """
    Simple structure, that encapsulates collision data

    .. todo::
        - check integrity (non neg weights,
          no multiple occurrences, physical correctness)
        - print method - visualization of collisions
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
    """
    def __init__(self, cnf):
        self._cnf = cnf
        self._collision_arr = np.array([], dtype=int)
        self._weight_arr = np.array([], dtype=float)
        self._n = 0
        self._mat = csr_matrix(np.array([[]]))
        if self._cnf.collision_steps_per_time_step != 0:
            self._setup()
        return

    @property
    def cnf(self):
        """:obj:`~boltzmann.configuration.Configuration` :
        Points at the Configuration
        """
        return self._cnf

    @property
    def collision_arr(self):
        """:obj:`~numpy.ndarray` of :obj:`int` :
        Array of all simulated collisions

        Each collision :obj:`c` is described by the 4-tuple
        :attr:`collision_arr` [ :obj:`c`, :].
        It is an array of 4 indices of the
        :obj:`~boltzmann.configuration.SVGrid`,
        ordered is as follows:

            * :obj:`collision_arr` [c, 0]`
              = pre collision velocity of Specimen 1
            * :obj:`collision_arr` [c, 1]`
              = post collision velocity of Specimen 1
            * :obj:`collision_arr` [c, 2]`
              = pre collision velocity of Specimen 2
            * :obj:`collision_arr` [c, 3]`
              = post collision velocity of Specimen 2
        """
        return self._collision_arr

    @property
    def weight_arr(self):
        """:obj:`~numpy.ndarray` of :obj:`float` :
        Array of numeric integration weights
        for each collision in
        :obj:`collision_arr`.
        Array of shape=(:attr:`n`,)
        """
        return self._weight_arr

    @property
    def n(self):
        """:obj:`int` : Total number of Collisions"""
        return self._n

    @property
    def mat(self):
        """:obj:`~scipy.sparse.csr.csr_matrix` :
        Auxiliary Matrix (sparse), for fast execution of
        the collision step in :meth:`Calculation.run`
        """
        return self._mat

    #####################################
    #           Configuration           #
    #####################################
    def _setup(self):
        """Generates the :obj:`collision_arr`,
        based on the
        :attr:`~boltzmann.configuration.Configuration.coll_select_scheme`
        """
        gen_col_time = time()
        print('Generating Collision Array...', end='\r')
        if self.cnf.coll_select_scheme == 'Complete':
            if self.cnf.sv.form == 'rectangular':
                self._generate_collisions_complete()
            else:
                raise AttributeError('Currently, only rectangular '
                                     'grids are supported')
        else:
            message = 'Unsupported Selection Scheme:' \
                      '{}'.format(self.cnf.coll_select_scheme)
            raise AttributeError(message)
        print('Generating Collision Array...Done\n'
              'Time taken =  {} seconds\n'
              'Total Number of Collisions = {}\n'
              ''.format(round(time() - gen_col_time, 3), self.n))

        self.generate_collision_matrix()
        self.check_integrity()
        return

    # Todo Simplify - Looks horrible
    # noinspection PyAssignmentToLoopOrWithParameter
    def _generate_collisions_complete(self):
        """Generate all possible, non-useless collisions."""
        assert self.cnf.coll_select_scheme == 'Complete'
        # collect collisions in these lists
        col_arr = []
        weight_arr = []

        # Abbreviations
        species = self.cnf.s
        sv = self.cnf.sv

        # Each collision is an array of 4 Velocity-Indices
        # Ordered as: [[v_pre_s0, v_post_s0],
        #              [v_pre_s1, v_post_s1]]
        v = np.zeros((2, 2), dtype=int)
        # physical velocities (in multiples of d[s])
        # indexed as in v:  pv[i, j] = sv.G[ v[i, j] ]
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

        for s[0] in range(species.n):
            m[0] = species.mass[s[0]]
            d[0] = sv.d[s[0]]
            slc[0] = sv.index[s[0]:s[0] + 2]
            for s[1] in range(s[0], species.n):
                m[1] = species.mass[s[1]]
                d[1] = sv.d[s[1]]
                slc[1] = sv.index[s[1]:s[1] + 2]
                # v[0, 0] = v_pre_s0
                for v[0, 0] in range(slc[0, 0], slc[0, 1]):
                    pv[0, 0] = sv.G[v[0, 0]]
                    # v[0, 1] = v_post_s0
                    for v[0, 1] in range(v[0, 0]+1, slc[0, 1]):
                        # due to range, ignores v=(a,a,X,X) (no effect)
                        pv[0, 1] = sv.G[v[0, 1]]
                        # Velocity difference (dimensional, multiples of d[1])
                        # dpv_1 = - dpv_0 = pv[0, 0] - pv[0, 1]
                        dpv_1 = pv[0, 0] - pv[0, 1]
                        # v[1, 0] = v_pre_s1
                        for v[1, 0] in range(slc[1, 0], slc[1, 1]):
                            pv[1, 0] = sv.G[v[1, 0]]
                            # Calculate
                            pv[1, 1] = pv[1, 0] + dpv_1
                            # get index ov v[1, 1]
                            _v11 = sv.get_index(s[1], pv[1, 1])
                            if _v11 is not None:
                                v[1, 1] = _v11
                            else:
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
                            weight = self.compute_weight_arr(s)
                            weight_arr.append(weight)
        assert len(col_arr) == len(weight_arr)
        self._collision_arr = np.array(col_arr, dtype=int)
        self._weight_arr = np.array(weight_arr, dtype=float)
        self._n = self._collision_arr.shape[0]
        return

    def compute_weight_arr(self, specimen):
        """Computes the Collision weight

        Currently only depends on the colliding Specimen .
        This will change in the future.
        """
        col_rate = self._cnf.s.collision_rate_matrix[specimen[0], specimen[1]]
        n_cols = self._cnf.collision_steps_per_time_step
        if n_cols != 0:
            return col_rate / n_cols
        else:
            return col_rate

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
        # TODO Add nicely to Docstring
        # Energy Conservation law:
        #    m[0]*(d[0]*pv[0, 0])**2 + m[1]*(d[1]*pv[1, 0])**2
        # == m[0]*(d[0]*pv[0, 1])**2 + m[1]*(d[1]*pv[1, 1])**2
        # m[i]*d[i] = CONST for all Specimen
        # => can be canceled in Equation:
        #    d[0]*pv[0, 0]**2 + d[1]*pv[1, 0]**2
        # == d[0]*pv[0, 1]**2 + d[1]*pv[1, 1]**2
        pre_energy = np.array(d[0] * (pv[0, 0]**2)
                              + d[1] * (pv[1, 0]**2))
        pre_energy = pre_energy.sum()
        post_energy = np.array(d[0] * (pv[0, 1]**2)
                               + d[1] * (pv[1, 1]**2))
        post_energy = post_energy.sum()
        return np.allclose(pre_energy, post_energy)

    def generate_collision_matrix(self):
        if self._cnf.collision_steps_per_time_step == 0:
            return None
        gen_mat_time = time()
        print('Generating Collision Matrix...',
              end='\r')
        # Size of complete velocity grid
        rows = self._cnf.sv.index[-1]
        # Number of different collisions
        columns = self.n
        col_matrix = np.zeros(shape=(rows, columns),
                              dtype=float)
        for [i_col, col] in enumerate(self.collision_arr):
            # Negative sign for pre-collision velocities
            # => necessary for stability
            #   v[i]*v[j] - v[k]*v[l] is used as collision term
            #   => v'[*] = ... - X*u[*]
            col_weight = self._cnf.t.d * self.weight_arr[i_col]
            col_matrix[col, i_col] = [-1, 1, -1, 1]
            col_matrix[col, i_col] *= col_weight
        col_mat = csr_matrix(col_matrix)
        self._mat = col_mat
        print("Generating Collision Matrix...Done\n"
              "Time taken =  {} seconds\n"
              "".format(round(time() - gen_mat_time, 2)))
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check"""
        assert type(self._collision_arr) == np.ndarray
        assert self._collision_arr.dtype == int
        assert self._collision_arr.shape == (self._n, 4)
        for col in self._collision_arr:
            assert col[0] < col[1]
            assert col[0] < col[2]
            di_0 = self._cnf.sv.G[col[1]] - self._cnf.sv.G[col[0]]
            di_1 = self._cnf.sv.G[col[3]] - self._cnf.sv.G[col[2]]
            assert all(np.array(di_1 + di_0) == 0)
            s = [self._cnf.sv.get_specimen(v_col) for v_col in col]
            assert s[0] == s[1] and s[2] == s[3]
        assert type(self._weight_arr) == np.ndarray
        assert self._weight_arr.dtype == float
        assert self._weight_arr.shape == (self._n,)
        assert type(self._n) is int
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
