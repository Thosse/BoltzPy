
import numpy as np

from time import time


class Collisions:
    """
    Simple structure, that encapsulates collision data

    .. todo::
        - check integrity (non neg weights,
          no multiple occurrences, physical correctness)
        - print method
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
        self._collision_arr = np.array([], dtype=int)
        self._weight_arr = np.array([], dtype=float)
        self._n = 0
        self._cnf = cnf
        return

    @property
    def collision_arr(self):
        """:obj:`~numpy.ndarray` of :obj:`int`:
        Array of all simulated collisions.

        A collision :obj:`c` is described by the 4-tuple
        :attr:`c_arr` [ :obj:`c`, :].
        It is an array of 4 indices of the
        :obj:`~boltzmann.configuration.SVGrid`,
        ordered is as follows:

            * :obj:`c_arr[c, 0]` = v_pre_collision of Specimen 1
            * :obj:`c_arr[c, 1]` = v_post_collision of Specimen 1
            * :obj:`c_arr[c, 2]` = v_pre_collision of Specimen 2
            * :obj:`c_arr[c, 3]` = v_post_collision of Specimen 2
        """
        return self._collision_arr

    @property
    def weight_arr(self):
        """:obj:`~numpy.ndarray` of :obj:`float`:
        Array of numeric integration weights
        for each collision in c_arr.
        Array of shape=(:attr:`~boltzmann.configuration.Collisions.n`,)
        """
        return self._weight_arr

    @property
    def n(self):
        """:obj:`int`: Total number of Collisions."""
        return self._n

    @property
    def cnf(self):
        """:obj:`~boltzmann.configuration.Configuration`:
        Points at the Configuration"""
        return self._cnf

    def setup(self):
        """Generates the Collisions, based on the Selection Scheme"""
        gen_col_time = time()
        if self.cnf.collision_selection_scheme == 'Complete':
            if self.cnf.sv.form is 'rectangular':
                self._generate_collisions_complete()
            else:
                # Todo Throw exception
                print('Currently only supports rectangular grids!')
                assert False
        else:
            # Todo Throw exception
            print('ERROR - Unspecified Collision Scheme')
            assert False
        print("Generation of Collision list - Done\n"
              "Total Number of Collisions = {}\n"
              "Time taken =  {} seconds"
              "".format(self.n, round(time() - gen_col_time, 3)))
        return

    # Todo Simplify - Looks horrible
    # noinspection PyAssignmentToLoopOrWithParameter
    def _generate_collisions_complete(self):
        """Generate all possible, non-useless collisions."""
        assert self.cnf.collision_selection_scheme is 'Complete'
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
        alpha = self._cnf.s.alpha[specimen[0], specimen[1]]
        n_cols = self._cnf.collision_steps_per_time_step
        return alpha / n_cols

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
        """Checks if the collision denoted by v fulfills energy conservation.:

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
