from . import species as b_spc
from . import svgrid as b_svg

import numpy as np
import math
from time import time


# Todo this can be optimized
def get_close_int(float_array, precision=1e-6, dtype=int):
    int_array = np.zeros(shape=float_array.shape,
                         dtype=dtype)
    for (i, real_number) in enumerate(float_array):
        close_int = int(math.floor(real_number))
        if math.fabs(close_int - real_number) < precision:
            int_array[i] = close_int
        elif math.fabs(close_int+1 - real_number) < precision:
            int_array[i] = close_int+1
        else:
            assert False, "{}-th Number of the Array " \
                          "is not close to an integer " \
                          " ({})".format(i, real_number)
    return int_array


class Collisions:
    """
    Simple structure, that encapsulates collision data

    .. todo::
        - **Add Stefans Generation-Scheme**
        - idea: are index-differences v01-v00==-v_11+v10 (vectorwise?)
        - can both the transport and the collisions
          be implemented as interpolations? -> GPU Speed-Up
        - How to sort the arrays for maximum efficiency?
        - check integrity (non neg weights,
          no multiple occurrences, physical correctness)
        - count collisions for each pair of specimen?
        - calculate index difference instead of dpv[0]
          -> maybe Problematic, for different construction schemes
          e.g. Hans grid-idea (one v-grid rotated 45°),
          rotated grids need to be tested, but should be fine
          => Implement as optimized collision generation
        - only for implemented index_difference:
          choose v[2] out of smaller grid
          which only contains possible values
        - Check if its faster to switch v[0, 1] and v[1, 0]?
        - @generate: replace for loops by numpy.apply_along_axis
          (this probably needs several additional functions)

    Attributes
    ----------
    i_arr : np.ndarray
        Describes each collision by the 4 indices
        of the colliding velocities.
        Integer-Array of shape=(n, 4).
    weight : np.ndarray
        Denotes Integration weight of each collision.
        Float-Array of shape=(n,)
    n : int
        Total number of Collisions.
    scheme : str
        Denotes by which scheme the collisions ware generated

       """
    SELECTION_SCHEMES = ['complete']

    def __init__(self,
                 float_type,
                 int_type):
        self.i_arr = np.array([], dtype=int_type)
        self.weight = np.array([], dtype=float_type)
        self.n = 0
        self.scheme = 'not_initialized'

    def generate_collisions(self,
                            species,
                            sv_grid):
        assert self.scheme in Collisions.SELECTION_SCHEMES
        assert type(species) is b_spc.Species
        assert type(sv_grid) is b_svg.SVGrid
        if self.scheme is 'complete':
            self._generate_collisions_complete(species, sv_grid)
        else:
            print('ERROR - Unspecified Collision Scheme')
            assert False
        return

    def _generate_collisions_complete(self,
                                      species,
                                      sv_grid):
        assert self.scheme is 'complete'
        gen_col_time = time()
        i_arr = []
        weight = []
        # Error margin for boundary checks
        err = np.amin(sv_grid.d) / 1000

        # Each collision is an array of 4 indices, ordered as:
        # (v_pre_0, v_post_0, v_pre_1, v_post_1)
        v = np.zeros((2, 2), dtype=sv_grid.iType)
        # physical velocities, indexed by v
        # (pv_pre_0, pv_post_0, pv_pre_1, pv_post_1)
        pv = np.zeros((2, 2, sv_grid.dim), dtype=sv_grid.fType)
        # physical difference in velocities
        dpv = np.zeros((2, sv_grid.dim), dtype=sv_grid.fType)

        # For each colliding specimen we keep track of
        # specimen-indices
        s = np.zeros((2,), dtype=sv_grid.iType)
        # masses
        m = np.zeros((2,), dtype=sv_grid.iType)
        # the slices in the sv_grid, slc[spc, :] = [start, end+1]
        slc = np.zeros((2, 2), dtype=sv_grid.iType)

        for s[0] in range(species.n):
            slc[0] = sv_grid.index[s[0]:s[0]+2]
            m[0] = species.mass[s[0]]
            # noinspection PyAssignmentToLoopOrWithParameter
            for s[1] in range(s[0], species.n):
                slc[1] = sv_grid.index[s[1]:s[1]+2]
                m[1] = species.mass[s[1]]
                d1 = sv_grid.d[s[1]]
                # v_pre of s[0]
                for v[0, 0] in range(slc[0, 0], slc[0, 1]):
                    pv[0, 0] = sv_grid.G[v[0, 0]]
                    # v_post of s[0]
                    # ignores (a,a,b,b)-collisions (no effect), due to range
                    # noinspection PyAssignmentToLoopOrWithParameter
                    for v[0, 1] in range(v[0, 0]+1, slc[0, 1]):
                        pv[0, 1] = sv_grid.G[v[0, 1]]
                        dpv[0] = pv[0, 1] - pv[0, 0]
                        dpv[1] = -m[0] / m[1] * dpv[0]
                        # iterate through pre-collision velocities of s[1]
                        # noinspection PyAssignmentToLoopOrWithParameter
                        for v[1, 0] in range(slc[1, 0], slc[1, 1]):
                            # Ignore (a,b,b,a)-Collisions (no effect)
                            if v[0, 1] == v[1, 0]:
                                continue
                            pv[1, 0] = sv_grid.G[v[1, 0]]
                            pv[1, 1] = pv[1, 0] + dpv[1]
                            # Ignore (a,X,a,X)-Collisions (no effect)
                            if np.allclose(pv[0, 0], pv[1, 0]):
                                continue
                            # check if v[1, 1] is in grid boundaries
                            _lower_bound = sv_grid.b[s[1], :, 0]-err
                            if np.less(pv[1, 1], _lower_bound).any():
                                continue
                            _upper_bound = sv_grid.b[s[1], :, 1]+err
                            if np.greater(pv[1, 1], _upper_bound).any():
                                continue
                            # check energy conservation
                            pre_energy = np.array(m[0]*(pv[0, 0]**2)
                                                  + m[1]*(pv[1, 0]**2))
                            pre_energy = pre_energy.sum()
                            post_energy = np.array(m[0]*(pv[0, 1]**2)
                                                   + m[1]*(pv[1, 1]**2))
                            post_energy = post_energy.sum()
                            if not np.allclose(pre_energy, post_energy):
                                continue
                            # check if pv[1, 1] is grid point
                            _v11 = (pv[1, 1] - sv_grid.b[s[1], :, 0])/d1
                            _v11 = get_close_int(_v11)
                            _v11 = sv_grid.get_local_flat_index(s[1], _v11)
                            # generate flat index
                            v[1, 1] = slc[1, 0] + _v11
                            i_arr.append(v.flatten())
                            weight.append(1)
        assert len(i_arr) == len(weight)
        self.i_arr = np.array(i_arr, dtype=sv_grid.iType)
        self.weight = np.array(weight, dtype=sv_grid.fType)
        self.n = self.i_arr.shape[0]
        print("Generation of Collision list - Done\n"
              "Total Number of Collisions = {}\n"
              "Time taken =  {} seconds"
              "".format(self.n, round(time() - gen_col_time, 3)))
        return
