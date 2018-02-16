import boltzmann.configuration.species as b_spc
import boltzmann.configuration.svgrid as b_svg

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
    """Simple structure, that encapsulates collision data

    - idea: are index-differences v0->v2 == v_3->v1 ?
    - count collisions for each pair of specimen
    - can both the transport and the collisions be implemented
      as interpolations?
    - How to sort the arrays for maximum efficiency?
    - check integrity (non neg weights,
      no multiple occurrences, physical correctness)

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
                            sv_grid,
                            selection_scheme):
        assert selection_scheme in Collisions.SELECTION_SCHEMES
        assert type(species) is b_spc.Species
        assert type(sv_grid) is b_svg.SVGrid
        if selection_scheme is 'complete':
            self._generate_collisions_complete(species, sv_grid)
        else:
            print('ERROR - Unspecified Collision Scheme')
            assert False
        return

    # Todo needs speed up -> vectorize / parallelize
    def _generate_collisions_complete(self,
                                      species,
                                      sv_grid):
        tmp_time = time()
        i_arr = []
        weight = []

        # Each collision is an array of 4 indices, ordered as:
        # (v_pre_0, v_post_0, v_pre_1, v_post_1)
        v = np.zeros((2, 2), dtype=sv_grid.iType)
        # physical velocities, indexed by v
        # (pv_pre_0, pv_post_0, pv_pre_1, pv_post_1)
        pv = np.zeros((2, 2, sv_grid.dim), dtype=sv_grid.fType)
        dv = np.zeros((2, sv_grid.dim), dtype=sv_grid.fType)

        # For each colliding specimen we keep track of
        # specimen-indices
        s = np.zeros((2,), dtype=sv_grid.iType)
        # masses
        m = np.zeros((2,), dtype=sv_grid.iType)
        # the slices in the sv_grid, slc[spc, :] = [start, end+1]
        slc = np.zeros((2, 2), dtype=sv_grid.iType)
        # physical difference in velocities
        # Todo calculate index difference instead of dv[0]
        # Todo even index_difference choose v[2] out of smaller grid
        # Todo which only contains possible values

        # Todo possibly faster if switching v[0, 1] and v[1, 0]
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
                        dv[0] = pv[0, 1] - pv[0, 0]
                        dv[1] = -m[0] / m[1] * dv[0]
                        # iterate through pre-collision velocities of s[1]
                        # noinspection PyAssignmentToLoopOrWithParameter
                        for v[1, 0] in range(slc[1, 0], slc[1, 1]):
                            pv[1, 0] = sv_grid.G[v[1, 0]]
                            pv[1, 1] = pv[1, 0] + dv[1]
                            # Ignore (a,b,b,a)-Collisions (no effect)
                            if v[0, 1] == v[1, 0]:
                                continue
                            # Ignore (a,X,a,X)-Collisions (no effect)
                            if np.allclose(pv[0, 0], pv[1, 0]):
                                continue
                            # check if v[1, 1] is in grid boundaries
                            if np.less(pv[1, 1],
                                       sv_grid.b[s[1], :, 0]).any():
                                continue
                            if np.greater(pv[1, 1],
                                          sv_grid.b[s[1], :, 1]).any():
                                continue
                            # check energy conservation
                            # noinspection PyTypeChecker
                            pre_energy = np.sum(m[0]*(pv[0, 0]**2)
                                                + m[1]*(pv[1, 0]**2))
                            # noinspection PyTypeChecker
                            post_energy = np.sum(m[0]*(pv[0, 1]**2)
                                                 + m[1]*(pv[1, 1]**2))
                            if pre_energy != post_energy:
                                continue
                            # check if pv[1, 1] is grid point
                            _v11 = (pv[1, 1] - sv_grid.b[s[1], :, 0])/d1
                            _v11 = get_close_int(_v11)
                            _v11 = sv_grid.get_local_flat_index(s[1], _v11)
                            # generate flat index
                            v[1, 1] = slc[1, 0] + _v11
                            i_arr.append(v.flatten())
                            weight.append(1)
        print("Time taken =  {}".format(time()-tmp_time))
        assert len(i_arr) == len(weight)
        self.index = np.array(i_arr, dtype=sv_grid.iType)
        self.weight = np.array(weight, dtype=sv_grid.fType)
        self.n = len(i_arr)
