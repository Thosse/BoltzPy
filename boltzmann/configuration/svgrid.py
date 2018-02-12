import numpy as np
import math

import boltzmann.configuration.species as b_spc
import boltzmann.configuration.grid as b_grd


def np_gcd(array_of_ints):
    if array_of_ints.shape == (0,):
        return 1
    gcd = array_of_ints[0]
    for new_number in array_of_ints[1:]:
        gcd = math.gcd(gcd, new_number)
    return gcd


class SVGrid:
    """Class of the concatenated Velocity Grid of all Specimen.

    .. todo::
        - Todo Add unit tests
        - should this be inheriting from Grid?
        - n[0:dim] should all be equal values. Reduce this?

    Attributes
    ----------
    dim : int
        Dimensionality of all Grids.  Applies to all Specimen.
    b : np.ndarray(fType)
        Denotes [minimum, maximum] physical Grid values for each specimen.
        Array of shape=(s.n, dim, 2) and dtype=fType.
    d : np.ndarray(fType)
        Denotes the Grid step size for each specimen.
        Array of shape=(s.n, dim).
    n : np.ndarray(iType)
        n[:, 0:dim] denotes the number of Grid points per dimension
        for each specimen.
        n[:, -1] denotes the total number of Grid points
        for each specimen.
        Array of shape=(s.n, dim+1,).
    index : np.ndarray(iType)
        index[i] denotes the beginning of the i-th velocity grid.
        By Definition index[0] is 0 and index[-1] is n[:,-1].sum().
        Array of shape=(s.n+1) and dType=iType.
    shape : str
        Shape of all Grids. Applies to all Specimen.
        Can only be rectangular so far.
    fType : np.dtype
        Determines data type of floats.
        Applies to all Specimen.
    iType : np.dtype
        Determines data type of integers.
        Applies to all Specimen.

    """

    def __init__(self,
                 species,
                 velocity_grid,
                 check_integrity=True):
        assert type(species) is b_spc.Species
        assert type(velocity_grid) is b_grd.Grid
        self.fType = velocity_grid.fType
        self.iType = velocity_grid.iType
        self.dim = velocity_grid.dim
        self.shape = velocity_grid.shape

        b = velocity_grid.b
        gcd = np_gcd(species.mass)
        # Todo the conversion is unnecessary, only for PyCharm
        m = np.array(species.mass // gcd)
        m_min = np.amin(m)
        # The lightest specimen have the sparsest grid
        self.d = velocity_grid.d * m_min/m
        self.d = np.array(self.d, dtype=self.fType)

        self.n = np.zeros((species.n, self.dim+1), dtype=self.iType)
        for _s in range(species.n):
            self.n[_s, 0:self.dim] = (b[:, 1] - b[:, 0]) / self.d[_s] + 1
            self.n[_s, -1] = self.n[_s, 0:self.dim].prod()

        self.index = np.zeros((species.n+1,), dtype=self.iType)
        for _s in range(species.n):
            self.index[_s+1] = self.index[_s] + self.n[_s, -1]

        self.b = np.zeros(shape=(species.n, self.dim, 2),
                          dtype=self.fType)
        for _s in range(species.n):
            prev_w = b[:, 1] - b[:, 0]
            curr_w = (self.n[_s, 0:self.dim] - np.ones(self.dim)) * self.d[_s]
            diff = prev_w - curr_w
            assert all(diff >= 0)
            self.b[_s, :, 0] = b[:, 0] + diff/2
            self.b[_s, :, 1] = b[:, 1] - diff/2

        if check_integrity:
            self.check_integrity()
        return

    def make_grid(self):
        sv_grid = np.zeros(shape=(self.index[-1], self.dim),
                           dtype=self.fType)
        for _s in range(self.n.shape[0]):
            _v = b_grd.Grid(self.dim,
                            self.n[_s, 0:self.dim],
                            self.d[_s],
                            shape=self.shape,
                            float_data_type=self.fType,
                            integer_data_type=self.iType,
                            offset=self.b[_s, :, 0])
            if self.shape is 'rectangular':
                _grid = _v.make_grid().reshape((_v.n[-1], _v.dim))
            else:
                print("ERROR - Unspecified Grid Shape")
                assert False
            sv_grid[self.index[_s]:self.index[_s+1], :] = _grid[:]
        return sv_grid

    def check_integrity(self):
        assert type(self.dim) is int
        assert self.dim in [1, 2, 3]
        s_n = self.b.shape[0]
        assert type(self.b) is np.ndarray
        assert self.b.dtype == self.fType
        assert self.b.shape == (s_n, self.dim, 2)
        for _s in range(s_n):
            for _d in range(self.dim):
                assert self.b[_s, _d, 1] - self.b[_s, _d, 0] > 0
        assert type(self.d) is np.ndarray
        assert self.d.dtype == self.fType
        assert all(self.d > 0)
        assert type(self.n) is np.ndarray
        assert self.n.shape == (s_n, self.dim+1)
        assert self.n.dtype == self.iType
        assert (self.n >= 2).all
        assert self.shape in b_grd.Grid.GRID_SHAPES
        assert type(self.fType) == type
        for _s in range(s_n):
            _width1 = self.b[_s, :, 1] - self.b[_s, :, 0]
            _width2 = ((self.n[_s, 0:self.dim] - np.ones(self.dim))
                       * self.d[_s])
            np.testing.assert_almost_equal(_width1, _width2)
        if self.shape is 'rectangular':
            for _s in range(s_n):
                assert self.n[_s, 0:self.dim].prod() == self.n[_s, -1]
        return

    def print(self):
        print("Dimension = {}".format(self.dim))
        print("Shape = {}".format(self.shape))
        print("Float Data Type = {}".format(self.fType))
        print("Integer Data Type = {}".format(self.iType))
        print('')
        for _s in range(self.n.shape[0]):
            print('Specimen_{}'.format(_s))
            print("Boundaries =")
            print(self.b[_s])
            print("Number of Total Grid Points = {}".format(self.n[_s, -1]))
            print("Grid Points per Dimension = "
                  "{}".format(self.n[_s, 0:self.dim]))
            print("Step Size = {}".format(self.d[_s]))
            print('')
