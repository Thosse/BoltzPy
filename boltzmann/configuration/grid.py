import numpy as np
import math


def get_close_int(real_number, precision=1e-6):
    close_int = int(math.floor(real_number))
    if math.fabs(close_int - real_number) < precision:
        return close_int
    elif math.fabs(close_int+1 - real_number) < precision:
        return close_int + 1
    else:
        assert False, "Number {} is not close to an integer".format(real_number)


class Grid:
    """Basic class for Positional-, and Time-Grids.

    .. todo::
        - Check offset for correctness and sanity,
          and find better solution for sv-grids, if possible
        - Todo Add unit tests
        - Todo Add Circular shape
        - add rotation of grid (useful for velocities)
        - Enable non-uniform grids?
        - Todo Add adaptive Grids

    Attributes
    ----------
    dim : iType
        Grid dimensionality.
    b : np.ndarray(fType)
        Denotes [minimum, maximum] physical values in the grid.
        Array of shape=(dim,2) and dtype=fType
    d : fType
        Step size of the grid.
    n : np.ndarray(iType)
        n[0:dim] denotes the number of grid points per dimension.
        n[-1] denotes the total number of grid points.
        Array of shape=(dim+1,) and dType=iType.
    G : np.ndarray
        The physical grid.
        G[i] denotes the physical coordinates of the i-th grid point.
        Array of shape(n[-1], dim) and dtype=fType.
    shape : str
        Shape of Grid, can only be rectangular so far.
    offset : np.ndarray(fType)
        Only to be used for V.Grids!
        Shifts Grid to be centered around offset.
        If offset is omitted(==None), then Grid is centered around Zero.
    fType : np.dtype
        Determines data type of floats.
    iType : np.dtype
        Determines data type of integers.

    """
    GRID_SHAPES = ['rectangular', 'not_initialized']

    def __init__(self,
                 dimension,
                 number_of_points_per_dimension,
                 step_size,
                 shape='rectangular',
                 float_data_type=np.float32,
                 integer_data_type=np.int32,
                 offset=None,
                 check_integrity=True,
                 create_grid=True):
        self.fType = float_data_type
        self.iType = integer_data_type
        self.dim = dimension
        self.n = np.zeros(shape=(self.dim + 1,), dtype=self.iType)
        self.n[0:self.dim] = number_of_points_per_dimension
        if shape is 'rectangular':
            self.n[-1] = self.n[0:self.dim].prod()
        elif shape is 'not_initialized':
            self.n[-1] = 0
        else:
            print('ERROR - unsupported Grid shape')
            assert False
        self.d = float_data_type(step_size)
        self.shape = shape
        if offset is None:
            self.offset = np.zeros(shape=(self.dim,), dtype=self.fType)
        else:
            if type(offset) is list:
                assert len(offset) is self.dim
            if type(offset) is np.ndarray:
                assert offset.shape == (self.dim,)
            self.offset = np.array(offset, dtype=self.fType)
        self.b = np.zeros(shape=(self.dim, 2), dtype=self.fType)
        self.b[:, 0] = self.offset
        self.b[:, 1] = [self.offset[i] + (self.n[i]-1)*self.d
                        for i in range(self.dim)]
        if create_grid:
            self.G = self.make_grid()
        else:
            self.G = np.zeros((0,), dtype=self.fType)
        if check_integrity:
            self.check_integrity()

    def check_integrity(self):
        assert type(self.dim) is int
        assert self.dim in [1, 2, 3]
        assert type(self.b) is np.ndarray
        assert self.b.dtype == self.fType
        assert self.b.shape == (self.dim, 2)
        for i_d in range(self.dim):
            assert self.b[i_d, 1] - self.b[i_d, 0] > 0
        assert type(self.d) is self.fType
        assert self.d > 0
        assert type(self.n) is np.ndarray
        assert self.n.dtype == self.iType
        assert (self.n >= 2).all
        assert self.shape in Grid.GRID_SHAPES
        assert type(self.fType) == type
        _width1 = self.b[:, 1] - self.b[:, 0]
        _width2 = (self.n[0:self.dim] - np.ones((self.dim,))) * self.d
        np.testing.assert_almost_equal(_width1, _width2)
        if self.shape is 'rectangular':
            assert self.n[0:self.dim].prod() == self.n[-1]
        assert self.G.dtype == self.fType
        assert self.G.shape == (self.n[-1], self.dim)
        # Todo check boundaries
        return

    def __make_rectangular_grid(self):
        assert self.shape == 'rectangular'
        _grid_size = (self.n[-1], self.dim)
        # Create list of 1D grids for each dimension
        _list_of_1D_grids = [np.linspace(self.b[i_d, 0],
                                         self.b[i_d, 1],
                                         self.n[i_d])
                             for i_d in range(self.dim)]
        # Create mesh grid from 1D grids
        # Note that *[a,b,c] == a,b,c
        _mesh_list = np.meshgrid(*_list_of_1D_grids)
        grid = np.array(_mesh_list, dtype=self.fType)
        # bring meshgrid into desired order/structure
        if self.dim is 2:
            grid = np.array(grid.transpose((2, 1, 0)))
        elif self.dim is 3:
            grid = np.array(grid.transpose((2, 1, 3, 0)))
        elif self.dim is not 1:
            print("Error")
            assert False
        grid = grid.reshape(_grid_size)
        return grid

    def make_grid(self):
        if self.shape is 'rectangular':
            return self.__make_rectangular_grid()
        else:
            print("ERROR - Unspecified Grid Shape")
            assert False

    def print(self,
              physical_grids=False):
        print("Dimension = {}".format(self.dim))
        print("Boundaries = \n{}".format(self.b))
        print("Number of Total Grid Points = {}".format(self.n[-1]))
        if self.dim is not 1:
            print("Grid Points per Dimension = {}".format(self.n[0:self.dim]))
        print("Step Size = {}".format(self.d))
        print("Shape = {}".format(self.shape))
        print("Float Data Type = {}".format(self.fType))
        print("Integer Data Type = {}".format(self.iType))
        if physical_grids:
            print('Physical Grid:')
            print(self.G)
        print('')

    # def __compute_n(self):
    #     assert self.d > 0
    #     _l = self.b[:, 1] - self.b[:, 0]
    #     _zero = np.zeros((self.dim,), dtype=self.fType)
    #     assert all(_l > _zero)
    #     n = np.zeros((self.dim+1,), dtype=self.iType)
    #     # Calculate number of grid points per dimension
    #     # Todo Catch Error, if boundaries and d don't match
    #     n[0:self.dim] = [get_close_int(i/self.d) + 1 for i in _l]
    #     # calculate total number of grid points
    #     if self.shape is 'rectangular':
    #         n[-1] = n[0:self.dim].prod()
    #     else:
    #         # Todo proper exception throwing
    #         print('ERROR')
    #         assert False
    #     return n
