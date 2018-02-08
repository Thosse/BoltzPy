import numpy as np
import math


def np_gcd(array_of_ints):
    assert array_of_ints.dtype == np.int32
    gcd = array_of_ints[0]
    for new_number in array_of_ints[1:]:
        gcd = math.gcd(gcd, new_number)
    return gcd


def get_close_int(real_number, precision=1e-6):
    close_int = int(math.floor(real_number))
    if math.fabs(close_int - real_number) < precision:
        return close_int
    elif math.fabs(close_int+1 - real_number) < precision:
        return close_int + 1
    else:
        assert False, "Number {} is not close to an integer".format(real_number)


#############################################################################
#                 Creation of Space and Velocity Grids/Arrays               #
#############################################################################
class Grid:
    """Basic class for Positional-, Time- or Velocity-Grids.

    .. todo::
        - Todo Add unit tests
        - Todo Add "check for consistency" method
        - Todo Add Circular shape
        - add rotation of grid (useful for velocities)
        - Enable non-uniform grids?
        - Todo Add adaptive Grids

    Attributes
    ----------
    dim : int
        Grid dimensionality.
    b : :obj:'np.ndarray'
        Denotes [minimum, maximum] physical values in the grid.
        Array of shape==(dim,2) and dtype=float
    d : fType
        Step size of the grid.
    n : np.ndarray(int)
        n[0:dim] denotes the number of grid points per dimension.
        n[-1] denotes the total number of grid points.
        Array of shape==(dim+1,)
    shape : str
        Shape of Grid can only be rectangular so far
    fType : np.dtype
        Determines data type of floats.
    iType : np.dtype
        Determines data type of integers.

    """
    GRID_SHAPES = ['rectangular']
    DATA_TYPES = [np.float16, np.float32, np.float64]

    def __init__(self,
                 dimension,
                 boundaries,
                 delta,
                 shape='rectangular',
                 float_data_type=np.float32,
                 integer_data_type=np.int32):
        self.dim = dimension
        self.b = np.array(boundaries, dtype=float_data_type)
        self.d = float_data_type(delta)
        self.shape = shape
        self.fType = float_data_type
        self.iType = integer_data_type
        # Todo: do this without empty __init__s
        self.n = np.zeros(shape=(self.dim+1,), dtype=self.iType)
        if self.shape is 'rectangular':
            self.n = self.__compute_n()
            # Todo: Update boundaries to new value n*d?
            Grid.check_integrity(self)
        else:
            print('Empty initialization of Grid')

    def __compute_n(self):
        assert self.d > 0
        _l = self.b[:, 1] - self.b[:, 0]
        _zero = np.zeros((self.dim,), dtype=self.fType)
        assert all(_l > _zero)
        n = np.zeros((self.dim+1,), dtype=self.iType)
        # Calculate number of grid points per dimension
        # Todo Catch Error, if boundaries and d don't match
        n[0:self.dim] = [get_close_int(i/self.d) + 1 for i in _l]
        # calculate total number of grid points
        if self.shape is 'rectangular':
            n[-1] = n[0:self.dim].prod()
        else:
            # Todo proper exception throwing
            print('ERROR')
            assert False
        return n

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
        # Todo make this assertion work
        # assert type(self.fType) in Grid.DATA_TYPES
        assert (self.n == self.__compute_n()).all
        if self.shape is 'rectangular':
            assert self.n[0:self.dim].prod() == self.n[-1]
            _width1 = self.b[:, 1] - self.b[:, 0]
            _width2 = (self.n[0:self.dim] - np.ones((self.dim,))) * self.d
            np.testing.assert_almost_equal(_width1, _width2)

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

    def print(self):
        print("Dimension = {}".format(self.dim))
        print("Boundaries = \n{}".format(self.b))
        print("Number of Total Grid Points = {}".format(self.n[-1]))
        print("Grid Points per Dimension = {}".format(self.n[0:self.dim]))
        print("Step Size = {}".format(self.d))
        print("Shape = {}".format(self.shape))
        print("Float Data Type = {}".format(self.fType))
        print("Integer Data Type = {}".format(self.iType))
        print("")
