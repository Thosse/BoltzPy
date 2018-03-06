import numpy as np


class Grid:
    """Basic class for Positional-, and Time-Grids.

    .. todo::
        - Todo Add unit tests
        - Todo Add Circular shape
        - add rotation of grid (useful for velocities)
        - Enable non-uniform/adaptive Grids
          (see :class:`~boltzmann.calculation.Calculation`)

    Attributes
    ----------
    dim : int
        Grid dimensionality.
    d : float
        Step size of the grid.
    n : np.ndarray(int)
        Number of grid points per dimension.
        Array of shape=(dim,) and dType=int.
    size : int
        Total number of grid points.
    G : np.ndarray(float)
        The physical grid.
        G[i] denotes the physical coordinates of the i-th grid point.
        Array of shape(size, dim) and dtype=fType.
    shape : str
        Geometric shape of Grid.
         Can only be rectangular so far.
    multi : int
        Multiplicator, that determines the ratio of
        'actual step size' / :attr:`~boltzmann.configuration.Grid.d`.
        In several cases the Entries in G are multiplied by
        :attr:`~boltzmann.configuration.Grid.multi`
        and :attr:`~boltzmann.configuration.Grid.d` are divided by
        :attr:`~boltzmann.configuration.Grid.multi`.
        This does not change the physical Values, but enables a variety
         of features( Currently: several calculation steps per write,
         simple centralizing ov velocity grids)
    is_centered : bool
        True if Grid has been centered, False otherwise.
    """
    GRID_SHAPES = ['rectangular']

    def __init__(self):
        self.dim = 0
        self.n = np.zeros(shape=(self.dim,), dtype=int)
        self.size = 0
        self.d = 0.0
        self.shape = ''
        self.G = np.zeros((0,), dtype=int)
        self.multi = 1
        self.is_centered = False

    def setup(self,
              dimension,
              number_of_points_per_dimension,
              step_size,
              shape='rectangular',
              create_grid=True,
              multiplicator=1,):
        self.dim = dimension
        self.n = np.array(number_of_points_per_dimension,
                          dtype=int)
        self.d = float(step_size)

        if shape is 'rectangular':
            self.size = int(self.n.prod())
        else:
            print('ERROR - unsupported Grid shape')
            assert False
        self.shape = shape

        if create_grid:
            self.create_grid()
        else:
            self.G = np.zeros((0,), dtype=int)

        if multiplicator is not 1:
            self.change_multiplicator(multiplicator)

        self.check_integrity()
        return

    def create_grid(self):
        if self.shape is 'rectangular':
            self._create_rectangular_grid()
        else:
            print("ERROR - Unspecified Grid Shape")
            assert False

    def _create_rectangular_grid(self):
        assert self.shape == 'rectangular'
        grid_size = (self.size, self.dim)
        # Create list of 1D grids for each dimension
        _list_of_1D_grids = [np.arange(0, self.n[i_d])
                             for i_d in range(self.dim)]
        # Create mesh grid from 1D grids
        # Note that *[a,b,c] == a,b,c
        _mesh_list = np.meshgrid(*_list_of_1D_grids)
        grid = np.array(_mesh_list, dtype=int)
        # bring meshgrid into desired order/structure
        if self.dim is 2:
            grid = np.array(grid.transpose((2, 1, 0)))
        elif self.dim is 3:
            grid = np.array(grid.transpose((2, 1, 3, 0)))
        elif self.dim is not 1:
            print("Error")
            assert False
        self.G = grid.reshape(grid_size)
        return

    # Todo move this into a property
    def change_multiplicator(self,
                             new_multi):
        # Centering can cause conflicts with the multiplicator
        if self.is_centered:
            self.revert_center()
            self.change_multiplicator(new_multi)
            self.center()
            return

        assert type(new_multi) is int, 'Unexpceted type of ' \
                                       'multiplicator: {}' \
                                       ''.format(type(new_multi))
        assert new_multi > 1, 'Multiplicator mus be positive:' \
                              'submitted Value: {}' \
                              ''.format(new_multi)
        assert (self.G % self.multi == 0).all, 'The current Grid is not ' \
                                               'a multiple of its multi.' \
                                               'This should not be possible!'
        self.G //= self.multi
        self.d *= self.multi
        self.G *= new_multi
        self.d /= new_multi
        self.multi = new_multi
        return

    def center(self):
        if self.is_centered:
            return
        alternation = self.multi*(self.n - 1)
        self.change_multiplicator(2*self.multi)
        self.G -= alternation
        self.is_centered = True
        return

    def revert_center(self):
        if not self.is_centered:
            return
        assert self.multi % 2 is 0, 'A centered grid should have an even' \
                                    ' multiplicator. ' \
                                    'The current multiplicator is {}' \
                                    ''. format(self.multi)
        alternation = (self.multi // 2) * (self.n - 1)
        self.G += alternation
        self.is_centered = False
        self.change_multiplicator(self.multi // 2)
        return

    @staticmethod
    def get_boundaries(grid, d):
        # Workaround for (X,)-shapes
        if len(grid.shape) is 1:
            dim = 1
        else:
            dim = grid.shape[-1]
        min_val = np.min(grid, axis=0)
        max_val = np.max(grid, axis=0)
        bound = np.array([min_val, max_val])*d
        if dim is 2:
            bound = bound.transpose()
        elif dim is not 1:
            assert False, 'This is not tested for 3d'
            # Todo figure out transpose order
        return bound

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        assert type(self.dim) is int
        assert self.dim in [1, 2, 3]
        assert type(self.d) is float
        assert self.d > 0
        assert type(self.n) is np.ndarray
        assert self.n.dtype == int
        assert self.n.shape == (self.dim,)
        assert all(self.n >= 2)
        assert type(self.size) is int
        assert self.size >= 2
        assert self.shape in Grid.GRID_SHAPES
        if self.shape is 'rectangular':
            assert self.n.prod() == self.size
        assert self.G.dtype == int
        assert self.G.shape == (self.size, self.dim)
        assert type(self.multi) is int
        assert self.multi >= 1
        assert (self.G % self.multi == 0).all
        assert type(self.is_centered) is bool
        if self.is_centered:
            assert self.multi % 2 is 0
        return

    def print(self,
              physical_grids=False):
        print("Dimension = {}".format(self.dim))
        print("Shape = {}".format(self.shape))
        print("Number of Total Grid Points = {}".format(self.size))
        if self.dim is not 1:
            print("Grid Points per Dimension = {}".format(self.n))
        print("Step Size = {}".format(self.d * self.multi))
        if self.multi is not 1:
            print("Multiplicator = {}".format(self.multi))
        print('Is centered Grid = {}'.format(self.is_centered))
        print("Boundaries:\n{}".format(Grid.get_boundaries(self.G, self.d)))
        if physical_grids:
            print('Physical Grid:')
            print(self.G*self.d)
        print('')
