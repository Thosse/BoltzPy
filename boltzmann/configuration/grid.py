
import numpy as np


class Grid:
    """Basic class for Positional-Space or Time-Space Grids.

    .. todo::
        - Todo Add unit tests
        - Todo Add Circular shape
        - add rotation of grid (useful for velocities)
        - Enable non-uniform/adaptive Grids
          (see :class:`~boltzmann.calculation.Calculation`)

    """
    # Todo move into config property
    GRID_FORMS = ['rectangular']

    def __init__(self):
        self._dim = 0
        self._n = np.zeros(shape=(self.dim,), dtype=int)
        self._size = 0
        self._d = 0.0
        self._form = ''
        self._G = np.zeros(shape=(self.size, self.dim), dtype=int)
        self._multi = 1
        self._flag_is_centered = False

    @property
    def dim(self):
        """:obj:`int`: Grid dimensionality"""
        return self._dim

    @property
    def n(self):
        """:obj:`array` of :obj:`int`:
        Number of grid points per dimension.
        Array of shape=(dim,)."""
        return self._n

    @property
    def size(self):
        """:obj:`int`:
        Total number of grid elements."""
        return self._size

    @property
    def d(self):
        """:obj:`float`:
        Step size of the grid."""
        return self._d

    @property
    def form(self):
        """:obj:`str`:
        Geometric form of Grid.

        Can only be rectangular so far."""
        return self._form

    # Todo edit docstring
    @property
    def G(self,):
        """:obj:`~np.ndarray` of :obj:`int`:
        The actual grid, given in multiples of
        :attr:`d`.

        G[i] denotes the physical coordinates of the i-th grid point.
        Array of shape(size, dim) and dtype=fType."""
        return self._G

    # Todo edit docstring
    @property
    def multi(self, ):
        """:obj:`int`:
        Multiplicator, that determines the ratio of
        'actual step size' / :attr:`d`.
        In several cases the Entries in G are multiplied by
        :attr:`multi`
        and :attr:`d` are divided by
        :attr:`multi`.
        This does not change the physical Values, but enables a variety
        of features( Currently: several calculation steps per write,
        simple centralizing ov velocity grids)"""
        return self._multi

    @property
    def is_centered(self, ):
        """:obj:`bool`:
        True if Grid has been centered
        (i.e. :meth:`center` was called).
        False otherwise."""
        return self._flag_is_centered

    @property
    def boundaries(self):
        min_val = np.min(self.G, axis=0)
        max_val = np.max(self.G, axis=0)
        bound = np.array([min_val, max_val]) * self.d
        if self.dim is 2:
            bound = bound.transpose()
        elif self.dim is not 1:
            assert False, 'This is not tested for 3d'
            # Todo figure out transpose order
        return bound

    #####################################
    #           Configuration           #
    #####################################
    def setup(self,
              dimension,
              number_of_points_per_dimension,
              step_size,
              form='rectangular',
              create_grid=True,
              multi=1):
        self._dim = dimension
        self._n = np.array(number_of_points_per_dimension,
                           dtype=int)
        self._d = float(step_size)

        if form is 'rectangular':
            self._size = int(self.n.prod())
        else:
            print('ERROR - unsupported Grid shape')
            # Todo throw exception
            assert False
        self._form = form

        if create_grid:
            self._create_grid()
        else:
            self._G = np.zeros((0,), dtype=int)

        if multi is not 1:
            self.change_multiplicator(multi)

        self.check_integrity()
        return

    def _create_grid(self):
        """Create Grid, based on :attr:`form`"""
        if self.form is 'rectangular':
            self._create_rectangular_grid()
        else:
            print("ERROR - Unspecified Grid Shape")
            assert False

    def _create_rectangular_grid(self):
        assert self.form == 'rectangular'
        grid_shape = (self.size, self.dim)
        # Create list of 1D grids for each dimension
        list_of_1D_grids = [np.arange(0, self.n[i_d])
                            for i_d in range(self.dim)]
        # Create mesh grid from 1D grids
        # Note that *[a,b,c] == a,b,c
        mesh_list = np.meshgrid(*list_of_1D_grids)
        grid = np.array(mesh_list, dtype=int)
        # bring meshgrid into desired order/structure
        if self.dim is 2:
            grid = np.array(grid.transpose((2, 1, 0)))
        elif self.dim is 3:
            grid = np.array(grid.transpose((2, 1, 3, 0)))
        elif self.dim is not 1:
            print("Error - Unsupported Grid dimension: "
                  "{}".format(self.dim))
            # Todo throw exception
            assert False
        self._G = grid.reshape(grid_shape)
        return

    # Todo edit behavior, check need to revert center
    # Todo doubling or halving should be doable without revert center + center
    def change_multiplicator(self,
                             new_multi):
        assert type(new_multi) is int, 'Unexpected type of ' \
                                       'multiplicator: {}' \
                                       ''.format(type(new_multi))
        assert new_multi > 1, 'Multiplicator mus be positive:' \
                              'submitted Value: {}' \
                              ''.format(new_multi)

        # Centering can cause conflicts with the multiplicator
        if self.is_centered:
            self.revert_center()
            self.change_multiplicator(new_multi)
            self.center()
            return
        else:
            assert (self.G % self.multi == 0).all, 'The current, uncentered ' \
                                                   'Grid is not ' \
                                                   'a multiple of its multi.'
        self._G //= self.multi
        self._d *= self.multi
        self._G *= new_multi
        self._d /= new_multi
        self._multi = new_multi
        return

    def center(self):
        """Center the Grid
        (:attr:`G`)
        around zero.
        """
        # Todo only increase multiplier, if necessary
        if self.is_centered:
            return
        alternation = self.multi*(self.n - 1)
        self.change_multiplicator(2*self.multi)
        self._G -= alternation
        self._flag_is_centered = True
        return

    def revert_center(self):
        if not self.is_centered:
            return
        assert self.multi % 2 is 0, 'A centered grid should have an even' \
                                    ' multiplicator. ' \
                                    'The current multiplicator is {}' \
                                    ''. format(self.multi)
        alternation = (self.multi // 2) * (self.n - 1)
        self._G += alternation
        self._flag_is_centered = False
        self.change_multiplicator(self.multi // 2)
        return

    def reshape(self, shape):
        """Changes the shape of the Grid
        (:attr:`G`).
        """
        self._G = self._G.reshape(shape)
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check"""
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
        assert self.form in Grid.GRID_FORMS
        if self.form is 'rectangular':
            assert self.n.prod() == self.size
        assert self.G.dtype == int
        # Todo Exception for t-Grid - make this streamlined
        if self.dim == 1:
            assert (self.G.shape == (self.size, self.dim)
                    or self.G.shape == (self.size,))
        else:
            assert self.G.shape == (self.size, self.dim)
        assert type(self.multi) is int
        assert self.multi >= 1
        assert (self.G % self.multi == 0).all
        assert type(self.is_centered) is bool
        if self.is_centered:
            assert self.multi % 2 is 0
        return

    def print(self, physical_grids=False):
        """Prints all Properties for Debugging Purposes"""
        print("Dimension = {}".format(self.dim))
        print("Geometric Form = {}".format(self.form))
        print("Number of Total Grid Points = {}".format(self.size))
        if self.dim is not 1:
            print("Grid Points per Dimension = {}".format(self.n))
        print("Step Size = {}".format(self.d * self.multi))
        if self.multi is not 1:
            print("Multiplicator = {}".format(self.multi))
        print('Is centered Grid = {}'.format(self.is_centered))
        print("Boundaries:\n{}".format(Grid.boundaries))
        if physical_grids:
            print('Physical Grid:')
            print(self.G*self.d)
        print('')
