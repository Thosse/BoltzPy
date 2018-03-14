
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
        """:obj:`int`: Grid dimensionality."""
        return self._dim

    @property
    def n(self):
        """:obj:`~numpy.ndarray` of :obj:`int`:
        Number of grid points per dimension.
        Array of shape (:attr:`dim`,).
        """
        return self._n

    @property
    def size(self):
        """:obj:`int`:
        Total number of grid points.
        """
        return self._size

    @property
    def d(self):
        """:obj:`float`:
        Internal step size of the grid.

        Physical step size of uniform :class:`Grid` may differ
        (see :attr:`multi`, :class:`Configuration`).
        """
        return self._d

    @property
    def supported_forms(self):
        """:obj:`set` of :obj:`str`:
       Set of all currently supported geometric forms(:attr:`form`)
       for the :class:`Grid`.
       """
        supported_forms = {'rectangular'}
        return supported_forms

    @property
    def form(self):
        """:obj:`str`:
        Geometric form of :class:`Grid`,
        must be an element of :attr:`supported_forms`.
        """
        return self._form

    @property
    def G(self,):
        """:obj:`~np.ndarray` of :obj:`int`:
        Physical values of the :class:`Grid` points,
        given in multiples of :attr:`d`.

        G[i] denotes the physical value/coordinates of grid point i.
        Array of shape (:attr:`size`, :attr:`dim`)
        """
        return self._G

    @property
    def multi(self, ):
        """:obj:`int`:
        Determines the ratio of
        physical step size / internal step size (:attr:`d`).

        In several cases the Entries in :attr:`G` are multiplied by
        :attr:`multi`
        and :attr:`d` are divided by
        :attr:`multi`.
        This does not change the physical values, but enables a variety
        of features (see :attr:`Configuration.t`, :attr:`Configuration.sv`)
        """
        return self._multi

    @property
    def is_centered(self, ):
        """:obj:`bool`:
        True if Grid has been centered
        (i.e. :meth:`center` was called).
        False otherwise.
        """
        return self._flag_is_centered

    @property
    def boundaries(self):
        """:obj:`~numpy.ndarray` of :obj:`float`:
        Minimum and maximum physical values of all :class:`Grid` points.

        Array of shape (2, :attr:`dim`).
        """
        min_val = np.min(self.G, axis=0)
        max_val = np.max(self.G, axis=0)
        bound = np.array([min_val, max_val]) * self.d
        return bound

    #####################################
    #           Configuration           #
    #####################################
    def setup(self,
              dimension,
              number_of_points_per_dimension,
              step_size,
              form='rectangular',
              multi=1,
              ):
        """Constructs :obj:`Grid` object.

        Parameters
        ----------
        dimension : :obj:`int`
        number_of_points_per_dimension : :obj:`int`
        step_size : :obj:`int`
        form : :obj:`int`, optional
        multi : :obj:`int`, optional
        """
        self._dim = dimension
        self._n = np.array(number_of_points_per_dimension,
                           dtype=int)
        self._multi = multi
        self._d = float(step_size) / self._multi

        if form is 'rectangular':
            self._size = int(self.n.prod())
        else:
            message = "Unsupported Grid Form: {}".format(form)
            raise ValueError(message)
        self._form = form
        self._construct_grid()
        self.check_integrity()
        return

    def _construct_grid(self):
        """Call specialized method to construct :attr:`G`,
        based on :attr:`form`.
        """
        if self.form is 'rectangular':
            self._construct_rectangular_grid()
        else:
            message = "Unsupported Grid Form: {}".format(self.form)
            raise ValueError(message)
        return

    def _construct_rectangular_grid(self):
        """Construct a rectangular :attr:`G`."""
        assert self.form == 'rectangular'
        grid_shape = (self.size, self.dim)
        # Create list of 1D grids for each dimension
        list_of_1D_grids = [np.arange(0, self.n[i_d]*self._multi, self._multi)
                            for i_d in range(self.dim)]
        # Create mesh grid from 1D grids
        # Note that *[a,b,c] == a,b,c
        mesh_list = np.meshgrid(*list_of_1D_grids)
        grid = np.array(mesh_list, dtype=int)
        # bring meshgrid into desired order/structure
        if self.dim is 1:
            grid = np.array(grid.transpose((1, 0)))
        elif self.dim is 2:
            grid = np.array(grid.transpose((2, 1, 0)))
        elif self.dim is 3:
            grid = np.array(grid.transpose((2, 1, 3, 0)))
        else:
            message = "Error - Unsupported Grid dimension: " \
                      "{}".format(self.dim)
            raise AttributeError(message)
        assert grid.shape == tuple(self.n) + (self.dim,)
        self._G = grid.reshape(grid_shape)
        return

    def double_multiplicator(self):
        """Double the current :attr:`multi`.

        Also doubles all Entries in :attr:`G` and halves :attr:`d`.
        """
        self._G *= 2
        self._d /= 2
        self._multi *= 2
        return

    def halve_multiplicator(self):
        """Halve the current :attr:`multi`.

        Also halves all Entries in :attr:`G` and doubles :attr:`d`.
        """
        assert self.multi % 2 == 0, "All Entries in :attr:`G`" \
                                    "should be multiples of 2."
        assert all(self.G % 2 == 0), "All Entries in :attr:`G`" \
                                     "should be multiples of 2."
        self._G /= 2
        self._d *= 2
        self._multi /= 2
        return

    def center(self):
        """Centers the Grid
        (:attr:`G`)
        around zero and sets :attr:`is_centered` to :obj:`True`.
        """
        if self.is_centered:
            return
        alternation = self.multi*(self.n - 1)
        self.double_multiplicator()
        self._G -= alternation
        self._flag_is_centered = True
        return

    def revert_center(self):
        """Reverts the changes to (:attr:`G`) made in :meth:`center`
        and sets :attr:`is_centered` back to :obj:`False`
        """
        if not self.is_centered:
            return
        assert self.multi % 2 is 0, 'A centered grid should have an even' \
                                    ' multiplicator. ' \
                                    'The current multiplicator is {}' \
                                    ''. format(self.multi)
        alternation = (self.multi // 2) * (self.n - 1)
        self._G += alternation
        self._flag_is_centered = False
        self.halve_multiplicator()
        return

    def reshape(self, shape):
        """Changes the shape of (:attr:`G`)."""
        self._G = self._G.reshape(shape)
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check. Checks Integrity of all Attributes"""
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
        assert self.form in self.supported_forms
        if self.form is 'rectangular':
            assert self.n.prod() == self.size
        assert self.G.dtype == int
        assert self.G.ndim in [1, 2]
        if self.G.ndim is 1:
            assert self.dim is 1
            assert self.G.shape == (self.size,)
        else:
            assert self.G.shape == (self.size, self.dim)
        assert type(self.multi) is int
        assert self.multi >= 1
        assert (self.G % self.multi == 0).all
        assert type(self.is_centered) is bool
        if self.is_centered:
            assert self.multi % 2 is 0
        assert self.boundaries.shape == (2, self.dim)
        return

    def print(self, physical_grids=False):
        """Prints all Properties for Debugging Purposes"""
        print("Dimension = {}".format(self.dim))
        print("Geometric Form = {}".format(self.form))
        print("Number of Total Grid Points = {}".format(self.size))
        if self.dim is not 1:
            print("Grid Points per Dimension = {}".format(self.n))
        if self.multi is not 1:
            print("Multiplicator = {}".format(self.multi))
            print("Internal Step Size = {}".format(self.d))
        print("Physical Step Size = {}".format(self.d * self.multi))
        print('Is centered Grid = {}'.format(self.is_centered))
        print("Boundaries:\n{}".format(self.boundaries))
        if physical_grids:
            print('Physical Grid:')
            print(self.G*self.d)
        print('')

G = Grid()
G.setup(1, [5], 0.1)
G.print(True)

G.double_multiplicator()
G.print(True)
