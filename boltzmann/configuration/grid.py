
import numpy as np


class Grid:
    """Basic class for Positional-Space or Time-Space Grids."""
    def __init__(self):
        self._dim = 0
        self._form = ''
        self._multi = 1
        self._d = 0.0
        self._n = np.zeros(shape=(self.dim,), dtype=int)
        self._size = 0
        self._G = np.zeros(shape=(self.size, self.dim), dtype=int)
        self._flag_is_centered = False

    @property
    def dim(self):
        """:obj:`int`: :obj:`Grid` dimensionality."""
        return self._dim

    @property
    def form(self):
        """:obj:`str`:
        Geometric form of :class:`Grid`,
        must be an element of :attr:`supported_forms`.
        """
        return self._form

    @property
    def multi(self, ):
        """:obj:`int`:
        Determines the ratio of
        physical step size / internal step size (:attr:`d`).

        See :meth:`double_multiplicator`, :meth:`halve_multiplicator`.
        :attr:`multi` does not change the physical values,
        but enables a variety of features
        (see :attr:`Configuration.t`, :attr:`Configuration.sv`)
        """
        return self._multi

    @property
    def d(self):
        """:obj:`float` :
        Internal step size of the grid.

        Physical step size of uniform :class:`Grid` may differ
        (see :attr:`multi`, :class:`Configuration`).
        """
        return self._d

    @property
    def n(self):
        """:obj:`~numpy.ndarray` of :obj:`int` :
        Number of grid points per dimension.
        Array of shape (:attr:`dim`,).
        """
        return self._n

    @property
    def size(self):
        """:obj:`int` :
        Total number of grid points.
        """
        return self._size

    @property
    def G(self,):
        """:obj:`~np.ndarray` of :obj:`int` :
        Physical values of the :class:`Grid` points,
        given in multiples of :attr:`d`.

        G[i] denotes the physical value/coordinates of grid point i.
        Array of shape (:attr:`size`, :attr:`dim`)
        """
        return self._G

    @property
    def is_centered(self, ):
        """:obj:`bool` :
        True if Grid has been centered
        (i.e. :meth:`center` was called).
        False otherwise.
        """
        return self._flag_is_centered

    @property
    def supported_forms(self):
        """:obj:`set` of :obj:`str` :
       Set of all currently supported geometric forms(:attr:`form`)
       for the :class:`Grid`.
       """
        supported_forms = {'rectangular'}
        return supported_forms

    @property
    def boundaries(self):
        """:obj:`~numpy.ndarray` of :obj:`float` :
        Minimum and maximum physical values over all :class:`Grid` points.

        Array of shape (2, :attr:`dim`).
        """
        # in uninitialized Grids: Min/Max operation raises Errors
        if self.size == 0:
            return np.zeros((2, self.dim), dtype=float)
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
        number_of_points_per_dimension : :obj:`list` of :obj:`int`
        step_size : :obj:`float`
        form : :obj:`str`, optional
        multi : :obj:`int`, optional
        """
        self._dim = dimension
        self._n = np.array(number_of_points_per_dimension,
                           dtype=int)
        self._multi = multi
        self._d = float(step_size) / self._multi

        if form == 'rectangular':
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
        if self.form == 'rectangular':
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
        if self.dim == 1:
            grid = np.array(grid.transpose((1, 0)))
        elif self.dim == 2:
            grid = np.array(grid.transpose((2, 1, 0)))
        elif self.dim == 3:
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
        assert self.multi % 2 == 0, 'A centered grid should have an even' \
                                    ' multiplicator. ' \
                                    'The current multiplicator is {}' \
                                    ''. format(self.multi)
        alternation = (self.multi // 2) * (self.n - 1)
        self._G += alternation
        self._flag_is_centered = False
        self.halve_multiplicator()
        return

    def reshape(self, shape):
        """Changes the shape of :attr:`G`."""
        self._G = self._G.reshape(shape)
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_file):
        """Creates and Returns a :obj:`Grid` object,
        based on the parameters in the given file.

        Parameters
        ----------
        hdf5_file : h5py.File
            Opened HDF5 :obj:`Configuration` file.

        Returns
        -------
        :obj:`Grid`
        """
        grid = Grid()
        # read data from file
        dim = int(hdf5_file["Dimension"].value)
        n = hdf5_file["Points_per_Dimension"].value
        d = hdf5_file["Step_Size"].value
        form = hdf5_file["Form"].value
        multi = int(hdf5_file["Multiplicator"].value)
        # Todo Check Integrity
        # setup g
        grid.setup(dim, n, d, form, multi)
        grid.check_integrity()
        return grid

    def save(self, hdf5_file):
        """Writes the parameters of the :obj:`Species` object
        to the given file.

        Parameters
        ----------
        hdf5_file : h5py.File
            Opened HDF5 :obj:`Configuration` file.
        """
        self.check_integrity()
        # Clean State of Current group
        for key in hdf5_file.keys():
            del hdf5_file[key]
        # read data from file
        hdf5_file["Dimension"] = self.dim
        hdf5_file["Points_per_Dimension"] = self.n
        hdf5_file["Step_Size"] = self.d
        hdf5_file["Form"] = self.form
        hdf5_file["Multiplicator"] = self.multi
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
        if self.form == 'rectangular':
            assert self.n.prod() == self.size
        assert self.G.dtype == int
        assert self.G.ndim in [1, 2]
        if self.G.ndim == 1:
            assert self.dim == 1
            assert self.G.shape == (self.size,)
            assert self.boundaries.shape == (2,)
        else:
            assert self.G.shape == (self.size, self.dim)
            assert self.boundaries.shape == (2, self.dim)
        assert type(self.multi) is int
        assert self.multi >= 1
        assert (self.G % self.multi == 0).all
        assert type(self.is_centered) is bool
        if self.is_centered:
            assert self.multi % 2 == 0
        return

    def __str__(self, write_physical_grids=True):
        """Converts the instance to a string, describing all attributes."""
        description = ''
        description += "Dimension = {}\n".format(self.dim)
        description += "Geometric Form = {}\n".format(self.form)
        description += "Number of Total Grid Points = {}\n".format(self.size)
        if self.dim != 1:
            description += "Grid Points per Dimension = {}\n".format(self.n)
        if self.multi != 1:
            description += "Multiplicator = {}\n".format(self.multi)
            description += "Internal Step Size = {}\n".format(self.d)
        description += "Physical Step Size = {}\n".format(self.d * self.multi)
        description += 'Is centered Grid = {}\n'.format(self.is_centered)
        description += "Boundaries:\n"
        description += '\t' + self.boundaries.__str__().replace('\n', '\n\t')
        if write_physical_grids:
            description += '\n'
            description += 'Physical Grid:\n\t'
            description += (self.G*self.d).__str__().replace('\n', '\n\t')
        return description
