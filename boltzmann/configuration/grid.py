
import boltzmann.constants as b_const

import numpy as np


# Todo add property/attribute pd - Physical step size?
# Todo add check parameters
# Todo break line in multi attribute docstring
class Grid:
    """Basic class for Positional-Space or Time-Space Grids.

    Notes
    -----
    Note that changing :attr:`Grid.multi`
    does not change the physical step size
    or physical values of the :obj:`Grid`,
    but allows features like adaptive Positional-Grids, or.

    Attributes
    ----------
    form : :obj:`str`
        Geometric form of the :class:`Grid`.
        Must be an element of
        :const:`~boltzmann.constants.SUPP_GRID_FORMS`.
    dim : :obj:`int`
        The :obj:`Grid` dimensionality. Must be in
        :const:`~boltzmann.constants.SUPP_GRID_DIMENSIONS`.
    n : :obj:`~numpy.ndarray` of :obj:`int`
        Number of :obj:`Grid` points per dimension.
        Array of shape (:attr:`dim`,).
    size : :obj:`int`
        Total number of :obj:`Grid` points.
    d : :obj:`float`
        Smallest possible step size of the :obj:`Grid`.
        Note that the physical step size of a uniform :obj:`Grid`
        is :attr:`multi` * :attr:`d`.
    multi : :obj:`int`
        Ratio of physical step size / :attr:`d`.
        Thus all values in :attr:`G` are multiples of :attr:`multi`.
    G : :obj:`~np.ndarray` of :obj:`int`
        G[i] denotes the physical value/coordinates of
        :class:`Grid` point i
        in multiples of :attr:`d`.

        Array of shape (:attr:`size`, :attr:`dim`)
    is_centered : :obj:`bool`
        True if :obj:`Grid` / :attr:`G` has been centered,
        i.e. :meth:`center` was called.
        False otherwise.

    """
    def __init__(self):
        self.form = ''
        self.dim = 0
        self.n = np.zeros(shape=(self.dim,), dtype=int)
        self.size = 0
        self.d = 0.0
        self.multi = 1
        self.G = np.zeros(shape=(self.size, self.dim), dtype=int)
        self.is_centered = False

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
        self.dim = dimension
        self.n = np.array(number_of_points_per_dimension,
                          dtype=int)
        self.multi = multi
        self.d = float(step_size) / self.multi

        if form == 'rectangular':
            self.size = int(self.n.prod())
        else:
            message = "Unsupported Grid Form: {}".format(form)
            raise ValueError(message)
        self.form = form
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
        list_of_1D_grids = [np.arange(0, self.n[i_d]*self.multi, self.multi)
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
        self.G = grid.reshape(grid_shape)
        return

    def double_multiplicator(self):
        """Double the current :attr:`multi`.

        Also doubles all Entries in :attr:`G` and halves :attr:`d`.
        """
        self.G *= 2
        self.d /= 2
        self.multi *= 2
        return

    def halve_multiplicator(self):
        """Halve the current :attr:`multi`.

        Also halves all Entries in :attr:`G` and doubles :attr:`d`.
        """
        assert self.multi % 2 == 0, "All Entries in :attr:`G`" \
                                    "should be multiples of 2."
        assert all(self.G % 2 == 0), "All Entries in :attr:`G`" \
                                     "should be multiples of 2."
        self.G /= 2
        self.d *= 2
        self.multi /= 2
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
        self.G -= alternation
        self.is_centered = True
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
        self.G += alternation
        self.is_centered = False
        self.halve_multiplicator()
        return

    def reshape(self, shape):
        """Changes the shape of :attr:`G`."""
        self.G = self.G.reshape(shape)
        return

    #####################################
    #           Utilities               #
    #####################################
    def boundaries(self):
        """Returns Minimum and maximum physical values
        over all :class:`Grid` points
        in array of shape (2, :attr:`dim`).

        Returns
        -------
        :obj:`~numpy.ndarray` of :obj:`float`

        """
        # in uninitialized Grids: Min/Max operation raises Errors
        if self.size == 0:
            return np.zeros((2, self.dim), dtype=float)
        min_val = np.min(self.G, axis=0)
        max_val = np.max(self.G, axis=0)
        bound = np.array([min_val, max_val]) * self.d
        return bound

    #####################################
    #           Serialization           #
    #####################################
    def load(self, hdf5_file):
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
        # read data from file
        dim = int(hdf5_file["Dimension"].value)
        n = hdf5_file["Points_per_Dimension"].value
        d = hdf5_file["Step_Size"].value
        form = hdf5_file["Form"].value
        multi = int(hdf5_file["Multiplicator"].value)
        # Todo Check Integrity
        # setup g
        self.setup(dim, n, d, form, multi)
        self.check_integrity()
        return

    def save(self, hdf5_file):
        """Writes the main attributes of the :obj:`Grid` instance
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
    def check_integrity(self, complete_check=True):
        """Sanity Check.
        Besides asserting all conditions in :meth:`check_parameters`
        it asserts the correct type of all attributes of the instance.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all attributes must be set (not None).
            If False, then unassigned attributes are ignored.
        """
        self.check_parameters(grid_form=self.form,
                              grid_dimension=self.dim,
                              number_of_grid_points_per_axis=self.n,
                              grid_size=self.size,
                              grid_step_size=self.d,
                              grid_multiplicator=self.multi,
                              grid_array=self.G,
                              flag_is_centered=self.is_centered,
                              complete_check=complete_check)
        # Additional Conditions on instance:
        # parameter can be list, instance attributes must be np.ndarray
        assert isinstance(self.n, np.ndarray)
        # Todo remove dirty dimension hack -> should be uniformly true!
        if self.dim is not 1:
            assert self.boundaries().shape == (2, self.dim)
        return

    @staticmethod
    def check_parameters(grid_form=None,
                         grid_dimension=None,
                         number_of_grid_points_per_axis=None,
                         grid_size=None,
                         grid_step_size=None,
                         grid_multiplicator=None,
                         grid_array=None,
                         flag_is_centered=None,
                         complete_check=False):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        grid_form : :obj:`str`, optional
        grid_dimension : :obj:`int`, optional
        number_of_grid_points_per_axis : :obj:`list` [:obj:`int`], optional
        grid_size : :obj:`int`, optional
        grid_step_size : :obj:`float`, optional
        grid_multiplicator : :obj:`int`, optional
        grid_array : :obj:`np.ndarray` [:obj:`int`], optional
        flag_is_centered : :obj:`bool`, optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        """
        # For complete check, assert that all parameters are assigned
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all([param is not None for param in locals().values()])

        # check all parameters, if set
        if grid_form is not None:
            assert isinstance(grid_form, str)
            assert grid_form in b_const.SUPP_GRID_FORMS

        if grid_dimension is not None:
            assert isinstance(grid_dimension, int)
            assert grid_dimension in b_const.SUPP_GRID_DIMENSIONS

        if number_of_grid_points_per_axis is not None:
            if isinstance(number_of_grid_points_per_axis, list):
                _n = np.array(number_of_grid_points_per_axis, dtype=int)
                number_of_grid_points_per_axis = _n
            assert isinstance(number_of_grid_points_per_axis, np.ndarray)
            assert number_of_grid_points_per_axis.dtype == int
            assert all(number_of_grid_points_per_axis >= 2)

        if grid_dimension is not None \
                and number_of_grid_points_per_axis is not None:
            assert number_of_grid_points_per_axis.shape == (grid_dimension,)

        if grid_size is not None:
            assert isinstance(grid_size, int)
            assert grid_size >= 2

        if grid_form is not None \
                and number_of_grid_points_per_axis is not None \
                and grid_size is not None:
            if grid_form == 'rectangular':
                assert number_of_grid_points_per_axis.prod() == grid_size

        if grid_step_size is not None:
            assert isinstance(grid_step_size, float)
            assert grid_step_size > 0

        if grid_multiplicator is not None:
            assert isinstance(grid_multiplicator, int)
            assert grid_multiplicator >= 1

        if grid_array is not None:
            assert isinstance(grid_array, np.ndarray)
            assert grid_array.dtype == int
            assert grid_array.ndim in b_const.SUPP_GRID_DIMENSIONS

        if grid_multiplicator is not None and grid_array is not None:
            assert (grid_array % grid_multiplicator == 0).all()

        if grid_dimension is not None and grid_size is not None \
                and grid_array is not None:
            # Todo remove dirty hack! This should be uniform
            if grid_array.ndim == 1:
                assert grid_dimension == 1
                assert grid_array.shape == (grid_size,)
            else:
                assert grid_array.shape == (grid_size, grid_dimension)

        if flag_is_centered is not None:
            assert isinstance(flag_is_centered, bool)

        if grid_multiplicator is not None and flag_is_centered is not None:
            if flag_is_centered:
                assert grid_multiplicator % 2 == 0
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
        description += '\t' + self.boundaries().__str__().replace('\n', '\n\t')
        if write_physical_grids:
            description += '\n'
            description += 'Physical Grid:\n\t'
            description += (self.G*self.d).__str__().replace('\n', '\n\t')
        return description
