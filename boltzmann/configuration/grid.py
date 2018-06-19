
import boltzmann.constants as b_const

import numpy as np

from math import isclose


# Todo how to reference class attributes in numpy style?
# Todo break line in multi attribute docstring
# Todo rename attribute self.G -> array? iGrid?
# Todo create property pGrid?
class Grid:
    """Basic class for all Grids.

    Notes
    -----
        Note that changing :attr:`~Grid.multi`
        does not change the :attr:`spacing`
        or physical values of the :obj:`Grid`.
        It does change the values of
        :attr:`~Grid.d` and :attr:`~Grid.G`
        though.

        The purpose of :attr:`Grid.multi` is to allow features
        like adaptive (Positional- or Time-) Grids ,
        or write intervals for Time-Grids.

    Attributes
    ----------
    form : :obj:`str`
        Geometric form of the :class:`Grid`.
        Must be an element of
        :const:`~boltzmann.constants.SUPP_GRID_FORMS`.
    dim : :obj:`int`
        The :obj:`Grid` dimensionality. Must be in
        :const:`~boltzmann.constants.SUPP_GRID_DIMENSIONS`.
    n : :obj:`np.ndarray` [:obj:`int`]
        Number of :obj:`Grid` points per dimension.
        Array of shape (:attr:`dim`,).
    size : :obj:`int`
        Total number of :obj:`Grid` points.
    d : :obj:`float`
        Smallest possible step size of the :obj:`Grid`.
    multi : :obj:`int`
        Ratio of :attr:`spacing` / :attr:`d`.
        Thus all values in :attr:`G` are multiples of :attr:`multi`.
    G : :obj:`np.ndarray` [:obj:`int`]
        G[i] denotes the physical value/coordinates of
        :class:`Grid` point i
        in multiples of :attr:`d`.
        Array of shape (:attr:`size`, :attr:`dim`)
    is_centered : :obj:`bool`
        True if :attr:`Grid.G` has been centered,
        i.e. :meth:`center` was called.
        False otherwise.
    """
    def __init__(self,
                 grid_form=None,
                 grid_dimension=None,
                 grid_points_per_axis=None,
                 grid_spacing=None,
                 grid_multiplicator=1):
        self.check_parameters(grid_form=grid_form,
                              grid_dimension=grid_dimension,
                              grid_points_per_axis=grid_points_per_axis,
                              grid_spacing=grid_spacing,
                              grid_multiplicator=grid_multiplicator)
        self.form = grid_form
        self.dim = grid_dimension
        if grid_points_per_axis is not None:
            self.n = np.array(grid_points_per_axis, dtype=int)
        else:
            self.n = None
        self.size = None    # calculated in setup()
        if grid_spacing is not None:
            self.d = grid_spacing / grid_multiplicator
        else:
            self.d = None
        self.multi = grid_multiplicator
        self.G = None   # generated in setup()
        # Todo replace by property -> check boundaries[0]? -> problem: offset
        self.is_centered = False
        self.setup()
        return

    #####################################
    #           Properties              #
    #####################################
    @property
    def spacing(self):
        """Denotes the (physical) distance between two :class:`Grid` points.
        :attr:`spacing` = :attr:`d` * :attr:`multi`
         """
        return self.d * self.multi

    @property
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
    #           Configuration           #
    #####################################
    def setup(self):
        """Automatically constructs
        :attr:`~Grid.G` and :attr:`Grid.size`.
        """
        necessary_params = [self.form, self.dim, self.n, self.d, self.multi]
        if any([val is None for val in necessary_params]):
            return False
        self.check_integrity(False)

        if self.form == 'rectangular':
            self.size = int(self.n.prod())
            self._construct_rectangular_grid()
        else:
            message = "This Grid form is not implemented yet: " \
                      "{}".format(self.form)
            raise NotImplementedError(message)
        self.check_integrity()
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

    # Todo remove -> replace by property setter
    def double_multiplicator(self):
        """Double the current :attr:`multi`.

        Also doubles all Entries in :attr:`G` and halves :attr:`d`.
        """
        self.G *= 2
        self.d /= 2
        self.multi *= 2
        return

    # Todo remove -> replace by property setter
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

# Todo rename -> centralize
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

    # Todo rename -> decentralize
    def revert_center(self):
        """Reverts the changes to :attr:`G` made in :meth:`center`
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

# Todo remove -> fix dirty hack, for time grid shape
    def reshape(self, shape):
        """Changes the shape of :attr:`G`."""
        self.G = self.G.reshape(shape)
        return

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
        # read attributes from file
        try:
            self.form = hdf5_file["Form"].value
        except KeyError:
            self.form = None
        try:
            self.dim = int(hdf5_file["Dimension"].value)
        except KeyError:
            self.dim = None
        try:
            self.n = hdf5_file["Points_per_Dimension"].value
        except KeyError:
            self.n = None
        try:
            self.d = hdf5_file["Step_Size"].value
        except KeyError:
            self.d = None
        try:
            self.multi = int(hdf5_file["Multiplicator"].value)
        except KeyError:
            self.multi = None
        self.check_integrity(False)
        self.setup()
        return

    def save(self, hdf5_file):
        """Writes the main attributes of the :obj:`Grid` instance
        to the given file.

        Parameters
        ----------
        hdf5_file : h5py.File
            Opened HDF5 :obj:`Configuration` file.
        """
        self.check_integrity(False)
        # Clean State of Current group
        for key in hdf5_file.keys():
            del hdf5_file[key]
        # write all set attributes to file
        if self.dim is not None:
            hdf5_file["Dimension"] = self.dim
        if self.n is not None:
            hdf5_file["Points_per_Dimension"] = self.n
        if self.d is not None:
            hdf5_file["Step_Size"] = self.d
        if self.form is not None:
            hdf5_file["Form"] = self.form
        if self.multi is not None:
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
                              grid_points_per_axis=self.n,
                              grid_size=self.size,
                              grid_step_size=self.d,
                              grid_multiplicator=self.multi,
                              grid_spacing=self.spacing,
                              grid_array=self.G,
                              flag_is_centered=self.is_centered,
                              complete_check=complete_check)
        # Additional Conditions on instance:
        # parameter can be list, instance attributes must be np.ndarray
        assert self.n is None or isinstance(self.n, np.ndarray)
        # Todo remove dirty dimension hack -> should be uniformly true!
        # Todo check before, that boundaries can be calculated
        if self.dim is not None \
                and self.dim is not 1:
            if self.G is not None and self.d is not None:
                assert self.boundaries.shape == (2, self.dim)
        return

    @staticmethod
    def check_parameters(grid_form=None,
                         grid_dimension=None,
                         grid_points_per_axis=None,
                         grid_size=None,
                         grid_step_size=None,
                         grid_multiplicator=None,
                         grid_spacing=None,
                         grid_array=None,
                         flag_is_centered=None,
                         complete_check=False):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        grid_form : :obj:`str`, optional
        grid_dimension : :obj:`int`, optional
        grid_points_per_axis : :obj:`list` [:obj:`int`], optional
        grid_size : :obj:`int`, optional
        grid_step_size : :obj:`float`, optional
        grid_multiplicator : :obj:`int`, optional
        grid_spacing : :obj:`float`, optional
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

        if grid_points_per_axis is not None:
            if isinstance(grid_points_per_axis, list):
                grid_points_per_axis = np.array(grid_points_per_axis,
                                                dtype=int)
            assert isinstance(grid_points_per_axis, np.ndarray)
            assert grid_points_per_axis.dtype == int
            assert all(grid_points_per_axis >= 2)

        if grid_dimension is not None \
                and grid_points_per_axis is not None:
            assert grid_points_per_axis.shape == (grid_dimension,)

        if grid_size is not None:
            assert isinstance(grid_size, int)
            assert grid_size >= 2

        if grid_form is not None \
                and grid_points_per_axis is not None \
                and grid_size is not None:
            if grid_form == 'rectangular':
                assert grid_points_per_axis.prod() == grid_size

        if grid_step_size is not None:
            assert isinstance(grid_step_size, float)
            assert grid_step_size > 0

        if grid_multiplicator is not None:
            assert isinstance(grid_multiplicator, int)
            assert grid_multiplicator >= 1

        if grid_spacing is not None:
            assert isinstance(grid_spacing, float)
            assert grid_spacing > 0

        if grid_step_size is not None \
                and grid_multiplicator is not None \
                and grid_spacing is not None:
            assert isclose(grid_spacing, grid_multiplicator*grid_step_size)

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
        description += "Spacing = {}\n".format(self.spacing)
        description += 'Is centered Grid = {}\n'.format(self.is_centered)
        description += "Boundaries:\n"
        description += '\t' + self.boundaries.__str__().replace('\n', '\n\t')
        if write_physical_grids:
            description += '\n'
            description += 'Physical Grid:\n\t'
            description += (self.G*self.d).__str__().replace('\n', '\n\t')
        return description
