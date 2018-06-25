
import boltzmann.constants as b_const

import numpy as np

from math import isclose


# Todo how to reference class attributes in numpy style?
# Todo break line in multi attribute docstring
# Todo add offset property?
class Grid:
    r"""Basic class for all Grids.

    Notes
    -----
        Note that changing :attr:`~Grid.multi`
        does not change the :attr:`spacing`
        or physical values of the :obj:`Grid`.
        It does change the values of
        :attr:`~Grid.d` and :attr:`~Grid.iG`
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
        Thus all values in :attr:`iG` are multiples of :attr:`multi`.
    iG : :obj:`np.ndarray` [:obj:`int`]
        The *integer Grid* iG. It describes the
        position/physical values (:attr:`pG`)
        of all :class:`Grid` points.
        All entries are factors, such that
        :math:`pG = iG \cdot d`.
        Array of shape (:attr:`size`, :attr:`dim`).
    """
    def __init__(self,
                 grid_form=None,
                 grid_dimension=None,
                 grid_shape=None,
                 grid_spacing=None,
                 grid_multiplicator=1,
                 grid_is_centered=False):
        self.check_parameters(grid_form=grid_form,
                              grid_dimension=grid_dimension,
                              grid_shape=grid_shape,
                              grid_spacing=grid_spacing,
                              # Todo giving in svGrids.setup a Grid.multi=2
                              # Todo leads to serious errors in Collisions
                              #grid_multiplicator=grid_multiplicator,
                              grid_is_centered=grid_is_centered)
        self.form = grid_form
        self.dim = grid_dimension
        if grid_shape is not None:
            self.n = np.array(grid_shape, dtype=int)
        else:
            self.n = None
        self.size = None    # calculated in setup()
        if grid_spacing is not None:
            self.d = grid_spacing / grid_multiplicator
        else:
            self.d = None
        self.multi = grid_multiplicator
        self.iG = None   # generated in setup()
        self.setup(grid_is_centered)
        return

    #####################################
    #           Properties              #
    #####################################
    @property
    def spacing(self):
        """:obj:`float` :
        Denotes the (physical) distance between two :class:`Grid` points.
        It holds
        :math:`spacing = d \cdot multi`.
         """
        return self.d * self.multi

    @property
    def pG(self):
        """:obj:`np.ndarray` [:obj:`float`] :
        Construct the *physical Grid* (**time intense!**).

            The physical Grid pG denotes the physical values of
            all :class:`Grid` points.

                :math:`pG := iG \cdot d`

            Array of shape (:attr:`size`, :attr:`dim`).
         """
        return self.iG * self.d

    @property
    def boundaries(self):
        """ :obj:`~numpy.ndarray` of :obj:`float`:
        Denotes the minimum and maximum physical values
        / position over all :class:`Grid` points
        in array of shape (2, :attr:`dim`).
        """
        # in uninitialized Grids: Min/Max operation raises Errors
        assert self.iG is not None, "The Grid is not initialized, yet." \
                                    "Boundaries can not be computed!"
        pG = self.pG
        min_val = np.min(pG, axis=0)
        max_val = np.max(pG, axis=0)
        bound = np.array([min_val, max_val])
        return bound

    @property
    def is_centered(self):
        """:obj:`bool` :
        True if the :class:`Grid` instance is centered around zero.

        Checks the first and last integer Grid points.
         """
        # Grid must be properly initialized first
        assert self.iG is not None, "The Grid is not initialized, yet."
        # Todo remove dirty dimension hack
        if len(self.iG.shape) == 1:
            tmp_zero = 0
        else:
            tmp_zero = np.zeros(self.dim)
        if np.array_equal(self.iG[0], tmp_zero):
            return False
        else:
            assert np.array_equal(self.iG[0], -self.iG[-1])
            return True

    #####################################
    #           Configuration           #
    #####################################
    def setup(self,
              grid_is_centered=False):
        """Automatically constructs
        :attr:`Grid.iG` and :attr:`Grid.size`.

        Parameters
        ----------
        grid_is_centered : :obj:`bool`, optional
            I set to :obj:`True` (non-default),
            the newly created Grid is :meth:`centralized <centralize>`.
        """
        necessary_params = [self.form, self.dim, self.n, self.d, self.multi]
        if any([val is None for val in necessary_params]):
            return
        self.check_integrity(False)

        if self.form == 'rectangular':
            self.size = int(self.n.prod())
            self._construct_rectangular_grid()
        else:
            message = "This Grid form is not implemented yet: " \
                      "{}".format(self.form)
            raise NotImplementedError(message)

        if grid_is_centered:
            self.centralize()
        self.check_integrity()
        return

    def _construct_rectangular_grid(self):
        """Construct a rectangular :attr:`iG`."""
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
        self.iG = grid.reshape(grid_shape)
        return

    # Todo remove -> replace by property setter
    def double_multiplicator(self):
        """Double the current :attr:`multi`.

        Also doubles all Entries in :attr:`iG` and halves :attr:`d`.
        """
        self.iG *= 2
        self.d /= 2
        self.multi *= 2
        return

    # Todo remove -> replace by property setter
    def halve_multiplicator(self):
        """Halve the current :attr:`multi`.

        Also halves all Entries in :attr:`iG` and doubles :attr:`d`.
        """
        assert isinstance(self.iG, np.ndarray)
        assert self.multi % 2 == 0, "All Entries in :attr:`iG`" \
                                    "should be multiples of 2."
        assert np.all(self.iG % 2 == 0), "All Entries in :attr:`iG`" \
                                         "should be multiples of 2."
        self.iG /= 2
        self.d *= 2
        self.multi /= 2
        return

    def centralize(self):
        """Shifts the integer Grid (:attr:`iG`)
        to be centered around zero.
        """
        assert isinstance(self.iG, np.ndarray)
        if self.is_centered:
            return
        # calculate shift
        shift = self.multi*(self.n - 1)
        # double the multiplicator
        self.double_multiplicator()
        # shift the integer Grid
        self.iG -= shift
        return

    def decentralize(self):
        """Reverts the changes made to :attr:`iG` in :meth:`centralize`.
        """
        assert isinstance(self.iG, np.ndarray)
        if not self.is_centered:
            return
        assert self.multi % 2 == 0, 'A centered grid must have an even ' \
                                    'multi. The current multi is ' \
                                    '{}'. format(self.multi)
        # calculate shift
        shift = (self.multi // 2) * (self.n - 1)
        # shift the integer Grid
        self.iG += shift
        # halve the multiplicator
        assert np.all(self.iG % 2 == 0)
        self.halve_multiplicator()
        return

# Todo remove -> fix dirty hack, for time grid shape
    def reshape(self, shape):
        """Changes the shape of :attr:`iG`."""
        self.iG = self.iG.reshape(shape)
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
        # Todo add is_setup property
        if self.iG is None:
            grid_is_centered = None
            boundaries = None
        else:
            grid_is_centered = self.is_centered
            boundaries = self.boundaries
        # todo properly assert boundaries
        self.check_parameters(grid_form=self.form,
                              grid_dimension=self.dim,
                              grid_shape=self.n,
                              grid_size=self.size,
                              grid_step_size=self.d,
                              grid_multiplicator=self.multi,
                              grid_spacing=self.spacing,
                              grid_array=self.iG,
                              grid_is_centered=grid_is_centered,
                              complete_check=complete_check)
        # Additional Conditions on instance:
        # parameter can be list, instance attributes must be np.ndarray
        assert self.n is None or isinstance(self.n, np.ndarray)
        # Todo remove dirty dimension hack -> should be uniformly true!
        # Todo check before, that boundaries can be calculated
        if self.dim is not None \
                and self.dim is not 1:
            if self.iG is not None and self.d is not None:
                assert self.boundaries.shape == (2, self.dim)
        return

    @staticmethod
    def check_parameters(grid_form=None,
                         grid_dimension=None,
                         grid_shape=None,
                         grid_size=None,
                         grid_step_size=None,
                         grid_multiplicator=None,
                         grid_spacing=None,
                         grid_array=None,
                         grid_is_centered=None,
                         complete_check=False):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        grid_form : :obj:`str`, optional
        grid_dimension : :obj:`int`, optional
        grid_shape : :obj:`list` [:obj:`int`], optional
        grid_size : :obj:`int`, optional
        grid_step_size : :obj:`float`, optional
        grid_multiplicator : :obj:`int`, optional
        grid_spacing : :obj:`float`, optional
        grid_array : :obj:`np.ndarray` [:obj:`int`], optional
        grid_is_centered : :obj:`bool`, optional
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

        if grid_shape is not None:
            if isinstance(grid_shape, list):
                grid_shape = np.array(grid_shape,
                                      dtype=int)
            assert isinstance(grid_shape, np.ndarray)
            assert grid_shape.dtype == int
            assert all(grid_shape >= 2)

        if grid_dimension is not None \
                and grid_shape is not None:
            assert grid_shape.shape == (grid_dimension,)

        if grid_size is not None:
            assert isinstance(grid_size, int)
            assert grid_size >= 2

        if grid_form is not None \
                and grid_shape is not None \
                and grid_size is not None:
            if grid_form == 'rectangular':
                assert grid_shape.prod() == grid_size

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

        if grid_dimension is not None and grid_size is not None \
                and grid_array is not None:
            # Todo remove dirty hack! This should be uniform
            if grid_array.ndim == 1:
                assert grid_dimension == 1
                assert grid_array.shape == (grid_size,)
            else:
                assert grid_array.shape == (grid_size, grid_dimension)

        if grid_multiplicator is not None and grid_array is not None:
            shifted_array = grid_array - grid_array[0]
            assert np.array_equal(shifted_array % grid_multiplicator,
                                  np.zeros(grid_array.shape, dtype=int))

        if grid_is_centered is not None:
            assert isinstance(grid_is_centered, bool)

        if grid_array is not None and grid_is_centered is not None:
            if grid_is_centered:
                assert np.array_equal(grid_array[0], -grid_array[-1])
            else:
                assert np.array_equal(grid_array[0],
                                      np.zeros(grid_array[0].shape))

        if grid_multiplicator is not None and grid_is_centered is True:
                assert grid_multiplicator % 2 == 0

        return

    def __str__(self, write_physical_grids=True):
        """Converts the instance to a string, describing all attributes."""
        description = ''
        description += "Dimension = {}\n".format(self.dim)
        description += "Geometric Form = {}\n".format(self.form)
        description += "Total Size = {}\n".format(self.size)
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
            description += self.pG.__str__().replace('\n', '\n\t')
        return description
