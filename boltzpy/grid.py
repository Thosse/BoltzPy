
import numpy as np
import h5py
from math import isclose

import boltzpy.constants as bp_c


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

    .. todo::
        - Add unit tests
        - Add Circular shape
        - Add rotation of grid (useful for velocities)
        - Enable non-uniform/adaptive Grids
          (see :class:`~boltzpy.computation.Calculation`)
        - Add Plotting-function to grids

    Attributes
    ----------
    form : :obj:`str`
        Geometric form of the :class:`Grid`.
        Must be an element of
        :const:`~boltzpy.constants.SUPP_GRID_FORMS`.
    dim : :obj:`int`
        The :obj:`Grid` dimensionality. Must be in
        :const:`~boltzpy.constants.SUPP_GRID_DIMENSIONS`.
    n : :obj:`~numpy.array` [:obj:`int`]
        Number of :obj:`Grid` points per dimension.
    d : :obj:`float`
        Smallest possible step size of the :obj:`Grid`.
    multi : :obj:`int`
        Ratio of :attr:`spacing` / :attr:`d`.
        Thus all values in :attr:`iG` are multiples of :attr:`multi`.
    iG : :obj:`~numpy.array` [:obj:`int`]
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
        self.check_parameters(form=grid_form,
                              dimension=grid_dimension,
                              shape=grid_shape,
                              spacing=grid_spacing,
                              # Todo giving in svGrids.setup a Grid.multi=2
                              # Todo leads to serious errors in Collisions
                              #grid_multiplicator=grid_multiplicator,
                              is_centered=grid_is_centered)
        self.form = grid_form
        self.dim = grid_dimension
        if grid_shape is not None:
            self.n = np.array(grid_shape, dtype=int)
        else:
            self.n = None
        if grid_spacing is not None:
            self.d = grid_spacing / grid_multiplicator
        else:
            self.d = None
        self.multi = grid_multiplicator
        # self.iG is generated in setup()
        self.iG = None
        self.setup(grid_is_centered)
        return

    #####################################
    #           Properties              #
    #####################################
    @property
    def size(self):
        """:obj:`int` :
        The total number of grid points.
        """
        if self.form is None:
            return None
        if self.form == 'rectangular':
            return int(self.n.prod())
        else:
            message = "This Grid form is not implemented yet: " \
                      "{}".format(self.form)
            raise NotImplementedError(message)

    @property
    def spacing(self):
        """:obj:`float` :
        The (physical) distance between two :class:`Grid` points.
        It holds :math:`spacing = d \cdot multi`.
        """
        try:
            return self.d * self.multi
        except TypeError:
            return None

    @property
    def pG(self):
        """:obj:`~numpy.array` [:obj:`float`] :
        Construct the *physical Grid* (**computationally heavy!**).

            The physical Grid pG denotes the physical values of
            all :class:`Grid` points.

                :math:`pG := iG \cdot d`

            Array of shape (:attr:`size`, :attr:`dim`).
         """
        try:
            return self.iG * self.d
        except TypeError:
            return None

    @property
    def boundaries(self):
        """ :obj:`~numpy.array` of :obj:`float`:
        Minimum and maximum physical values
        of all :class:`Grid` points
        in array of shape (2, :attr:`dim`).
        """
        # if Grid is not initialized -> None
        if self.iG is None:
            return None
        min_val = np.min(self.pG, axis=0)
        max_val = np.max(self.pG, axis=0)
        boundaries = np.array([min_val, max_val])
        return boundaries

    # Todo Move is_centered, centralize and decentralize to svgrid?
    # Todo      -> time or position grids don't need to be centered
    @property
    def is_centered(self):
        """:obj:`bool` :
        True if the :class:`Grid` instance is centered around zero.

        Checks the first and last integer Grid points.
         """
        try:
            is_not_centered = np.array_equal(self.iG[0], np.zeros(self.dim))
            if is_not_centered:
                return False
            else:
                assert np.array_equal(self.iG[0], -self.iG[-1])
                return True
        except TypeError:
            # iG is None
            assert self.iG is None
            return None

    @property
    def is_configured(self):
        """:obj:`bool` :
        True, if all necessary attributes of the instance are set.
        False Otherwise.
        """
        necessary_params = [self.form,
                            self.dim,
                            self.n,
                            self.d,
                            self.multi]
        if any([val is None for val in necessary_params]):
            return False
        else:
            return True

    @property
    def is_set_up(self):
        """:obj:`bool` :
        True, if the instance is completely set up and ready to call
        :meth:`~Simulation.run_computation`.
        False Otherwise.
        """
        return self.is_configured and self.iG is not None

    #####################################
    #           Configuration           #
    #####################################
    def setup(self,
              grid_is_centered=False):
        """Construct :attr:`Grid.iG` and :attr:`Grid.size`.

        Parameters
        ----------
        grid_is_centered : :obj:`bool`, optional
            I set to :obj:`True` (non-default),
            the newly created Grid is :meth:`centralized <centralize>`.
        """
        if not self.is_configured:
            return
        else:
            self.check_integrity(False)

        if self.form == 'rectangular':
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
    def double_multi(self):
        """Double the current :attr:`multi`.

        Also doubles all Entries in :attr:`iG` and halves :attr:`d`.
        """
        self.iG *= 2
        self.d /= 2
        self.multi *= 2
        return

    # Todo remove -> replace by property setter
    def halve_multi(self):
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
        """Shift the integer Grid (:attr:`iG`) to be centered around zero.
        """
        assert isinstance(self.iG, np.ndarray)
        if self.is_centered:
            return
        # calculate shift
        shift = self.multi*(self.n - 1)
        # double the multiplicator
        self.double_multi()
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
        self.halve_multi()
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`Grid` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Grid`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "Grid"
        self = Grid()

        # read attributes from file
        try:
            self.form = hdf5_group["Form"].value
        except KeyError:
            self.form = None
        try:
            self.dim = int(hdf5_group["Dimension"].value)
        except KeyError:
            self.dim = None
        try:
            self.n = hdf5_group["Points_per_Dimension"].value
        except KeyError:
            self.n = None
        try:
            self.d = hdf5_group["Step_Size"].value
        except KeyError:
            self.d = None
        try:
            self.multi = int(hdf5_group["Multiplicator"].value)
        except KeyError:
            self.multi = None

        self.check_integrity(False)
        self.setup()
        self.check_integrity(False)
        return self

    def save(self, hdf5_group):
        """Write the main parameters of the :obj:`Grid` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity(False)

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = "Grid"

        # write all set attributes to file
        if self.dim is not None:
            hdf5_group["Dimension"] = self.dim
        if self.n is not None:
            hdf5_group["Points_per_Dimension"] = self.n
        if self.d is not None:
            hdf5_group["Step_Size"] = self.d
        if self.form is not None:
            hdf5_group["Form"] = self.form
        if self.multi is not None:
            hdf5_group["Multiplicator"] = self.multi
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self, complete_check=True):
        """Sanity Check.

        Assert all conditions in :meth:`check_parameters`.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all attributes must be assigned (not None).
            If False, then unassigned attributes are ignored.
        """
        self.check_parameters(form=self.form,
                              dimension=self.dim,
                              shape=self.n,
                              size=self.size,
                              step_size=self.d,
                              multi=self.multi,
                              spacing=self.spacing,
                              idx_grid=self.iG,
                              boundaries=self.boundaries,
                              is_centered=self.is_centered,
                              complete_check=complete_check)
        return

    @staticmethod
    def check_parameters(form=None,
                         dimension=None,
                         shape=None,
                         size=None,
                         step_size=None,
                         multi=None,
                         spacing=None,
                         idx_grid=None,
                         boundaries=None,
                         is_centered=None,
                         complete_check=False):
        """Sanity Check.

        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        form : :obj:`str`, optional
        dimension : :obj:`int`, optional
        shape : :obj:`~numpy.array` [:obj:`int`], optional
        size : :obj:`int`, optional
        step_size : :obj:`float`, optional
        multi : :obj:`int`, optional
        spacing : :obj:`float`, optional
        idx_grid : :obj:`~numpy.array` [:obj:`int`], optional
        boundaries : :obj:`~numpy.array` [:obj:`float`], optional
        is_centered : :obj:`bool`, optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        """
        # For complete check, assert that all parameters are assigned
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all([param is not None for param in locals().values()])

        # check all parameters, if set
        if form is not None:
            assert isinstance(form, str)
            assert form in bp_c.SUPP_GRID_FORMS

        if dimension is not None:
            assert isinstance(dimension, int)
            assert dimension in bp_c.SUPP_GRID_DIMENSIONS

        if shape is not None:
            assert isinstance(shape, np.ndarray)
            assert shape.dtype == int
            assert all(shape >= 2)
            if dimension is not None:
                assert shape.shape == (dimension,)

        if size is not None:
            assert isinstance(size, int)
            assert size >= 2
            if form is not None and shape is not None:
                if form == 'rectangular':
                    assert shape.prod() == size
                else:
                    raise NotImplementedError

        if step_size is not None:
            assert isinstance(step_size, float)
            assert step_size > 0

        if multi is not None:
            assert isinstance(multi, int)
            assert multi >= 1

        if spacing is not None:
            assert isinstance(spacing, float)
            assert spacing > 0
            if step_size is not None and multi is not None:
                assert isclose(spacing, multi * step_size)

        if idx_grid is not None:
            assert isinstance(idx_grid, np.ndarray)
            assert idx_grid.dtype == int
            assert idx_grid.ndim in bp_c.SUPP_GRID_DIMENSIONS
            if dimension is not None and size is not None:
                assert idx_grid.shape == (size, dimension)
            if multi is not None:
                shifted_array = idx_grid - idx_grid[0]
                assert np.array_equal(shifted_array % multi,
                                      np.zeros(idx_grid.shape, dtype=int))

        if boundaries is not None:
            assert isinstance(boundaries, np.ndarray)
            assert boundaries.dtype == float
            assert boundaries.ndim == 2
            assert boundaries.shape[0] == 2
            if dimension is not None:
                assert boundaries.shape == (2, dimension)

        if is_centered is not None:
            assert isinstance(is_centered, bool)
            if is_centered and multi is not None:
                    assert multi % 2 == 0
            if idx_grid is not None:
                if is_centered:
                    assert np.array_equal(idx_grid[0], -idx_grid[-1])
                else:
                    assert np.array_equal(idx_grid[0],
                                          np.zeros(idx_grid[0].shape))
        return

    def __str__(self, write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance."""
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
