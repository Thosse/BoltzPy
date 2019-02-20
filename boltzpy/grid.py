
import numpy as np
import h5py
from math import isclose

import boltzpy.constants as bp_c


class Grid:
    r"""Basic class for all Grids.

    .. todo::
        - add index_spacing documentation
        - Add unit tests
        - Add grid.plot() method
        - Add form = "circular"
            - Create circular shape from rectangle, by cutting
            - construct collisions from rectangular grid, then adjust to circular grid
        - Add rotation of grid (useful for velocities)
        - Enable non-uniform/adaptive Grids
          (see :class:`~boltzpy.computation.Calculation`)

    Parameters
    ----------
    ndim : :obj:`int`
        The number of :obj:`Grid` dimensions.
        Must be in :const:`~boltzpy.constants.SUPP_GRID_DIMENSIONS`.
    shape : :obj:`~numpy.array` [:obj:`int`]
        Number of :obj:`Grid` points for each dimension.
        Array of shape (:attr:`dim`).
    form : :obj:`str`
        Geometric form of the :class:`Grid`.
        Must be an element of
        :const:`~boltzpy.constants.SUPP_GRID_FORMS`.
    physical_spacing : :obj:`float`
        Step size for the physical grid points.
    index_spacing : :obj:`int`, optional
        This allows
        centered velocity grids (without the zero),
        write-intervalls for time grids
        and possibly adaptive positional grids.
    is_centered : :obj:`bool`, optional
        True if the Grid should be centered around zero.


    Attributes
    ----------
    ndim : :obj:`int`
        The number of :obj:`Grid` dimensions.
    shape : :obj:`~numpy.array` [:obj:`int`]
        Number of :obj:`Grid` points for each dimension.
    form : :obj:`str`
        Geometric form of the :class:`Grid`.
    delta : :obj:`float`
        Smallest possible step size of the :obj:`Grid`.
    index_spacing : :obj:`int`, optional
        This allows
        centered velocity grids (without the zero),
        write-intervalls for time grids
        and possibly adaptive positional grids.
    is_centered : :obj:`float`
        True if the Grid should be centered around zero.
    iG : :obj:`~numpy.array` [:obj:`int`]
        The *integer Grid*  describes the
        physical values (:attr:`pG`)
        of all :class:`Grid` points
        in multiples of :attr:`delta`.
        Using integers allows precise computations,
        compared to floats.
    """
    def __init__(self,
                 ndim=None,
                 shape=None,
                 form=None,
                 physical_spacing=None,
                 index_spacing=2,
                 is_centered=False):
        self.ndim = ndim
        if shape is not None:
            self.shape = np.array(shape, dtype=int)
        else:
            self.shape = None
        self.form = form
        if physical_spacing is not None:
            self.delta = physical_spacing / index_spacing
        else:
            self.delta = None
        self.index_spacing = index_spacing
        self.is_centered = is_centered
        self.iG = None
        # set up self.iG, if all necessary parameters are given
        if self.is_configured:
            self.setup()
        return

    #####################################
    #           Properties              #
    #####################################
    @property
    def size(self):
        """:obj:`int` :
        The total number of grid points.
        """
        if self.iG is None:
            return None
        else:
            return self.iG[:, 0].size

    @property
    def physical_spacing(self):
        r""":obj:`float` :
        The physical distance between two grid points.

        It holds :math:`physical \_ spacing = delta \cdot index \_ spacing`.
        """
        if self.delta is None or self.index_spacing is None:
            return None
        else:
            return self.delta * self.index_spacing

    # Todo move into function, set parameter for matrix/tensor style?
    @property
    def pG(self):
        r""":obj:`~numpy.array` [:obj:`float`] :
        Construct the *physical Grid* (**computationally heavy!**).

        The physical Grid pG denotes the physical values of
        all :class:`Grid` points.

        :math:`pG := iG \cdot delta`

        Array of shape (:attr:`size`, :attr:`ndim`).
         """
        if self.iG is None or self.delta is None:
            return None
        else:
            return self.iG * self.delta

    @property
    def boundaries(self):
        """ :obj:`~numpy.array` of :obj:`float`:
        Minimum and maximum physical values
        of all :class:`Grid` points
        in array of shape (2, :attr:`dim`).
        """
        # if Grid is not initialized -> None
        if self.iG is None or self.delta is None:
            return None
        else:
            min_val = np.min(self.iG, axis=0)
            max_val = np.max(self.iG, axis=0)
            boundaries = self.delta * np.array([min_val, max_val])
            return boundaries

    @property
    def is_configured(self):
        """:obj:`bool` :
        True, if all necessary attributes of the instance are set.
        False Otherwise.
        """
        necessary_params = [self.ndim,
                            self.shape,
                            self.delta,
                            self.form,
                            self.index_spacing,
                            self.is_centered]
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
        # Todo check sefl.__dict__?
        return self.is_configured and self.iG is not None

    #####################################
    #           Configuration           #
    #####################################
    # Todo this should be simpler -> use np.mgrid?
    def setup(self):
        """Construct the index grid (:attr:`Grid.iG`)."""
        self.check_integrity(False)
        assert self.is_configured

        # Todo This process is too confusing
        # Create rectangular Grid first
        # Create list of axes (1D arrays)
        axes = [np.arange(0,
                          points_on_axis * self.index_spacing,
                          self.index_spacing)
                for points_on_axis in self.shape]
        # Create mesh grid from axes
        # Note that *[a,b,c] == a,b,c
        # it unpacks a list
        meshgrid = np.meshgrid(*axes)
        grid = np.array(meshgrid, dtype=int)
        # bring meshgrid into desired order/structure
        if self.ndim == 1:
            grid = np.array(grid.transpose((1, 0)))
        elif self.ndim == 2:
            grid = np.array(grid.transpose((2, 1, 0)))
        elif self.ndim == 3:
            grid = np.array(grid.transpose((2, 1, 3, 0)))
        else:
            msg = "Error - Unsupported Grid dimension: " \
                  "{}".format(self.ndim)
            raise AttributeError(msg)
        assert grid.shape == tuple(self.shape) + (self.ndim,)
        self.iG = grid.reshape((np.prod(self.shape),
                                self.ndim))

        # Cut into the desired geometric form
        if self.form == 'rectangular':
            pass
        elif self.form == 'circular':
            raise NotImplementedError
            # Todo just cut off all unwanted indices
        else:
            msg = "Error - Unsupported Grid form: " \
                  "{}".format(self.form)
            raise AttributeError(msg)

        # center the Grid if necessary
        if self.is_centered:
            self.centralize()
        self.check_integrity()
        return

    def centralize(self):
        """Shift the integer Grid (:attr:`iG`) to be centered around zero.
        """
        assert isinstance(self.iG, np.ndarray)
        assert self.index_spacing is not None
        double_shift = np.max(self.iG, axis=0) + np.min(self.iG, axis=0)
        if np.all(double_shift % 2 == 0):
            shift = double_shift // 2
            self.iG -= shift
        else:
            msg = "Even Grids can only be centralized, " \
                  "if the index_spacing is even. " \
                  "index_spacing = {}".format(self.index_spacing)
            raise AttributeError(msg)
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

        # read parameters from file
        params = dict()
        if "Dimensions" in hdf5_group.keys():
            params["ndim"] = int(hdf5_group["Dimensions"][()])
        if "Shape" in hdf5_group.keys():
            params["shape"] = hdf5_group["Shape"][()]
        if "Physical_Spacing" in hdf5_group.keys():
            params["physical_spacing"] = hdf5_group["Physical_Spacing"][()]
        if "Form" in hdf5_group.keys():
            params["form"] = hdf5_group["Form"][()]
        if "Index_Spacing" in hdf5_group.keys():
            params["index_spacing"] = int(hdf5_group["Index_Spacing"][()])
        else:
            params["index_spacing"] = None
        if "Is_Centered" in hdf5_group.keys():
            params["is_centered"] = bool(hdf5_group["Is_Centered"][()])
        else:
            params["is_centered"] = None

        self = Grid(**params)
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
        if self.ndim is not None:
            hdf5_group["Dimensions"] = self.ndim
        if self.shape is not None:
            hdf5_group["Shape"] = self.shape
        if self.physical_spacing is not None:
            hdf5_group["Physical_Spacing"] = self.physical_spacing
        if self.form is not None:
            hdf5_group["Form"] = self.form
        if self.index_spacing is not None:
            hdf5_group["Index_Spacing"] = self.index_spacing
        if self.is_centered is not None:
            hdf5_group["Is_Centered"] = self.is_centered

        # check that the class can be reconstructed from the save
        other = Grid.load(hdf5_group)
        # Todo Implement __eq__ method
        # assert self == other
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
        self.check_parameters(ndim=self.ndim,
                              shape=self.shape,
                              physical_spacing=self.physical_spacing,
                              form=self.form,
                              index_spacing=self.index_spacing,
                              is_centered=self.is_centered,
                              delta=self.delta,
                              size=self.size,
                              iG=self.iG,
                              pG=self.pG,
                              boundaries=self.boundaries,
                              complete_check=complete_check)
        return

    @staticmethod
    def check_parameters(ndim=None,
                         shape=None,
                         physical_spacing=None,
                         form=None,
                         index_spacing=None,
                         is_centered=None,
                         delta=None,
                         size=None,
                         iG=None,
                         pG=None,
                         boundaries=None,
                         complete_check=False):
        """Sanity Check.

        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        ndim : :obj:`int`, optional
        shape : :obj:`~numpy.array` [:obj:`int`], optional
        physical_spacing : :obj:`float`, optional
        form : :obj:`str`, optional
        index_spacing : :obj:`int`, optional
        is_centered : :obj:`bool`, optional
        delta : :obj:`float`, optional
        size : :obj:`int`, optional
        iG : :obj:`~numpy.array` [:obj:`int`], optional
        pG : :obj:`~numpy.array` [:obj:`float`], optional
        boundaries : :obj:`~numpy.array` [:obj:`float`], optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        """
        # For complete check, assert that all parameters are assigned
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all([param is not None for param in locals().values()])

        # check all parameters, if set
        if ndim is not None:
            assert isinstance(ndim, int)
            assert ndim in bp_c.SUPP_GRID_DIMENSIONS

        if shape is not None:
            assert isinstance(shape, np.ndarray)
            assert shape.dtype == int
            assert all(shape >= 2)

        if ndim is not None and shape is not None:
                assert shape.shape == (ndim,)

        if physical_spacing is not None:
            assert isinstance(physical_spacing, float)
            assert physical_spacing > 0

        if form is not None:
            assert isinstance(form, str)
            assert form in bp_c.SUPP_GRID_FORMS

        if index_spacing is not None:
            assert isinstance(index_spacing, int)
            assert index_spacing > 0

        if is_centered is not None:
            assert isinstance(is_centered, bool)

        if delta is not None:
            assert isinstance(delta, float)
            assert delta > 0

        if physical_spacing is not None \
                and index_spacing is not None \
                and delta is not None:
            assert isclose(physical_spacing,
                           index_spacing * delta)

        if size is not None:
            assert isinstance(size, int)
            assert size >= 1
            if form is 'rectangular' and shape is not None:
                assert shape.prod() == size

        if iG is not None:
            assert isinstance(iG, np.ndarray)
            assert iG.dtype == int
            assert iG.ndim is 2

        if ndim is not None and size is not None and iG is not None:
            assert iG.shape == (size, ndim)

        # distances between grid points are multiples of index spacing
        if index_spacing is not None and iG is not None:
            shifted_array = iG - iG[0]
            assert np.all(shifted_array % index_spacing == 0)

        if is_centered is not None and iG is not None:
            if is_centered:
                # Todo find something better like point symmetric...
                assert np.array_equal(iG[0], -iG[-1])
            else:
                assert np.all(iG[0] == 0)

        # Todo test pG

        if boundaries is not None:
            assert isinstance(boundaries, np.ndarray)
            assert boundaries.dtype == float
            assert boundaries.ndim == 2
            assert boundaries.shape[0] == 2

        return

    def __str__(self, write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance."""
        description = ''
        description += "Dimension = {}\n".format(self.ndim)
        description += "Shape = {}\n".format(self.shape)
        description += "Geometric Form = {}\n".format(self.form)
        description += "Total Size = {}\n".format(self.size)
        description += "Physical_Spacing = {}\n".format(self.physical_spacing)
        description += "Index_Spacing = {}\n".format(self.index_spacing)
        description += "Internal Step Size = {}\n".format(self.delta)
        description += 'Is_Centered = {}\n'.format(self.is_centered)
        description += "Boundaries:\n"
        description += '\t' + self.boundaries.__str__().replace('\n', '\n\t')
        if write_physical_grids:
            description += '\n'
            description += 'Physical Grid:\n\t'
            description += self.pG.__str__().replace('\n', '\n\t')
        return description
