
import numpy as np
import h5py
from math import isclose

import boltzpy as bp
import boltzpy.constants as bp_c


# noinspection PyPep8Naming
class Grid(bp.BaseClass):
    r"""Basic class for all Grids.

    .. todo::
        - in case of homogeneous simulation (grid shape = [1])
          force unnecessary parameters (spacings, delta) to be None.
          this forces possibles errors to occur in development.
        - add spacing documentation
        - Add unit tests
        - Add rotation of grid (useful for velocities)
        - Enable non-uniform/adaptive Grids
          (see :class:`~boltzpy.computation.Calculation`)

        Note
    ----
    The parameter :attr:`iMG` describes the
    position/physical values of all  Grid points.
    All entries must be viewed as multiples of :attr:`delta:

        :math:`pG = iG \cdot d`.

    Parameters
    ----------
    ndim : :obj:`int`
        The number of :obj:`Grid` dimensions.
        Must be in :const:`~boltzpy.constants.SUPP_GRID_DIMENSIONS`.
    shape : :obj:`tuple` [:obj:`int`]
        Number of :obj:`Grid` points for each dimension.
        Tuple of length :attr:`ndim`.
    physical_spacing : :obj:`float`
        Step size for the physical grid points.
    spacing : :obj:`int`, optional
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
    shape : :obj:`tuple` [:obj:`int`]
        Number of :obj:`Grid` points for each dimension.
    delta : :obj:`float`
        Smallest possible step size of the :obj:`Grid`.
    spacing : :obj:`int`, optional
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
                 physical_spacing=None,
                 spacing=2,
                 is_centered=False):
        self.check_parameters(ndim=ndim,
                              shape=shape,
                              physical_spacing=physical_spacing,
                              spacing=spacing,
                              is_centered=is_centered)
        self.ndim = ndim
        self.shape = shape
        if physical_spacing is not None:
            self.delta = physical_spacing / spacing
        else:
            self.delta = None
        self.spacing = spacing
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
        if self.delta is None or self.spacing is None:
            return None
        else:
            return self.delta * self.spacing

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
    def is_configured(self):
        """:obj:`bool` :
        True, if all necessary attributes of the instance are set.
        False Otherwise.
        """
        necessary_params = [self.ndim,
                            self.shape,
                            self.delta,
                            self.spacing,
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
        return self.is_configured and self.iG is not None

    #####################################
    #        Sorting and Ordering       #
    #####################################
    def key_distance(self, velocities):
        # NOTE: this acts as if the grid was infinite. This is desired for the partitioning
        assert isinstance(velocities, np.ndarray)
        # grids of even shape don't contain 0
        # thus need to be shifted into 0 for modulo operations
        assert len(set(self.shape)) == 1, "only works for square/cubic grids"
        if self.shape[0] % 2 == 0:
            velocities = velocities - self.spacing // 2
        distance = np.mod(velocities, self.spacing)
        distance = np.where(distance > self.spacing // 2,
                            distance - self.spacing,
                            distance)
        return distance

    @staticmethod
    def key_norm(velocities):
        norm = (velocities**2).sum(axis=-1)
        return norm

    def group(self, velocities):
        grouped_velocities = dict()
        keys = self.key_distance(velocities)
        for (i, v) in enumerate(velocities):
            key = tuple(keys[i])
            if key in grouped_velocities.keys():
                grouped_velocities[key].append(v)
            else:
                grouped_velocities[key] = [v]
        # Each Group is sorted by norm
        for (key, item) in grouped_velocities.items():
            item = sorted(item, key=self.key_norm)
            grouped_velocities[key] = np.array(item)
        return grouped_velocities

    def get_idx(self, values):
        """Find index of given values in :attr:`iG`
        Returns -1, if the value is not in this Grid.

        Parameters
        ----------
        values : :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        index : :obj:`~numpy.array` [:obj:`int']
        """
        assert isinstance(values, np.ndarray), (
            "values must be an np.array, not {}".format(type(values)))
        assert values.dtype == int, (
            "values must be an integer array, not {}".format(values.dtype))
        assert len(set(self.shape)) == 1, "only works for square/cubic grids"
        assert values.shape[-1] == self.ndim, "Only tested for 2D Velocities"
        assert values.shape[-1] == 2, "Only tested for 2D Velocities"
        BAD_VALUE = -2 * self.size
        # shift Grid to start (left bottom ) at 0
        values = values - self.iG[0]
        # divide by spacing to get the position on the (x,y,z) axis
        # sort out the values, that are not in the grid, by setting them to BAD_VALUE
        values = np.where(values % self.spacing == 0, values // self.spacing, BAD_VALUE)
        values = np.where(values >= 0, values, BAD_VALUE)
        values = np.where(values < self.shape[0], values, BAD_VALUE)
        # compute the (potential) index
        # Todo(LT) Reverse Ordering of Grids?
        factor = np.array([self.shape[0]**i for i in reversed(range(self.ndim))], dtype=int)
        idx = values.dot(factor)
        # remove Bad Values or points that are out of bounds
        idx = np.where(idx >= 0, idx, -1)
        return idx

    #####################################
    #              Utility              #
    #####################################
    def __contains__(self, item):
        assert isinstance(item, np.ndarray)
        return np.all(self.get_idx(item) != -1)

    def line(self, start, direction, steps):
        return (start + step * direction for step in steps
                if start + step * direction in self)

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
                          points_on_axis * self.spacing,
                          self.spacing)
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

        # center the Grid if necessary
        if self.is_centered:
            self.centralize()
        self.check_integrity()
        return

    def centralize(self):
        """Shift the integer Grid (:attr:`iG`) to be centered around zero.
        """
        assert isinstance(self.iG, np.ndarray)
        assert self.spacing is not None
        double_shift = np.max(self.iG, axis=0) + np.min(self.iG, axis=0)
        if np.all(double_shift % 2 == 0):
            shift = double_shift // 2
            self.iG -= shift
        else:
            msg = "Even Grids can only be centralized, " \
                  "if the spacing is even. " \
                  "spacing = {}".format(self.spacing)
            raise AttributeError(msg)
        return

    #####################################
    #           Visualization           #
    #####################################
    def plot(self, plot_object=None, **plot_style):
        """Plot the Grid using matplotlib.

        Parameters
        ----------
        plot_object : TODO Figure? matplotlib.pyplot?
        """
        show_plot_directly = plot_object is None
        pG = self.iG * self.delta
        if plot_object is None:
            # Choose standard pyplot
            import matplotlib.pyplot as plt
            plot_object = plt
        # Plot Grid as scatter plot
        if self.ndim == 2:
            plot_object.scatter(pG[:, 0], pG[:, 1],
                                **plot_style)
        else:
            raise NotImplementedError
        if show_plot_directly:
            plot_object.show()
        return plot_object

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
            # cast into tuple of ints
            shape = hdf5_group["Shape"][()]
            params["shape"] = tuple(int(width) for width in shape)
        # Todo remove physical spacing as attribute(parameter
        if "Physical_Spacing" in hdf5_group.keys():
            params["physical_spacing"] = float(hdf5_group["Physical_Spacing"][()])
        if "Spacing" in hdf5_group.keys():
            params["spacing"] = int(hdf5_group["Spacing"][()])
        else:
            params["spacing"] = None
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
        if self.spacing is not None:
            hdf5_group["Spacing"] = self.spacing
        if self.is_centered is not None:
            hdf5_group["Is_Centered"] = self.is_centered

        # check that the class can be reconstructed from the save
        other = Grid.load(hdf5_group)
        assert self == other
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self,
                        complete_check=True,
                        context=None):
        """Sanity Check.

        Assert all conditions in :meth:`check_parameters`.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all attributes must be assigned (not None).
            If False, then unassigned attributes are ignored.
        context : :class:`Simulation`, optional
            The Simulation, which this instance belongs to.
            This allows additional checks.
        """
        self.check_parameters(ndim=self.ndim,
                              shape=self.shape,
                              physical_spacing=self.physical_spacing,
                              spacing=self.spacing,
                              is_centered=self.is_centered,
                              delta=self.delta,
                              size=self.size,
                              iG=self.iG,
                              complete_check=complete_check,
                              context=context)
        return

    @staticmethod
    def check_parameters(ndim=None,
                         shape=None,
                         physical_spacing=None,
                         spacing=None,
                         is_centered=None,
                         delta=None,
                         size=None,
                         iG=None,
                         complete_check=False,
                         context=None):
        """Sanity Check.

        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        ndim : :obj:`int`, optional
        shape : :obj:`tuple` [:obj:`int`], optional
        physical_spacing : :obj:`float`, optional
        spacing : :obj:`int`, optional
        is_centered : :obj:`bool`, optional
        delta : :obj:`float`, optional
        size : :obj:`int`, optional
        iG : :obj:`~numpy.array` [:obj:`int`], optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        context : :class:`Simulation`, optional
            The Simulation, which this instance belongs to.
            This allows additional checks.
        """
        assert isinstance(complete_check, bool)
        # For complete check, assert that all parameters are assigned
        if complete_check is True:
            assert all(param_val is not None
                       for (param_key, param_val) in locals().items()
                       if param_key != "context")
        if context is not None:
            assert isinstance(context, bp.Simulation)

        # check all parameters, if set
        if ndim is not None:
            assert isinstance(ndim, int)
            assert ndim in bp_c.SUPP_GRID_DIMENSIONS

        if shape is not None:
            assert isinstance(shape, tuple)
            assert all(isinstance(width, int) for width in shape)
            assert (all(width >= 2 for width in shape)
                    or all(width == 1 for width in shape))

        if physical_spacing is not None:
            assert isinstance(physical_spacing, float)
            assert physical_spacing > 0

        if spacing is not None:
            assert isinstance(spacing, int)
            assert spacing > 0

        if is_centered is not None:
            assert isinstance(is_centered, bool)

        if delta is not None:
            assert isinstance(delta, float)
            assert delta > 0

        if size is not None:
            assert isinstance(size, int)
            assert size >= 1
            if shape is not None:
                assert np.prod(shape) == size

        if iG is not None:
            assert isinstance(iG, np.ndarray)
            assert iG.dtype == int
            assert iG.ndim == 2

        # check correct attribute relations
        if ndim is not None and shape is not None:
            assert len(shape) == ndim

        if all(attr is not None
               for attr in [physical_spacing, spacing, delta]):
            assert isclose(physical_spacing,
                           spacing * delta)

        if all(attr is not None for attr in [ndim, size, iG]):
            assert iG.shape == (size, ndim)

        # distances between grid points are multiples of index spacing
        if all(attr is not None for attr in [spacing, iG]):
            shifted_array = iG - iG[0]
            assert np.all(shifted_array % spacing == 0)

        if is_centered is not None and iG is not None:
            if is_centered:
                # Todo find something better like point symmetric...
                assert np.array_equal(iG[0], -iG[-1])
            else:
                assert np.all(iG[0] == 0)

        return

    def __str__(self, write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance."""
        description = ''
        description += "Dimension = {}\n".format(self.ndim)
        description += "Shape = {}\n".format(self.shape)
        description += "Total Size = {}\n".format(self.size)
        description += "Physical_Spacing = {}\n".format(self.physical_spacing)
        description += "Spacing = {}\n".format(self.spacing)
        description += "Internal Step Size = {}\n".format(self.delta)
        description += 'Is_Centered = {}\n'.format(self.is_centered)
        if write_physical_grids:
            description += '\n'
            description += 'Physical Grid:\n\t'
            description += self.pG.__str__().replace('\n', '\n\t')
        return description
