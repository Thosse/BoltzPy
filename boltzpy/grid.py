
import numpy as np

import boltzpy as bp


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

    Parameters
    ----------
    shape : :obj:`~numpy.array` [:obj:`int`]
        Number of :obj:`Grid` points for each dimension.
    delta : :obj:`float`
        Internal step size.
        This is NOT the physical distance between grid points.
    spacing : :obj:`int`
        An integer stretching factor.
        Using even spacings allows centered grids of even shape,
        It is used for velocity grids of multiple species with differing mass.
        It determines write-intervalls for time grids.
    is_centered : :obj:`bool`, optional
        True if the Grid should be centered around zero.


    Attributes
    ----------
    shape : :obj:`~numpy.array` [:obj:`int`]
        Number of :obj:`Grid` points for each dimension.
    delta : :obj:`float`
        Smallest possible step size of the :obj:`Grid`.
    spacing : :obj:`int`
        This allows
        centered velocity grids (without the zero),
        write-intervalls for time grids
        and possibly adaptive positional grids.
    is_centered : :obj:`float`
        If True, then the Grid will be centered around zero.
    ndim : :obj:`int`
        The number of :obj:`Grid` dimensions
    size : :obj:`int` :
        The total number of grid points.
    iG : :obj:`~numpy.array` [:obj:`int`] :
        The *integer Grid* describes the position of all :class:`Grid` points
        in multiples of :attr:`delta`, e.g. :math:`pG := iG \cdot delta`.

        This allows precise computations, without rounding errors.

        Array of shape (:attr:`size`, :attr:`ndim`).
    """
    def __init__(self,
                 shape,
                 delta,
                 spacing,
                 is_centered=False):
        self.shape = np.array(shape, dtype=int)
        self.delta = float(delta)
        self.spacing = int(spacing)
        self.is_centered = bool(is_centered)
        self.ndim = self.shape.size
        self.size = int(np.prod(self.shape))
        # if the grid is centered, all values are shifted by +offset
        # Note that True/False == 1/0
        # todo rename into shift
        self.offset = -(self.spacing
                        * (np.array(self.shape, dtype=int) - 1)
                        // 2) * self.is_centered
        # todo rename into i_vals
        self.iG = self.iv(np.arange(self.size))

        self.check_integrity()
        return

    #####################################
    #           Properties              #
    #####################################
    @property
    def physical_spacing(self):
        r""":obj:`float` :
        The physical distance between two grid points.

        It holds :math:`physical \_ spacing = delta \cdot index \_ spacing`.
        """
        return self.delta * self.spacing

    # Todo rename into vals
    @property
    def pG(self):
        r""":obj:`~numpy.array` [:obj:`float`] :
        Construct the *physical Grid* (**computationally heavy!**).

        The physical Grid pG denotes the physical values of
        all :class:`Grid` points.

        :math:`pG := iG \cdot delta`

        Array of shape (:attr:`size`, :attr:`ndim`).
         """
        return self.pv(np.arange(self.size))

    @staticmethod
    def parameters():
        return {"shape",
                "delta",
                "spacing",
                "is_centered"}

    @staticmethod
    def attributes():
        attrs = Grid.parameters()
        attrs.update({"ndim",
                      "size",
                      "iG",
                      "physical_spacing",
                      "pG"})
        return attrs

    #####################################
    #         Indexes and Values        #
    #####################################
    # todo rename get_i_vals
    def iv(self, idx):
        """Return the integer values of the indexed grid points`
        Parameters
        ----------
        idx : :obj:`ìnt` or :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        values : :obj:`~numpy.array` [:obj:`int`]
        """
        assert np.all(idx >= 0)
        assert np.all(idx < self.size)
        idx = np.copy(idx)
        values = np.empty(np.shape(idx) + (self.ndim,), dtype=int)
        # calculate the values, by iterating over the dimension
        for i in range(self.ndim):
            multi = np.prod(self.shape[i + 1:self.ndim], dtype=int)
            values[..., i] = idx // multi
            idx -= multi * values[..., i]
            values[..., i] = self.spacing * values[..., i]
        # centralize Grid around zero, by adding the offset
        values += self.offset
        return values

    # todo rename get_vals
    def pv(self, idx):
        """Return the physical values of the indexed grid points`
        Parameters
        ----------
        idx : :obj:`ìnt` or :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        values : :obj:`~numpy.array` [:obj:`float`]
        """
        return self.iv(idx) * self.delta

    def get_idx(self, values):
        """Find index of given values in :attr:`iG`
        Returns -1, if the value is not in this Grid.

        Parameters
        ----------
        values : :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        index : :obj:`~numpy.array` [:obj:`int`]
        """
        assert isinstance(values, np.ndarray), (
            "values must be an np.array, not {}".format(type(values)))
        assert values.dtype == int, (
            "values must be an integer array, not {}".format(values.dtype))
        BAD_VALUE = -2 * self.size ** self.ndim
        # shift Grid to start (left bottom ) at 0
        values = values - self.iG[0]
        # divide by spacing to get the position on the (x,y,z) axis
        values = np.where(values % self.spacing == 0,
                          values // self.spacing,
                          BAD_VALUE)
        # sort out the values, that are not in the grid
        values = np.where(values >= 0, values, BAD_VALUE)
        values = np.where(values < self.shape, values, BAD_VALUE)
        # compute the (potential) index
        factor = np.array([np.prod(self.shape[i+1:]) for i in range(self.ndim)],
                          dtype=int)
        idx = values.dot(factor)
        # remove Bad Values or points that are out of bounds
        idx = np.where(idx >= 0, idx, -1)

        # Todo assert that result is correct
        # noinspection PyUnreachableCode
        # if __debug__:
        #     pass
        return idx

    #####################################
    #        Sorting and Ordering       #
    #####################################
    def key_distance(self, values):
        # NOTE: this acts as if the grid was infinite.
        # This is desired for the partitioning
        assert isinstance(values, np.ndarray)
        # Even or centered grids  mav have an offset % spacing != 0
        # thus the grid points are not multiples of the spacing
        # values must be shifted to reverse that offset
        values = values - self.offset
        distance = np.mod(values, self.spacing)
        distance = np.where(distance > self.spacing // 2,
                            distance - self.spacing,
                            distance)
        # Now values - distance is in the (infinite) Grid
        return distance

    @staticmethod
    def key_norm(values):
        norm = (values**2).sum(axis=-1)
        return norm

    @staticmethod
    def group(keys, values, as_array=True, sort_values=True):
        # Reuse implementation in CollisionModel
        grp = bp.CollisionModel.group(keys, values, as_array)
        if sort_values:
            # unpack dict values, for consistent access
            arrays = grp if as_array else grp.values()
            # sort arrays by norm
            for arr in arrays:
                order = np.argsort(Grid.key_norm(arr), kind="stable")
                arr[...] = arr[order]
        return grp

    #####################################
    #              Utility              #
    #####################################
    def __contains__(self, item):
        assert isinstance(item, np.ndarray)
        assert item.shape[-1] == self.ndim
        return np.all(self.get_idx(item) != -1)

    def hyperplane(self, start, normal):
        shifted_vels = self.iG - start
        pos = np.where(shifted_vels @ normal == 0)
        return self.iG[pos]

    #####################################
    #           Visualization           #
    #####################################
    def plot(self, plot_object=None, **plot_style):
        """Plot the Grid using matplotlib.

        Parameters
        ----------
        plot_object : TODO Figure? matplotlib.pyplot?
        """
        if plot_object is None:
            import matplotlib.pyplot as plt
            projection = "3d" if self.ndim == 3 else None
            ax = plt.figure().add_subplot(projection=projection)
        else:
            ax = plot_object
        # transpose grid points to unpack each coordinate separately
        assert self.pG.ndim == 2
        grid_points = self.pG.transpose()
        # Plot Grid as scatter plot
        ax.scatter(*grid_points,  **plot_style)
        if plot_object is None:
            # noinspection PyUnboundLocalVariable
            plt.show()
        return plot_object

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check"""
        bp.BaseClass.check_integrity(self)
        assert isinstance(self.ndim, int)
        assert self.ndim >= 0

        assert isinstance(self.shape, np.ndarray)
        assert self.shape.size == self.ndim
        assert self.shape.dtype == int
        assert np.all(self.shape >= 1)

        assert isinstance(self.size, int)
        assert self.size >= 0
        assert np.prod(self.shape) == self.size

        assert isinstance(self.delta, float)
        assert self.delta > 0

        assert isinstance(self.spacing, int)
        assert self.spacing > 0

        assert isinstance(self.physical_spacing, float)
        assert self.physical_spacing > 0

        assert isinstance(self.is_centered, bool)
        if self.is_centered:
            assert (np.all(self.spacing % 2 == 0 or (np.array(self.shape) - 1) % 2 == 0))

        assert isinstance(self.iG, np.ndarray)
        assert self.iG.dtype == int
        # iG is a flattened array of ndim-vectors
        assert self.iG.ndim == 2
        # Grids must have the correct shape
        assert self.iG.shape == (self.size, self.ndim)
        # distances between grid points are multiples of index spacing
        assert np.all((self.iG - self.iv(0)) % self.spacing == 0)
        if self.is_centered:
            assert np.array_equal(self.iG, -self.iG[::-1])
        else:
            assert np.all(self.iv(0) == 0)
        return

    def __str__(self, write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance."""
        description = ''
        description += "ndim = {}\n".format(self.ndim)
        description += "shape = {}\n".format(self.shape)
        description += "size = {}\n".format(self.size)
        description += "delta = {}\n".format(self.delta)
        description += "spacing = {}\n".format(self.spacing)
        description += "physical_spacing = {}\n".format(self.physical_spacing)
        description += 'is_centered = {}\n'.format(self.is_centered)
        if write_physical_grids:
            description += '\n'
            description += 'Physical Grid:\n\t'
            description += self.pG.__str__().replace('\n', '\n\t')
        return description
