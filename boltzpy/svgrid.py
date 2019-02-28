import numpy as np
import h5py

import boltzpy as bp
import boltzpy.constants as bp_c


class SVGrid:
    r"""Manages the Velocity Grids of all
    :class:`~boltzpy.Species`.

    .. todo::
        - add fast method to get physical velocities from index (cython!)
          Apply this in pG attribute.
        - Todo Add unit tests
        - Todo add pMG back? If no -> remove passage from docstring

    Note
    ----
    Just as in the :class:`Grid` class,
    the parameter :attr:`iMG` describes the
    position/physical values of all  Grid points.
    All entries must be viewed as multiples of :attr:`delta:

        :math:`pMG = iMG \cdot d`.

    Note that velocity grid points may occur in multiple
    :class:`Velocity Grids <boltzpy.Grid>`.
    Array of shape (:attr:`size`, :attr:`ndim`)

    Parameters
    ----------
    ndim : :obj:`int`
        The number of :obj:`Grid` dimensions.
        Must be in :const:`~boltzpy.constants.SUPP_GRID_DIMENSIONS`.
    maximum_velocity : :obj:`float`
        Maximum physical velocity for every sub grid.
    shapes : :obj:`list` [:obj:`tuple` [:obj:`int`]]
        Contains the shape of each sub grid.
    spacings : :obj:`list` [:obj:`int`]
        Contains the index spacing of each sub grid.
    forms : :obj:`list` [:obj:`str`]
        Contains the geometric form of each sub grid
        Every element must be in
        :const:`~boltzpy.constants.SUPP_GRID_FORMS`.

    Attributes
    ----------
    ndim : :obj:`int`
        Dimensionality of all Velocity :class:`Grids <boltzpy.Grid>`.
        Must be in :const:`~boltzpy.constants.SUPP_GRID_DIMENSIONS`.
    maximum_velocity : :obj:`float`
        Maximum physical velocity for every sub grid.
    forms : :obj:`list` [:obj:`str`]
        Contains the :attr:`Grid.form` of each sub grid.
    shapes : :obj:`list` [:obj:`tuple` [:obj:`int`]]
        Contains the :attr:`Grid.shape` of each sub grid.
    spacings : :obj:`list` [:obj:`int`]
        Contains the :attr:`Grid.spacing` of each sub grid.
    delta : :obj:`float`
        Smallest possible step size of the :obj:`Grid`.
    index_range : :obj:`~numpy.array` [:obj:`int`]
        Denotes the beginning and end of a specimens velocity (sub) grid
        in the multi grid :attr:`iMG`.
        Array of shape (:attr:`size`, 2)
    vGrids : :obj:`~numpy.array` [:class:`~boltzpy.Grid`]
        Array of all Velocity :class:`Grids <boltzpy.Grid>`.
        Each Velocity Grids attribute
        :attr:`Grid.iG <boltzpy.Grid.iG>`
        links to its respective slice of :attr:`iMG`.
    iMG : :obj:`~numpy.array` [:obj:`int`]
        The *integer Multi-Grid*.
        It is a concatenation of all
        Velocity integer Grids
        (:attr:`Grid.iG <boltzpy.Grid>`);
        One for each :class:`Specimen`.
    """

    def __init__(self,
                 ndim=None,
                 maximum_velocity=None,
                 shapes=None,
                 spacings=None,
                 forms=None):
        self.check_parameters(ndim=ndim,
                              shapes=shapes,
                              spacings=None,
                              maximum_velocity=maximum_velocity,
                              forms=forms)
        self.ndim = ndim
        self.maximum_velocity = maximum_velocity
        self.shapes = shapes
        self.spacings = spacings
        self.forms = forms
        # the following attributes are set in setup()
        self.delta = None
        self.index_range = None
        self.vGrids = None
        self.iMG = None

        if self.is_configured:
            self.setup()
        return

    #####################################
    #           Properties              #
    #####################################
    @property
    def size(self):
        """:obj:`int` :
        The total number of velocity grid points
        over all sub grids.
        """
        if self.iMG is not None:
            return self.iMG.shape[0]
        else:
            return None

    @property
    def number_of_grids(self):
        """:obj:`int` :
        The number of different
        :class:`Velocity Grids <boltzpy.Grid>`.
        """
        if self.vGrids is not None:
            return self.vGrids.size
        else:
            return None

    @property
    def boundaries(self):
        """:obj:`list` [:obj:`~numpy.array` [:obj:`float`]] :
        :attr:`Grid.boundaries` of each sub grid.
        """
        if self.vGrids is not None:
            return [grid.boundaries for grid in self.vGrids]
        else:
            return None

    @property
    def is_configured(self):
        """:obj:`bool` :
        True, if all attributes necessary to run :meth:`setup` are set.
        False Otherwise.
        """
        necessary_params = [self.ndim,
                            self.shapes,
                            self.spacings,
                            self.forms]
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
        is_set_up = (self.index_range is not None
                     and self.iMG is not None
                     and self.vGrids is not None)
        self.check_integrity()
        return is_set_up

    #####################################
    #           Configuration           #
    #####################################
    def setup(self):
        """Construct the attributes
        :attr:`Grid.iMG`,
        :attr:`Grid.index_range`,
        :attr:`Grid.vGrids`."""
        # Basic asserts : is everything configured and correct?
        self.check_integrity(False)
        assert self.is_configured

        number_of_grids = len(self.shapes)
        self.index_range = np.zeros((number_of_grids, 2),
                                    dtype=int)
        self.vGrids = np.empty(number_of_grids, dtype='object')
        # set up sub grids, one by one
        for i in range(number_of_grids):
            # Todo the physical spacing is only a dummy so far
            new_grid = bp.Grid(ndim=self.ndim,
                               shape=self.shapes[i],
                               form=self.forms[i],
                               physical_spacing=1.0,
                               spacing=self.spacings[i],
                               is_centered=True)
            self.vGrids[i] = new_grid
            self.index_range[i, 1] = self.index_range[i, 0] + new_grid.size
            if i + 1 < number_of_grids:
                self.index_range[i + 1, 0] = self.index_range[i, 1]

        # Sub grids only have a view on the data
        # The actual data are stored in the multi grid
        self.iMG = np.zeros((self.index_range[-1, 1], self.ndim),
                            dtype=int)
        for (idx_G, G) in enumerate(self.vGrids):
            [beg, end] = self.index_range[idx_G]
            self.iMG[beg:end, :] = G.iG[...]
            G.iG = self.iMG[beg:end]

        # Todo find more elegant way for this
        self.delta = self.maximum_velocity / np.max(self.iMG)
        for G in self.vGrids:
            G.delta = self.delta
        return

    #####################################
    #               Indexing            #
    #####################################
    # Todo replace by get_index in respective vGrid?
    # Todo write Gird.binary_search (iterate over dimensions
    def find_index(self,
                   index_of_specimen,
                   desired_value):
        """Find index of given grid_entry in :attr:`iMG`
        Returns None, if the value is not in the specified Grid.

        Parameters
        ----------
        index_of_specimen : :obj:`int`
        desired_value : :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        global_index : :obj:`int` of :obj:`None`

        """
        index_grid = self.vGrids[index_of_specimen].iG
        grid_iterator = (idx for (idx, value) in enumerate(index_grid)
                         if np.all(value == desired_value))
        local_index = next(grid_iterator, None)
        if local_index is None:
            return None
        else:
            start_of_grid = self.index_range[index_of_specimen, 0]
            global_index = start_of_grid + local_index
            assert np.all(self.iMG[global_index] == desired_value)
            return global_index

    # Todo should be faster with next()
    # Todo change name
    # Todo delete - is it used anywhere?
    def get_specimen(self, velocity_idx):
        """Get :class:`boltzpy.Specimen` index
        of given velocity in :attr:`iMG`.

        Parameters
        ----------
        velocity_idx : :obj:`int`

        Returns
        -------
        index : :obj:`int`

        Raises
        ------
        err_idx : :obj:`IndexError`
            If *velocity_idx* is out of the range of
            :attr:`SVGrid.iMG`.
        """
        for (i, [beg, end]) in enumerate(self.index_range):
            if beg <= velocity_idx < end:
                return i
        msg = 'The given index ({}) points out of the boundaries of ' \
              'iMG, the concatenated Velocity Grid.'.format(velocity_idx)
        raise IndexError(msg)

    #####################################
    #           Visualization           #
    #####################################
    #: :obj:`list` [:obj:`dict`]:
    #: Default plot_styles for :meth::`plot`
    plot_styles = [{"marker": 'o', "color": "r", "facecolors": 'none'},
                   {"marker": 's', "color": "b", "facecolors": 'none'},
                   {"marker": 'x', "color": "black"},
                   {"marker": 'D', "color": "green", "facecolors": "none"}]

    def plot(self, plot_object=None):
        """Plot the Grid using matplotlib.

        Parameters
        ----------
        plot_object : TODO Figure? matplotlib.pyplot?
        """
        show_plot_directly = plot_object is None
        if plot_object is None:
            # Choose standard pyplot
            import matplotlib.pyplot as plt
            plot_object = plt
        # Plot Grids as scatter plot
        for (idx_G, G) in enumerate(self.vGrids):
            G.plot(plot_object, **(self.plot_styles[idx_G]))
        if show_plot_directly:
            plot_object.show()
        return plot_object

    #####################################
    #           Serialization           #
    #####################################
    # Todo read and write all Velocity Grids instead?
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`SVGrid` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`SVGrid`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "SVGrid"

        # read attributes from file
        params = dict()
        if "Dimensions" in hdf5_group.keys():
            params["ndim"] = int(hdf5_group["Dimensions"][()])
        if "Maximum_Velocity" in hdf5_group.keys():
            params["maximum_velocity"] = float(hdf5_group["Maximum_Velocity"][()])
        if "Shapes" in hdf5_group.keys():
            # cast into tuple of ints
            shapes = [tuple(int(width) for width in shape)
                      for shape in hdf5_group["Shapes"]]
            params["shapes"] = shapes
        if "Spacings" in hdf5_group.keys():
            spacings = [int(spacing)
                        for spacing in hdf5_group["Spacings"]]
            params["spacings"] = spacings
        if "Forms" in hdf5_group.keys():
            params["forms"] = list(hdf5_group["Forms"][()])

        self = SVGrid(**params)
        return self

    def save(self, hdf5_group):
        """Write the main parameters of the :class:`SVGrid` instance
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
        hdf5_group.attrs["class"] = "SVGrid"

        # write all set attributes to file
        if self.ndim is not None:
            hdf5_group["Dimensions"] = self.ndim
        if self.maximum_velocity is not None:
            hdf5_group["Maximum_Velocity"] = self.maximum_velocity
        if self.shapes is not None:
            hdf5_group["Shapes"] = self.shapes
        if self.spacings is not None:
            hdf5_group["Spacings"] = self.spacings
        if self.forms is not None:
            # Todo This is a dirty hack
            hdf5_group["Forms"] = np.array(self.forms,
                                           dtype=h5py.special_dtype(vlen=str))

        # check that the class can be reconstructed from the save
        other = SVGrid.load(hdf5_group)
        assert self == other
        return

    #####################################
    #           Verification            #
    #####################################
    # Todo compare with sim.s.masses -> additional asserts
    def check_integrity(self,
                        complete_check=True,
                        context=None):
        """Sanity Check.

        Assert all conditions in :meth:`check_parameters`

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
                              maximum_velocity=self.maximum_velocity,
                              delta=self.delta,
                              shapes=self.shapes,
                              spacings=self.spacings,
                              forms=self.forms,
                              index_range=self.index_range,
                              vGrids=self.vGrids,
                              iMG=self.iMG,
                              number_of_grids=self.number_of_grids,
                              size=self.size,
                              boundaries=self.boundaries,
                              complete_check=complete_check,
                              context=context)
        return

    # Todo write checks on boundaries
    @staticmethod
    def check_parameters(ndim=None,
                         maximum_velocity=None,
                         delta=None,
                         shapes=None,
                         spacings=None,
                         forms=None,
                         vGrids=None,
                         index_range=None,
                         iMG=None,
                         number_of_grids=None,
                         size=None,
                         boundaries=None,
                         complete_check=False,
                         context=None):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        ndim : :obj:`int`, optional
        maximum_velocity : :obj:`float`. optional
        delta : :obj:`float`, optional
        shapes : :obj:`list` [:obj:`tuple` [:obj:`int`]], optional
        spacings : :obj:`list` [:obj:`int`], optional
        forms : :obj:`list` [:obj:`str`], optional
        vGrids : :obj:`~numpy.array` [:class:`Grid`], optional
        index_range : :obj:`~numpy.array` [:obj:`int`], optional
        iMG : :obj:`~numpy.array` [:obj:`int`], optional
        number_of_grids : :obj:`int`, optional
        size : :obj:`int`, optional
        boundaries : :obj:`list` [:obj:`~numpy.array` [:obj:`float`]], optional
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

        # Todo define number of grids, if possible

        if ndim is not None:
            assert isinstance(ndim, int)
            assert ndim in bp_c.SUPP_GRID_DIMENSIONS
            if context is not None and context.p.ndim is not None:
                assert ndim >= context.p.ndim

        if maximum_velocity is not None:
            assert isinstance(maximum_velocity, float)
            assert maximum_velocity > 0

        if delta is not None:
            assert isinstance(delta, float)
            assert delta > 0

        if shapes is not None:
            assert isinstance(shapes, list)
            if number_of_grids is not None:
                assert number_of_grids == len(shapes)
            else:
                number_of_grids = len(shapes)
            for shape in shapes:
                bp.Grid.check_parameters(ndim=ndim,
                                         shape=shape,
                                         context=context)
                if any(entry != shape[0] for entry in shape):
                    raise NotImplementedError

        if spacings is not None:
            assert isinstance(spacings, list)
            if number_of_grids is not None:
                assert number_of_grids == len(spacings)
            else:
                number_of_grids = len(spacings)
            for spacing in spacings:
                bp.Grid.check_parameters(ndim=ndim,
                                         spacing=spacing,
                                         context=context)

        if forms is not None:
            assert isinstance(forms, list)
            if number_of_grids is not None:
                assert number_of_grids == len(forms)
            else:
                number_of_grids = len(forms)
            for form in forms:
                bp.Grid.check_parameters(ndim=ndim,
                                         form=form,
                                         context=context)

        if vGrids is not None:
            assert isinstance(vGrids, np.ndarray)
            if number_of_grids is not None:
                assert number_of_grids == vGrids.size
            else:
                number_of_grids = vGrids.size
            assert vGrids.ndim == 1
            assert vGrids.dtype == 'object'
            for (idx_G, G) in enumerate(vGrids):
                isinstance(G, bp.Grid)
                G.check_integrity()
                if ndim is None:
                    ndim = G.ndim
                else:
                    assert ndim == G.ndim
                if delta is None:
                    delta = G.delta
                else:
                    assert delta == G.delta
                if shapes is not None:
                    assert np.all(shapes[idx_G] == G.shape)
                if spacings is not None:
                    assert spacings[idx_G] == G.spacing
                if forms is not None:
                    assert forms[idx_G] == G.form
            if context is not None:
                assert vGrids.size == context.s.size

        if index_range is not None:
            assert isinstance(index_range, np.ndarray)
            assert index_range.dtype == int
            assert index_range.ndim == 2
            if number_of_grids is not None:
                assert number_of_grids == index_range.shape[0]
            else:
                number_of_grids = index_range.shape[0]
            assert index_range.shape[1] == 2
            assert all(0 <= idx for idx in index_range.flatten())
            assert all(index_range[i, 1] == index_range[i + 1, 0]
                       for i in range(number_of_grids - 1))
            assert all(beg < end for [beg, end] in index_range)
            if vGrids is not None:
                for (idx_G, G) in enumerate(vGrids):
                    [beg, end] = index_range[idx_G]
                    assert G.size == end - beg

        if iMG is not None:
            assert isinstance(iMG, np.ndarray)
            assert iMG.dtype == int
            assert iMG.ndim == 2
            if index_range is not None:
                assert iMG.shape[0] == index_range[-1, 1]
            if vGrids is not None:
                for G in vGrids:
                    assert np.shares_memory(iMG, G.iG)
        return

    def __eq__(self, other):
        if not isinstance(other, SVGrid):
            return False
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            return False
        for (key, value) in self.__dict__.items():
            other_value = other.__dict__[key]
            if type(value) != type(other_value):
                return False
            if isinstance(value, np.ndarray):
                if np.all(value != other_value):
                    return False
            else:
                if value != other_value:
                    return False
        return True

    def __lt__(self, other):
        if self.size <= other.size:
            return True
        else:
            return False

    def __str__(self,
                write_physical_grid=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance.
        """
        description = ''
        description += "Dimension = {}\n".format(self.ndim)
        description += "Total Size = {}\n".format(self.size)

        if self.vGrids is not None:
            for (idx_G, vGrid) in enumerate(self.vGrids):
                description += 'Specimen_{idx}:\n\t'.format(idx=idx_G)
                grid_str = vGrid.__str__(write_physical_grid)
                description += grid_str.replace('\n', '\n\t')
                description += '\n'
        return description[:-1]
