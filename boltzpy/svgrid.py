import numpy as np
import h5py
import math

import boltzpy as bp
import boltzpy.constants as bp_c


# Todo move offset out of SVGrid? extra sim.attribute?
class SVGrid:
    """Manages the Velocity Grids of all
    :class:`~boltzpy.Species` /
    :class:`~boltzpy.Specimen`.

    .. todo::
        - add method for physical velocities (offset + iMG*d).
          Apply this in pG attribute.
        - Ask Hans, should the Grid always contain the center/zero?
        - Todo Add unit tests
        - is it useful to implement different construction schemes?
          (grid boundaries/step_size can be chosen to fit in grids,
          based on the least common multiple,...)

    Attributes
    ----------
    form : :obj:`str`
        Geometric form of all Velocity :class:`Grids <boltzpy.Grid>`.
        Must be an element of
        :const:`~boltzpy.constants.SUPP_GRID_FORMS`.
    dim : :obj:`int`
        Dimensionality of all Velocity :class:`Grids <boltzpy.Grid>`.
        Must be in :const:`~boltzpy.constants.SUPP_GRID_DIMENSIONS`.
    masses :obj:`~numpy.array` [:obj:`int`] :
        Array of masses of the simulated :class:`Specimen`.
    vGrids : :obj:`~numpy.array` [:class:`~boltzpy.Grid`]
        Array of all Velocity :class:`Grids <boltzpy.Grid>`.
        Each Velocity Grids attribute
        :attr:`Grid.iG <boltzpy.Grid.iG>`
        links to its respective slice of :attr:`iMG`.
    iMG : :obj:`~numpy.array` [:obj:`int`]
        The *integer Multi-Grid*. It is a concatenation of the
        Velocity integer Grids
        (:attr:`Grid.iG <boltzpy.Grid>`) of all
        :class:`Species`.
        It describes the
        position/physical values (:attr:`pG`)
        of all Velocity Grid points.
        All entries are factors, such that
        :math:`pG = offset + iG \cdot d`.
        Note that some V-Grid points occur in multiple
        :class:`Velocity Grids <boltzpy.Grid>`.
        Array of shape (:attr:`size`, :attr:`dim`).
    offset : :obj:`~numpy.array` [:obj:`float`]
        Shifts the physical velocities, to be centered around :attr:`offset`.
        The physical value of any Velocity-Grid point v_i of Specimen S
        is :math:`offset + d[S] \cdot SVG[i]`.
        Array of shape=(dim,).
    """

    def __init__(self,
                 grid_form=None,
                 grid_dimension=None,
                 min_points_per_axis=None,
                 max_velocity=None,
                 masses=None,
                 velocity_offset=None):
        self.check_parameters(form=grid_form,
                              dimension=grid_dimension,
                              max_velocity=max_velocity,
                              min_points_per_axis=min_points_per_axis,
                              masses=masses,
                              velocity_offset=velocity_offset)
        self.form = grid_form
        self.dim = grid_dimension
        self._MAX_V = max_velocity
        self._MIN_N = min_points_per_axis
        self.masses = masses
        if velocity_offset is not None:
            self.offset = np.array(velocity_offset)
        elif grid_dimension is not None:
            self.offset = np.zeros(grid_dimension, dtype=float)
        else:
            self.offset = None

        # The remaining attributes are calculated in setup() method
        # _index[i] denotes the beginning of the i-th velocity Grid in iMG
        self._index = None
        self.vGrids = None
        self.iMG = None
        self.setup()
        return

    # TODO figure out whats useful
    # def __getitem__(self, specimen_idx=None, velocity_idx=None):

    #####################################
    #           Properties              #
    #####################################
    @property
    def size(self):
        """:obj:`int` :
        The total number of Velocity-Grid points
        over all :class:`~boltzpy.Species`.
        """
        if self._index is None:
            return None
        else:
            return self._index[-1]

    @property
    def n_grids(self):
        """:obj:`int` :
        The number of different
        :class:`Velocity Grids <boltzpy.Grid>`.
        """
        try:
            return self.vGrids.size
        except AttributeError:
            return None

    # Todo can be replaced by pv method?
    @property
    def pMG(self):
        """:obj:`~numpy.array` [:obj:`float`] :
        Construct the *physical Multi-Grid* (**computationally heavy!**).

            The physical Multi-Grid pG denotes the physical values /
            position of all grid points.

                :math:`pG := offset + iG \cdot d`

            Array of shape (:attr:`size`, :attr:`dim`)
         """
        pMG = np.zeros(self.iMG.shape, dtype=float)
        for (i_G, G) in enumerate(self.vGrids):
            [beg, end] = self.idx_range(i_G)
            pMG[beg: end] = G.pG
        return pMG

    # Todo def pv(velocity_idx) -> implement in Grid as well

    @property
    def boundaries(self):
        """:obj:`~numpy.array` [:obj:`float`] :
        Minimum and maximum physical values
        in all :class:`Velocity Grids <boltzpy.Grid>`.

        All :class:`~boltzpy.Specimen`
        have equal boundaries. Thus it is calculated based on
        an arbitrary :class:`~boltzpy.Specimen`.
        """
        if self.vGrids is None:
            return None
        else:
            return self.offset + self.vGrids[0].boundaries

    @property
    def is_configured(self):
        """Check if all necessary attributes of the instance are set.

        Returns
        -------
        :obj:`bool`
        """
        necessary_params = [self.form,
                            self.dim,
                            self._MAX_V,
                            self._MIN_N,
                            self.masses]
        if any([val is None for val in necessary_params]):
            return False
        else:
            return True

    @property
    def is_set_up(self):
        """Check if the instance is completely set up.

        Returns
        -------
        :obj:`bool`
        """
        ret_val = (self.is_configured
                   and self._index is not None
                   and self.iMG is not None
                   and self.vGrids is not None)
        return ret_val

    #####################################
    #           Configuration           #
    #####################################
    def setup(self):
        r"""Construct the
        :class:`Velocity Grids <boltzpy.Grid>` :attr:`vGrids`
        for the given
        :class:`species_array <boltzpy.Species>`
        and store their index Grids in :attr:`iMG`.

        1. Calculate the minimum number of :ref:`segments <segment>` any
           :class:`Velocity Grids <boltzpy.Grid>`
           must have.
        2. Calculate the number of
           :ref:`complete segments <complete_segment>`, such that
           every :class:`Velocity Grids <boltzpy.Grid>`
           has at least the minimum number of :ref:`segments <segment>`.
        3. Calculate the shape and spacing of each
           :class:`Velocity Grids <boltzpy.Grid>`,
           based on its :class:`Specimens <boltzpy.Specimen>`
           :attr:`~boltzpy.Specimen.mass`.
        4. Initialize and
           :meth:`~boltzpy.Grid.centralize` the
           :class:`Velocity Grids <boltzpy.Grid>`
           and store them in :attr:`vGrids` and :attr:`iMG`

        Notes
        -----
          .. _segment:

          * A **segment** is the area between two neighbouring grid points
            on any grid axis. Along any axis holds
            :math:`n_{Segments} = n_{points} - 1.`

          .. _complete_segment:

          * A **complete segment** is the smallest union of connected
            :ref:`segments <segment>`,
            beginning and ending with a shared grid point for all
            :class:`Velocity Grids <boltzpy.Grid>`.
            In the :class:`Velocity Grid <boltzpy.Grid>`
            of any :class:`~boltzpy.Specimen`,
            a single complete segment consists of exactly
            :attr:`~boltzpy.Specimen.mass` segments.
            Thus for any :class:`~boltzpy.Specimen`:

              :math:`n_{Segments} = n_{complete \: Segments} \cdot mass`


          * Heavier :class:`~boltzpy.Specimen`
            are more inert and move less.
            Thus, for any
            :class:`~boltzpy.Specimen` the
            :attr:`~boltzpy.Grid.spacing`
            of its :class:`Velocity Grid <boltzpy.Grid>`
            is inversely proportional to its
            :attr:`~boltzpy.Specimen.mass` :

              :math:`spacing[i] \sim \frac{1}{mass[i]}.`

          * A smaller :attr:`~boltzpy.Grid.spacing`
            leads to larger grids.
            Thus for any
            :class:`~boltzpy.Specimen` the size of its
            :class:`Velocity Grid <boltzpy.Grid>`
            grows cubic in its :attr:`~boltzpy.Specimen.mass`.

              :math:`size \sim n^{dim} \sim mass^{dim}`.
        """
        # only run setup, if all necessary parameters are set
        if not self.is_configured:
            return
        else:
            # sanity check
            self.check_integrity(False)

        # Construct self.vGrids
        min_mass = np.amin(self.masses)
        # Minimum number of segments (see docstring)
        min_segments = self._MIN_N - 1
        # Number of complete segments (see docstring)
        n_complete_segments = math.ceil(min_segments / min_mass)
        number_of_grid_points = [(n_complete_segments * mass) + 1
                                 for mass in self.masses]
        grid_shapes = [[n] * self.dim for n in number_of_grid_points]
        # spacing of the velocity grid for each specimen
        # Todo replace, allow Grid.init with number of points (array)
        spacings = [2 * self._MAX_V / (n_i - 1)
                    for n_i in number_of_grid_points]
        # Contains the velocity Grid of each specimen
        vGrids = [bp.Grid(grid_form=self.form,
                          grid_dimension=self.dim,
                          grid_shape=np.array(grid_shapes[i]),
                          grid_spacing=spacings[i],
                          grid_is_centered=True)
                  for i in range(self.masses.size)]
        self.vGrids = np.array(vGrids)

        # construct self._index
        self._index = np.zeros(self.masses.size + 1,
                               dtype=int)
        for (i_G, vGrid) in enumerate(vGrids):
            self._index[i_G + 1] = self._index[i_G] + vGrid.size

        # construct self.iMG
        self.iMG = np.zeros((self.size, self.dim),
                            dtype=int)
        # store the integer Grids (vGrid.iG), by writing them
        # consecutively into the integer Multi-Grid (self.iMG)
        for (i_G, vGrid) in enumerate(self.vGrids):
            [beg, end] = self.idx_range(i_G)
            self.iMG[beg: end, :] = vGrid.iG[:, :]
            vGrid.iG = self.iMG[beg: end, :]

        self.check_integrity()
        return

    #####################################
    #               Indexing            #
    #####################################
    # TODO Add idx_range as attribute (arr (s.size, 2)
    def idx_range(self, specimen_index):
        """Get delimiters [begin, end]
        of the indexed :class:`Specimens <boltzpy.Specimen>`
        Velocity :class:`Grid <boltzpy.Grid>`
        in :class:`iMG <boltzpy.SVGrid>`.

        Parameters
        ----------
        specimen_index : :obj:`int`
            Index of the :class:`Specimen` in the
            :class:`Specimen Array <Species>`.

        Returns
        -------
        :obj:`~numpy.array` [:obj:`int`]
        """
        return self._index[specimen_index: specimen_index + 2]

    # Todo indexing needs to be checked/edited for new SVGrid class
    # Todo replace by index in respective vGrid?
    def get_index(self,
                  specimen_index,
                  grid_entry):
        """Get position of given grid_entry in :attr:`iMG`

            1. Generate *grid_entries* index in attr:`SVGrid.iMG`,
               assuming it is an element of the given Specimens Velocity-Grid.
            2. Counter-check if the indexed value matches the given value.
               If this is true, return the index, otherwise return None.

        Parameters
        ----------
        specimen_index : :obj:`int`
        grid_entry : :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        :obj:`int`
            Index of *grid_entry* in :attr:`~SVGrid.iMG`.

        Raises
        ------
        ValueError
            If *grid_entry* is not in the Velocity Grid.
        """
        # Todo Throw exception if not a grid entry (instead of None)
        i_flat = 0
        # get vector-index, by reversing Grid.centralize() - method
        i_vec = np.array((grid_entry + self.vGrids[specimen_index].n - 1) // 2,
                         dtype=int)
        for _dim in range(self.dim):
            n_loc = self.vGrids[specimen_index].n[_dim]
            i_flat *= n_loc
            i_flat += i_vec[_dim]
        i_flat += self._index[specimen_index]
        if all(np.array(self.iMG[i_flat] == grid_entry)):
            return i_flat
        else:
            raise ValueError

    # Todo should be faster with next()
    # Todo change name
    def get_specimen(self, velocity_idx):
        """Get :class:`boltzpy.Specimen` index
        of given velocity in :attr:`iMG`.

        Parameters
        ----------
        velocity_idx : :obj:`int`

        Returns
        -------
        :obj:`int`

        Raises
        ------
        IndexError
            If *velocity_idx* is out of the range of
            :attr:`SVGrid.iMG`.
        """
        for i in range(self._index.size):
            if self._index[i] <= velocity_idx < self._index[i + 1]:
                return i
        msg = 'The given index ({}) points out of the boundaries of ' \
              'iMG, the concatenated Velocity Grid.'.format(velocity_idx)
        raise IndexError(msg)

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
        :class:`SVGrid`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "SVGrid"
        self = SVGrid()

        # read attributes from file
        try:
            key = "Form"
            self.form = hdf5_group[key].value
        except KeyError:
            self.form = None
        try:
            key = "Dimension"
            self.dim = int(hdf5_group[key].value)
        except KeyError:
            self.dim = None
        try:
            key = "Maximum Velocity"
            self._MAX_V = hdf5_group[key].value
        except KeyError:
            self._MAX_V = None
        try:
            key = "Minimum Number of Grid Points"
            self._MIN_N = int(hdf5_group[key].value)
        except KeyError:
            self._MIN_N = None
        try:
            key = "Masses"
            self.masses = hdf5_group[key].value
        except KeyError:
            self.masses = None
        try:
            key = "Velocity Offset"
            self.offset = hdf5_group[key].value
        except KeyError:
            self.offset = None

        self.check_integrity(False)
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
        if self.form is not None:
            hdf5_group["Form"] = self.form
        if self.dim is not None:
            hdf5_group["Dimension"] = self.dim
        if self._MAX_V is not None:
            hdf5_group["Maximum Velocity"] = self._MAX_V
        if self._MIN_N is not None:
            hdf5_group["Minimum Number of Grid Points"] = self._MIN_N
        if self.masses is not None:
            hdf5_group["Masses"] = self.masses
        if self.offset is not None:
            hdf5_group["Velocity Offset"] = self.offset
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
        """
        self.check_parameters(form=self.form,
                              dimension=self.dim,
                              max_velocity=self._MAX_V,
                              min_points_per_axis=self._MIN_N,
                              masses=self.masses,
                              idx_helper=self._index,
                              vgrid_arr=self.vGrids,
                              idx_multigrid=self.iMG,
                              velocity_offset=self.offset,
                              complete_check=complete_check,
                              context=context)
        # Additional Conditions on instance:
        # All Velocity Grids must have equal boundaries
        # Todo there should be a better solution for this
        if self.vGrids is not None:
            for G in self.vGrids:
                assert isinstance(G, bp.Grid)
                assert np.array_equal(G.boundaries + self.offset,
                                      self.boundaries)
        return

    @staticmethod
    def check_parameters(form=None,
                         dimension=None,
                         max_velocity=None,
                         min_points_per_axis=None,
                         masses=None,
                         idx_helper=None,
                         vgrid_arr=None,
                         idx_multigrid=None,
                         velocity_offset=None,
                         complete_check=False,
                         context=None):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        form : :obj:`str`, optional
        dimension : :obj:`int`, optional
        max_velocity : :obj:`float`. optional
        min_points_per_axis : :obj:`int` optional
        masses : :obj:`~numpy.array` [:obj:`int`], optional
        idx_helper : :obj:`~numpy.array` [:obj:`int`], optional
        vgrid_arr : :obj:`~numpy.array` [:class:`Grid`], optional
        idx_multigrid : :obj:`~numpy.array` [:obj:`int`], optional
        velocity_offset : :obj:`~numpy.array` [:obj:`float`], optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        context : :class:`Simulation`, optional
            The Simulation, which this instance belongs to.
        """
        if context is not None:
            assert isinstance(context, bp.Simulation)

        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all(param_val is not None
                       for (param_key, param_val) in locals().items()
                       if param_key != "context")

        # check all parameters, if set
        if form is not None:
            assert isinstance(form, str)
            assert form in bp_c.SUPP_GRID_FORMS

        if dimension is not None:
            assert isinstance(dimension, int)
            assert dimension in bp_c.SUPP_GRID_DIMENSIONS
            if context is not None and context.p.dim is not None:
                assert dimension >= context.p.dim

        if max_velocity is not None:
            assert isinstance(max_velocity, float)
            assert max_velocity > 0

        if min_points_per_axis is not None:
            assert isinstance(min_points_per_axis, int)
            assert min_points_per_axis > 1

        if masses is not None:
            assert isinstance(masses, np.ndarray)
            assert masses.dtype == int
            assert masses.ndim == 1
            assert all(m >= 1 for m in masses)
            if context is not None:
                assert masses.size == context.s.size
                assert all(masses == context.s.mass)

        # Todo assert one is None, IFF all are None
        # number of specimen, simplifies upcoming checks
        if idx_helper is not None:
            assert isinstance(idx_helper, np.ndarray)
            assert idx_helper.dtype == int
            assert idx_helper.ndim == 1
            assert all(0 <= idx for idx in idx_helper)
            assert all(idx_helper[i + 1] - idx_helper[i] > 0
                       for i in range(idx_helper.size - 1))

        if vgrid_arr is not None:
            assert isinstance(vgrid_arr, np.ndarray)
            assert vgrid_arr.dtype == 'object'
            assert all(isinstance(grid, bp.Grid) for grid in vgrid_arr)
            for grid in vgrid_arr:
                grid.check_integrity()
            assert vgrid_arr.ndim == 1
            if context is not None:
                assert vgrid_arr.size == context.s.size
            if vgrid_arr.size > 0:
                if dimension is None:
                    dimension = vgrid_arr[0].dim
                    form = vgrid_arr[0].form
                assert all(grid.dim == dimension
                           and grid.form == form
                           for grid in vgrid_arr)
            if idx_helper is not None:
                assert vgrid_arr.size == idx_helper.size - 1
            if min_points_per_axis is not None:
                for vGrid in vgrid_arr:
                    assert all(n_i >= min_points_per_axis for n_i in vGrid.n)

        if idx_multigrid is not None:
            assert isinstance(idx_multigrid, np.ndarray)
            assert idx_multigrid.dtype == int
            assert idx_multigrid.ndim == 2
            if idx_helper is not None:
                assert idx_multigrid.shape[0] == idx_helper[-1]
            if vgrid_arr is not None:
                for (vGrid_idx, vGrid) in enumerate(vgrid_arr):
                    beg = idx_helper[vGrid_idx]
                    end = idx_helper[vGrid_idx + 1]
                    assert vGrid.size == end - beg
                    assert np.array_equal(vGrid.iG, idx_multigrid[beg: end])

        if velocity_offset is not None:
            if type(velocity_offset) in [list, float]:
                velocity_offset = np.array(velocity_offset,
                                           dtype=float)
            assert velocity_offset.dtype == float

        if dimension is not None and velocity_offset is not None:
            assert velocity_offset.shape == (dimension,)

        return

    def __str__(self,
                write_physical_grid=False):
        """Convert the instance to a string, describing all attributes."""
        description = ''
        description += "Dimension = {}\n".format(self.dim)
        description += "Geometric Form = {}\n".format(self.form)
        description += "Total Size = {}\n".format(self.size)
        description += "Offset = {}\n".format(self.offset)

        if self.vGrids is not None:
            for (i_G, vGrid) in enumerate(self.vGrids):
                description += 'Specimen_{s}, mass = {m}:' \
                               '\n\t'.format(s=i_G, m=self.masses[i_G])
                grid_str = vGrid.__str__(write_physical_grid)
                description += grid_str.replace('\n', '\n\t')
                description += '\n'
        return description[:-1]
