
from boltzmann.configuration import species as b_spc
from boltzmann.configuration import grid as b_grd
import boltzmann.constants as b_const

import numpy as np
import h5py

import math


class SVGrid:
    """Manages the Velocity Grids of all
    :class:`~boltzmann.configuration.Species` /
    :class:`~boltzmann.configuration.Specimen`.

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
        Geometric form of all Velocity :class:`Grids <configuration.Grid>`.
        Must be an element of
        :const:`~boltzmann.constants.SUPP_GRID_FORMS`.
    dim : :obj:`int`
        Dimensionality of all Velocity :class:`Grids <configuration.Grid>`.
        Must be in :const:`~boltzmann.constants.SUPP_GRID_DIMENSIONS`.
    vGrids : :obj:`np.ndarray` [:class:`~boltzmann.configuration.Grid`]
        Array of all Velocity :class:`Grids <boltzmann.configuration.Grid`>.
        Each Velocity Grids attribute
        :attr:`Grid.iG <boltzmann.configuration.Grid.iG>`
        links to its respective slice of :attr:`iMG`.
    iMG : :obj:`np.ndarray` [:obj:`int`]
        The *integer Multi-Grid*. It is a concatenation of the
        Velocity integer Grids
        (:attr:`Grid.iG <boltzmann.configuration.Grid>`) of all
        :class:`Species`.
        It describes the
        position/physical values (:attr:`pG`)
        of all Velocity Grid points.
        All entries are factors, such that
        :math:`pG = offset + iG \cdot d`.
        Note that some V-Grid points occur in multiple
        :class:`Velocity Grids <boltzmann.configuration.Grid>`.
        Array of shape (:attr:`size`, :attr:`dim`).
    offset : :obj:`np.ndarray` [:obj:`float`]
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
                 velocity_offset=None,
                 species_array=None):
        self.check_parameters(grid_form=grid_form,
                              grid_dimension=grid_dimension,
                              max_velocity=max_velocity,
                              min_points_per_axis=min_points_per_axis,
                              velocity_offset=velocity_offset)
        self.form = grid_form
        self.dim = grid_dimension
        self._MAX_V = max_velocity
        self._MIN_N = min_points_per_axis
        if velocity_offset is not None:
            self.offset = np.array(velocity_offset)
        else:
            if grid_dimension is not None:
                self.offset = np.zeros(grid_dimension, dtype=float)
            else:
                self.offset = None

        # The remaining attributes are calculated in setup() method
        # _index[i] denotes the beginning of the i-th velocity Grid in iMG
        self._index = None
        self.vGrids = None
        self.iMG = None

        self.setup(species_array)
        return

    # Todo figure out whats useful - this currently is not
    # def __getitem__(self, specimen_index):
    #     """Returns the indexed :obj:`~boltzmann.configuration.Grid`."""
    #     return self.vGrids[specimen_index]

    #####################################
    #           Properties              #
    #####################################
    @property
    def size(self):
        """:obj:`int` :
        Denotes the total number of Velocity-Grid points
        over all :class:`~boltzmann.configuration.Specimen`.
        """
        if self._index is None:
            return None
        else:
            return self._index[-1]

    @property
    def n_grids(self):
        """:obj:`int` :
        Denotes the number of different
        :class:`Velocity Grids <boltzmann.configuration.Grid>`.
        """
        return self.vGrids.size

    @property
    def pMG(self):
        """:obj:`np.ndarray` [:obj:`float`] :
        Construct the *physical Multi-Grid* (**time intense!**).

            The physical Multi-Grid pG denotes the physical values /
            position of all :class:`Velocity Grid` points.

                :math:`pG := offset + iG \cdot d`

            Array of shape (:attr:`size`, :attr:`dim`)
         """
        pMG = np.zeros(self.iMG.shape, dtype=float) + self.offset
        for (i_G, G) in enumerate(self.vGrids):
            [beg, end] = self.range_of_indices(i_G)
            pMG[beg: end] += G.pG
        return pMG

    @property
    def boundaries(self):
        """:obj:`np.ndarray` [:obj:`float`] :
        Denotes the minimum and maximum physical values
        in the :class:`Velocity Grids <boltzmann.configuration.Grid>`.

            All :class:`~boltzmann.configuration.Specimen`
            have equal boundaries. Thus it is calculated based on
            an arbitrary :class:`~boltzmann.configuration.Specimen`.
        """
        if self.vGrids is None:
            return None
        else:
            return self.offset + self.vGrids[0].boundaries

    #####################################
    #           Configuration           #
    #####################################
    def setup(self, species_array):
        r"""Automatically constructs the
        :class:`Velocity Grids <boltzmann.configuration.Grid>`
        for the given
        :class:`species_array <boltzmann.configuration.Species>`
        and initializes the related attributes:
        :attr:`~SVGrid.vGrids`, :attr:`~SVGrid.iMG`,
        :attr:`~SVGrid._index` and :attr:`~SVGrid.size`


        1. Calculate the minimum number of :ref:`segments <segment>` any
           :class:`Velocity Grids <boltzmann.configuration.Grid>`
           should have on any axis.
        2. Calculate the number of
           :ref:`complete segments <complete_segment>`, such that
           any :class:`Velocity Grids <boltzmann.configuration.Grid>`
           has at least the minimum number of :ref:`segments <segment>`
           on any axis.
        3. Calculate the shape and spacing of each
           :class:`Velocity Grids <boltzmann.configuration.Grid>`,
           based on its :class:`Specimens <boltzmann.configuration.Specimen>`
           :attr:`~boltzmann.configuration.Specimen.mass`.
        4. Initialize and
           :meth:`~boltzmann.configuration.Grid.centralize` the
           :class:`Velocity Grids <boltzmann.configuration.Grid>`
           and  store them into :attr:`vGrids`.
        5. Set up the remaining attributes based on the
           :class:`Velocity Grids <boltzmann.configuration.Grid>`.

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
            :class:`Velocity Grids <boltzmann.configuration.Grid>`.
            In the :class:`Velocity Grid <boltzmann.configuration.Grid>`
            of any :class:`~boltzmann.configuration.Specimen`,
            a single complete segment consists of exactly
            :attr:`~boltzmann.configuration.Specimen.mass` segments.
            Thus for any :class:`~boltzmann.configuration.Specimen`:

              :math:`n_{Segments} = n_{complete \: Segments} \cdot mass`


          * Heavier :class:`~boltzmann.configuration.Specimen`
            are more inert and move less.
            Thus, for any
            :class:`~boltzmann.configuration.Specimen` the
            :attr:`~boltzmann.configuration.Grid.spacing`
            of its :class:`Velocity Grid <boltzmann.configuration.Grid>`
            is inversely proportional to its
            :attr:`~boltzmann.configuration.Specimen.mass` :

              :math:`spacing[i] \sim \frac{1}{mass[i]}.`

          * A smaller :attr:`~boltzmann.configuration.Grid.spacing`
            leads to larger grids.
            Thus for any
            :class:`~boltzmann.configuration.Specimen` the size of its
            :class:`Velocity Grid <boltzmann.configuration.Grid>`
            grows cubic in its :attr:`~boltzmann.configuration.Specimen.mass`.

              :math:`size \sim n^{dim} \sim mass^{dim}`.

        Parameters
        ----------
        species_array : :obj:`~boltzmann.configuration.Species`
        """
        necessary_params = [self.form,
                            self.dim,
                            self._MAX_V,
                            self._MIN_N,
                            species_array]
        # only run setup, if all necessary parameters are set
        if any([val is None for val in necessary_params]):
            return

        # sanity check
        self.check_integrity(False)
        assert type(species_array) is b_spc.Species
        species_array.check_integrity()

        # Construct self.vGrids
        mass_array = species_array.mass
        min_mass = np.amin(mass_array)
        # Minimum number of segments (see docstring)
        min_segments = self._MIN_N - 1
        # Number of complete segments (see docstring)
        n_complete_segments = math.ceil(min_segments / min_mass)
        number_of_grid_points = [(n_complete_segments * mass) + 1
                                 for mass in mass_array]
        grid_shapes = [[n] * self.dim for n in number_of_grid_points]
        # spacing of the velocity grid for each specimen
        # Todo replace, allow Grid.init with number of points (array)
        spacings = [2 * self._MAX_V / (n_i - 1)
                    for n_i in number_of_grid_points]
        # Contains the velocity Grid of each specimen
        vGrids = [b_grd.Grid(grid_form=self.form,
                             grid_dimension=self.dim,
                             grid_shape=grid_shapes[i],
                             grid_spacing=spacings[i],
                             grid_is_centered=True)
                  for i in range(species_array.n)]
        self.vGrids = np.array(vGrids)

        # construct self._index
        self._index = np.zeros(species_array.n + 1,
                               dtype=int)
        for (i_G, vGrid) in enumerate(vGrids):
            self._index[i_G + 1] = self._index[i_G] + vGrid.size

        # construct self.iMG
        self.iMG = np.zeros((self.size, self.dim),
                            dtype=int)
        # store the integer Grids (vGrid.iG), by writing them
        # consecutively into the integer Multi-Grid (self.iMG)
        for (i_G, vGrid) in enumerate(self.vGrids):
            [beg, end] = self.range_of_indices(i_G)
            self.iMG[beg: end, :] = vGrid.iG[:, :]
            vGrid.iG = self.iMG[beg: end, :]

        self.check_integrity()
        return

    #####################################
    #               Indexing            #
    #####################################

    def range_of_indices(self, specimen_index):
        """ Returns the beginning and end delimiters
        of the indexed
        :class:`Specimens <boltzmann.configuration.Specimen>`
        Velocity integer
        :class:`Grid <boltzmann.configuration.Grid>`
        in :class:`iMG <boltzmann.configuration.SVGrid>`.

        Parameters
        ----------
        specimen_index : :obj:`int`
            Index of the :class:`Specimen` in the
            :class:`Specimen Array <Species>`.

        Returns
        -------
        :obj:`np.ndarray` [:obj:`int`] :
            Delimiters of the indexed
            :class:`Specimens <boltzmann.configuration.Specimen>`
            Velocity integer :class:`~boltzmann.configuration.Grid`
            in :attr:`iMG`.
        """
        return self._index[specimen_index: specimen_index+2]

    # Todo indexing needs to be checked/edited for new SVGrid class
    def get_index(self,
                  specimen_index,
                  grid_entry):
        """Returns position of given grid_entry in :attr:`SVGrid.iMG`

        Firstly, we assume the given grid_entry is an element
        of the given Specimens Velocity-Grid.
        Under this condition we generate its index in attr:`SVGrid.iMG`.
        Secondly, we counter-check if the indexed value matches the given
        value.
        If this is true, we return the index,
        otherwise we return None.

        Parameters
        ----------
        specimen_index : int
        grid_entry : array(int)
            Array of shape=(self.dim,).

        Returns
        -------
        int
            Index/Position of grid_entry in :attr:`~SVGrid.iMG`.
        """
        # Todo Throw exception if not a grid entry (instead of None)
        i_flat = 0
        # get vector-index, by reversing Grid.centralize() - method
        i_vec = np.array((grid_entry+self.vGrids[specimen_index].n - 1) // 2,
                         dtype=int)
        for _dim in range(self.dim):
            n_loc = self.vGrids[specimen_index].n[_dim]
            i_flat *= n_loc
            i_flat += i_vec[_dim]
        i_flat += self._index[specimen_index]
        if all(np.array(self.iMG[i_flat] == grid_entry)):
            return i_flat
        else:
            return None

    def get_specimen(self, velocity_index):
        """Returns Specimen (index) of given velocity index.

        Parameters
        ----------
        velocity_index : int

        Returns
        -------
        int
            Index of Specimen.
        """
        for i in range(self._index.size):
            if self._index[i] <= velocity_index < self._index[i+1]:
                return i
        msg = 'The given index ({}) points out of the boundaries of ' \
              'iMG, the concatenated Velocity Grid.'.format(velocity_index)
        raise KeyError(msg)

    #####################################
    #           Serialization           #
    #####################################
    # Todo read and write all Velocity Grids instead?
    def load(self,
             hdf5_group,
             hdf5_species_group=None):
        """Sets up the class:`SVGrid` instance,
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : h5py.Group
            :class:`Velocity Grids <SVGrid>` Group
            in the HDF5 :class:`~boltzmann.Configuration` file.
        hdf5_species_group : h5py.Group
            :class:`~boltzmann.configuration.Species` Group
            in the HDF5 :class:`~boltzmann.Configuration` file.
        """
        assert isinstance(hdf5_group, h5py.Group)
        if hdf5_species_group is not None:
            assert isinstance(hdf5_species_group, h5py.Group)
            species = b_spc.Species().load(hdf5_species_group)
        else:
            species = None

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
            self._MAX_V = hdf5_group["Maximum Velocity"].value
        except KeyError:
            self._MAX_V = None
        try:
            self._MIN_N = int(hdf5_group["Minimum Number of Grid Points"].value)
        except KeyError:
            self._MIN_N = None
        try:
            self.offset = hdf5_group["Velocity Offset"].value
        except KeyError:
            self.offset = None
        self.check_integrity(False)
        self.setup(species)
        return

    def save(self, hdf5_group):
        """Writes the main attributes of the :obj:`Grid` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : h5py.Group
            :class:`Velocity Grids <SVGrid>` Group
            in the HDF5 :class:`~boltzmann.Configuration` file.
        """
        self.check_integrity(False)
        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        # write all set attributes to file
        if self.form is not None:
            hdf5_group["Form"] = self.form
        if self.dim is not None:
            hdf5_group["Dimension"] = self.dim
        if self._MAX_V is not None:
            hdf5_group["Maximum Velocity"] = self._MAX_V
        if self._MIN_N is not None:
            hdf5_group["Minimum Number of Grid Points"] = self._MIN_N
        if self.offset is not None:
            hdf5_group["Velocity Offset"] = self.offset
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
                              max_velocity=self._MAX_V,
                              min_points_per_axis=self._MIN_N,
                              velocity_offset=self.offset,
                              multi_grid_indices=self._index,
                              multi_class_array=self.vGrids,
                              multi_grid_array=self.iMG,
                              complete_check=complete_check)
        # Additional Conditions on instance:
        # All Velocity Grids must have equal boundaries
        if self.vGrids is not None:
            for G in self.vGrids:
                assert isinstance(G, b_grd.Grid)
                assert np.array_equal(G.boundaries + self.offset,
                                      self.boundaries)
        return

    @staticmethod
    def check_parameters(grid_form=None,
                         grid_dimension=None,
                         multi_grid_indices=None,
                         max_velocity=None,
                         min_points_per_axis=None,
                         multi_class_array=None,
                         multi_grid_array=None,
                         velocity_offset=None,
                         complete_check=False):
        """Sanity Check.
        Checks integrity of given parameters and their interactions."""
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

        if max_velocity is not None:
            assert isinstance(max_velocity, float)
            assert max_velocity > 0

        if min_points_per_axis is not None:
            assert isinstance(min_points_per_axis, int)
            assert min_points_per_axis > 1

        if velocity_offset is not None:
            if type(velocity_offset) in [list, float]:
                velocity_offset = np.array(velocity_offset,
                                           dtype=float)
            assert velocity_offset.dtype == float

        if grid_dimension is not None and velocity_offset is not None:
            assert velocity_offset.shape == (grid_dimension,)

        # Todo undo the assert all
        # Todo -> submitting a single parameter must be allowed
        multi_grid_params = [multi_grid_indices,
                             multi_class_array,
                             multi_grid_array]
        if any(param is not None for param in multi_grid_params):
            assert all(param is not None for param in multi_grid_params)
            assert grid_dimension is not None
            assert grid_form is not None
            assert isinstance(multi_grid_indices, np.ndarray)
            assert isinstance(multi_class_array, np.ndarray)
            assert isinstance(multi_grid_array, np.ndarray)

            assert multi_grid_indices.dtype == int
            assert multi_grid_indices.shape == (multi_class_array.size + 1,)
            assert all(index >= 0 for index in multi_grid_indices)
            assert all(multi_grid_indices[i + 1] - multi_grid_indices[i] > 0
                       for i in range(multi_class_array.size))
            assert multi_class_array.dtype == 'object'
            assert multi_grid_array.dtype == int
            assert multi_grid_array.shape == (multi_grid_indices[-1],
                                              grid_dimension)

            for (i_G, vGrid) in enumerate(multi_class_array):
                assert isinstance(vGrid, b_grd.Grid)
                vGrid.check_integrity()
                assert vGrid.dim == grid_dimension
                assert vGrid.form == grid_form
                if min_points_per_axis is not None:
                    assert all(np.greater_equal(vGrid.n, min_points_per_axis))
                beg = multi_grid_indices[i_G]
                end = multi_grid_indices[i_G + 1]
                assert vGrid.size == end - beg
                assert np.array_equal(vGrid.iG, multi_grid_array[beg: end])

        return

    def __str__(self,
                write_physical_grid=False):
        """Converts the instance to a string, describing all attributes."""
        description = ''
        description += "Dimension = {}\n".format(self.dim)
        description += "Geometric Form = {}\n".format(self.form)
        description += "Total Size = {}\n".format(self.size)
        description += "Offset = {}\n".format(self.offset)

        for (i_G, vGrid) in enumerate(self.vGrids):
            description += 'Specimen_{}:\n\t'.format(i_G)
            description += vGrid.__str__().replace('\n', '\n\t')
            description += '\n'
        return description[:-1]
