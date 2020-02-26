import os
import h5py
import numpy as np

import boltzpy.helpers.file_addresses as h_adr
import boltzpy.helpers.TimeTracker as h_tt
import boltzpy.AnimatedFigure as bp_af
import boltzpy.compute as bp_cp
import boltzpy.output as bp_out
import boltzpy.constants as bp_c
import boltzpy as bp


class Simulation:
    r"""Handles all aspects of a single simulation.

    Each instance correlates to a single file
    in which all parameters and computation results are  stored.
    An instance can be completely restored from its file.



    .. todo::
        - write __isequal__ magic methods for configuration (and subclasses)
        - write unittests for save/load(__init__) methods
        - Add Knudsen Number Attribute or Property?

            * Add method to get candidate for characteristic length
            * show this char length in GUI
            * automatically calculate Knudsen number with this information

        - add attribute to svgrid, so that velocity arrays
          can be stored in 2d/3d shape
          (just create a link, with the right shape)
        - link Species and SVGrid somehow
          -> adding Species, after setting up SVGrid
          should delete SVGrid or at least update it

            * Idea: each class has an is_set_up flag
            * after any change -> check flags of depending classes
            * main classes need to be linked for that!
        - Figure out nice way to implement boundary points
        - speed up init of psv grid <- ufunc's
        - @choose_rule: implement different 'shapes' to apply rules
          (e.g. a line with specified width,
          a ball with specified radius/diameter, ..).
          Switch between convex hull and span?
        - sphinx: link PSV-Grid to Calculation.data?
          link init_arr to Calculation.init_arr? No?
          in Initialization-Docstring
        - Add former block_index functionality for boundary points again
            * sort rule_arr and init_arr
            * set up reflection methods -> depends on position
                -> multiplies number of boundary rules
            * move into initialization module

    Notes
    -----
        * :attr:`t.iG` denotes the time steps
          when the results are written to the HDF5 file.
        * :attr:`t.spacing` denotes the number of calculation steps
          between two writes.

    Parameters
    ----------
    file_address : :obj:`str`, optional
        Address of the simulation file.
        Can be either a full path, a base file name or a file root.
        If no full path is given, then the file is placed in the
        :attr:`~boltzpy.constants.DEFAULT_DIRECTORY`.

    Attributes
    ----------
    s : :class:`Species`
        The simulated Specimen.
    t : :class:`Grid`
        The Time Grid.
    p : :class:`Grid`
        Position-Space Grid
    geometry: :class:`Geometry`
        Describes the behaviour for all position points.
        Contains the :class:`initialization rules <Rule>`
    sv : :class:`SVGrid`
        Velocity-Space Grids of all Specimen.
    scheme : :class:`Scheme`
        Contains all computation scheme parameters.
    output_parameters : :obj:`~numpy.array` [:obj:`str`]
        Output/Results of the Simulation.
        Each element must be in :const:`~boltzpy.constants.SUPP_OUTPUT`.
        Must be a 2D array.
    """

    def __init__(self, file_address=None):
        # set file address (using a setter method)
        [self._file_directory, self._file_root] = ['', '']
        self.file_address = file_address
        del file_address  # not needed anymore, could lead to typos

        # Open HDF5 file
        if os.path.exists(self.file_address + '.hdf5'):
            file = h5py.File(self.file_address + '.hdf5', mode='r')
        else:
            file = h5py.File(self.file_address + '.hdf5', mode='w-')
            file.attrs["class"] = "Simulation"

        #######################
        #   Grid Attributes   #
        #######################
        # load Species
        try:
            key = "Species"
            self.s = bp.Species.load(file[key])
        except KeyError:
            self.s = bp.Species()

        # load Time Grid
        try:
            key = "Time_Grid"
            self.t = bp.Grid.load(file[key])
        except KeyError:
            self.t = bp.Grid()
        self.t.ndim = 1

        # load Position Grid
        try:
            key = "Position_Grid"
            self.p = bp.Grid().load(file[key])
        except KeyError:
            self.p = bp.Grid()

        # load geometry
        try:
            key = "Geometry"
            self.geometry = bp.Geometry.load(file[key])
        except KeyError:
            self.geometry = bp.Geometry()

        # load Velocity Grids
        try:
            key = "Velocity_Grids"
            self.sv = bp.SVGrid.load(file[key])
        except KeyError:
            self.sv = bp.SVGrid()

        # load Collisions, if any
        try:
            key = "Collisions"
            self.coll = bp.Collisions.load(file[key])
        except KeyError:
            self.coll = bp.Collisions()

        # load Scheme
        try:
            key = "Scheme"
            self.scheme = bp.Scheme.load(file[key])
        except KeyError:
            self.scheme = bp.Scheme()
        # load output params Todo Move these into Scheme, as a (1D) Set
        try:
            key = "Computation/Output_Parameters"
            shape = file[key].attrs["shape"]
            self.output_parameters = file[key][()].reshape(shape)
        except KeyError:
            self.output_parameters = np.array([['Mass',
                                                'Momentum_X'],
                                               ['Momentum_X',
                                                'Momentum_Flow_X'],
                                               ['Energy',
                                                'Energy_Flow_X']])
        file.close()
        # Todo remove this, replace usage by geometry
        self.init_arr = self.geometry.init_array
        self.rule_arr = self.geometry.rules
        self.check_integrity(complete_check=False)
        return

    @property
    def file_address(self):
        """:obj:`str` :
        Full path of the :class:`Simulation` file.
        """
        return self._file_directory + self._file_root

    @file_address.setter
    def file_address(self, new_file_address):
        # separate file directory and file root, using a helper function
        # This standardizes the naming scheme
        h_separate = h_adr.split_address
        [new_file_directory, new_file_root] = h_separate(new_file_address)
        new_file_address = new_file_directory + new_file_root
        # Sanity check on (standardized) file address
        self.check_parameters(file_address=new_file_address)
        # change file_address
        [self._file_directory, self._file_root] = [new_file_directory,
                                                   new_file_root]
        self.check_parameters(file_address=self.file_address)
        return

    @property
    def n_rules(self):
        """:obj:`int` :
        Total number of :class:`initialization rules <Rule>` set up so far.
        """
        return self.geometry.rules.size

    @property
    def is_configured(self):
        """:obj:`bool` :
        True, if all necessary attributes of the instance are set.
        False Otherwise.
        """
        # Todo add output_parameters
        # Todo add initial_distribution / rule_arr
        return (self.s.is_configured
                and self.t.is_configured
                and self.p.is_configured
                and self.sv.is_configured
                and self.scheme.is_configured)

    @property
    def is_set_up(self):
        """:obj:`bool` :
        True, if the instance is completely set up and ready to call :meth:`~Simulation.run_computation`.
        False Otherwise.
        """
        # Todo add initial_distribution
        return (self.t.is_set_up
                and self.p.is_set_up
                and self.geometry.is_set_up
                and self.sv.is_set_up
                and self.coll.is_set_up)

    #############################
    #       Configuration       #
    #############################
    def add_specimen(self,
                     name=None,
                     mass=None,
                     collision_rate=None,
                     color=None):
        """Add a :class:`Specimen` to :attr:`s`.
        See :meth:`Species.add`

        Parameters
        ----------
        name : :obj:`str`, optional
        mass : :obj:`int`, optional
        collision_rate : :obj:`~numpy.array` [:obj:`float`] or :obj:`list` [:obj:`int`], optional
            Correlates to the collision probability between two specimen.
        color : :obj:`str`, optional
        """
        # Todo alternative type of collision_rate is really a list of int?
        # Todo  not a list of float?
        # Todo  this is also implemented in the asserts in the beginning
        # Todo  this also applies to edit Specimen
        if isinstance(collision_rate, list):
            assert all([isinstance(item, int) for item in collision_rate])
            collision_rate = np.array(collision_rate, dtype=float)
        self.s.add(name,
                   mass,
                   collision_rate,
                   color, )
        return

    def edit_specimen(self,
                      item,
                      name=None,
                      mass=None,
                      collision_rate=None,
                      color=None):
        """Edit the :class:`Specimen`, denoted by *item*, in :attr:`s`.
        See :meth:`Species.edit`

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or name of the :obj:`Specimen` to be edited
        name : :obj:`str`, optional
        mass : :obj:`int`, optional
        collision_rate : :obj:`~numpy.array` [:obj:`float`] or :obj:`list` [:obj:`int`], optional
            Correlates to the collision probability between two specimen.
        color : :obj:`str`, optional
        """
        if isinstance(collision_rate, list):
            assert all([isinstance(item, int) for item in collision_rate])
            collision_rate = np.array(collision_rate, dtype=float)
        self.s.edit(item,
                    name,
                    mass,
                    collision_rate,
                    color)
        return

    def remove_specimen(self, item):
        """Remove the :class:`Specimen`, denoted by *item*,
        from :attr:`s`.
        See :meth:`Species.remove`

        Parameters
        ----------
        item : :obj:`int` or :obj:`str`
            Index or name of the :obj:`Specimen` to be edited
        """
        self.s.remove(item)
        return

    # Todo Choose between step size or number of time steps
    # Todo remove calculations per time step -> adaptive RungeKutta
    def setup_time_grid(self,
                        max_time,
                        number_time_steps,
                        calculations_per_time_step=1):
        """Set up :attr:`t`.

        Calculate step size and call :class:`Grid() <Grid>`.

        Parameters
        ----------
        max_time : :obj:`float`
        number_time_steps : :obj:`int`
        calculations_per_time_step : :obj:`int`
        """
        step_size = max_time / (number_time_steps - 1)
        self.t = bp.Grid(ndim=1,
                         shape=(number_time_steps,),
                         physical_spacing=step_size,
                         spacing=calculations_per_time_step)
        return

    def setup_position_grid(self,
                            grid_dimension,
                            grid_shape,
                            grid_spacing):
        """Set up :attr:`p` and adjust :attr:`geometry` to the new shape.
        See :class:`Grid() <Grid>`

        Parameters
        ----------
        grid_dimension : :obj:`int`
        grid_shape : :obj:`tuple` [:obj:`int`]
        grid_spacing : :obj:`float`
        """
        self.p = bp.Grid(ndim=grid_dimension,
                         shape=grid_shape,
                         physical_spacing=grid_spacing)
        # Update shape of initialization_array
        self.geometry.shape = self.p.shape
        return

    def set_velocity_grids(self,
                           grid_dimension,
                           maximum_velocity,
                           shapes,
                           use_identical_spacing=False):
        """Set up :attr:`sv`.

        1. Generate a minimal Velocity :class:`Grid`.
        2. Use the minimal Grid as prototype in :meth:`SVGrid.setup`
           and setup the Velocity Grids for all :class:`Species`.

        Parameters
        ----------
        grid_dimension : :obj:`int`
        maximum_velocity : :obj:`float`
        shapes : :obj:`list` [:obj:`tuple` [:obj:`int`]]
        use_identical_spacing : :obj:`bool`, optional
            If True, then all specimen use equal grids.
            If False, then the spacing is adjusted to the mass ratio.
        """
        if not self.s.is_configured:
            raise AttributeError
        spacings = bp.SVGrid.generate_spacings(self.s.mass,
                                               use_identical_spacing)
        self.sv = bp.SVGrid(ndim=grid_dimension,
                            maximum_velocity=maximum_velocity,
                            shapes=shapes,
                            spacings=spacings)
        return

    #####################################
    #            Computation            #
    #####################################
    def compute(self, hdf5_group_name="Computation"):
        """Compute the fully configured Simulation"""
        self.check_integrity()
        # Todo write hash function in Computation folder
        #     file = h5py.File(self.file_address + '.hdf5')
        #     # hash = file["Computation"].attrs["Hash_Value"]
        #     # Todo define hashing method
        #     assert hash == self.__hash__()
        #     print("The saved results are up to date!"
        #           "A new computation is not necessary")
        #     return
        # else (KeyError, AssertionError):

        # Generate Computation data
        data = bp.Data(self.file_address)
        data.check_stability_conditions()

        # Generate output functions
        # Todo move into Scheme
        f_output = bp_out.generate_output_function(self, hdf5_group_name)

        # Start computation
        print('Computing:')
        time_tracker = h_tt.TimeTracker()
        # Todo this might be buggy, if data.tG changes
        # Todo e.g. in adaptive time schemes
        # Todo proposition: iterate over length?
        for (tw_idx, tw) in enumerate(data.tG[:, 0]):
            while data.t != tw:
                bp_cp.operator_splitting(data,
                                         self.geometry.transport,
                                         self.geometry.collision)
            # print time estimate
            time_tracker.print(tw, data.tG[-1, 0])
            # generate Output and write it to disk
            f_output(data, tw_idx)
        return

    #####################################
    #             Animation             #
    #####################################
    def animate(self, shape=(3, 2)):
        figure = bp_af.AnimatedFigure(tmax=self.t.size)
        specimen_names = [specimen.name for specimen in self.s.specimen_arr]
        moments = self.output_parameters.flatten()
        file = h5py.File(self.file_address + '.hdf5', mode='r')
        hdf5_group = file["Computation"]
        for (moment_idx, moment) in enumerate(moments):
            ax = figure.add_subplot(shape + (1 + moment_idx,),
                                    title=moment
                                    )
            # Todo This flatten() should NOT be necessary, fix with model/geometry
            xdata = (self.p.iG * self.p.delta).flatten()[1:-1]
            for (specimen_idx, specimen) in enumerate(specimen_names):
                ydata = hdf5_group[moment][..., 1:-1, specimen_idx]
                ax.plot(xdata, ydata)
        figure.save(self.file_address + '.mp4')
        return

    #####################################
    #           Serialization           #
    #####################################
    # Todo Create __is_equal__ method, compare to default  params -> dont save
    def save(self, file_address=None):
        """Write all parameters of the :class:`Simulation` instance
        to a HDF5 file.

        If a *file_address* is given, then this method works as a 'save as'.
        In this case the :attr:`file_address` of the instance is changed
        and the newly named instance is saved to a new .hdf5 file.

        Parameters
        ----------
        file_address : :obj:`str`, optional
            Change the instances :attr:`file_address` to this value.

            Is either a full path, a base file name or a file root.
            If it is a base file name or a file root,
            then the file is placed in the
            :attr:`~boltzpy.constants.DEFAULT_DIRECTORY`.
        """
        # Change Simulation.file_name, if file_address is given
        if file_address is not None:
            self.file_address = file_address
        del file_address

        # Sanity Check before saving
        self.check_integrity(False)

        # Todo if sv and collision parameters are the same -> keep Collisions
        # Create new HDF5 file (deletes old data, if any)
        file = h5py.File(self.file_address + ".hdf5", mode='w')
        file.attrs["class"] = "Simulation"

        # Save Species
        key = "Species"
        file.create_group(key)
        self.s.save(file[key])

        # Save Time Grid
        key = "Time_Grid"
        file.create_group(key)
        self.t.save(file[key])

        # Save Position Grid
        key = "Position_Grid"
        file.create_group(key)
        self.p.save(file[key])

        # Save Geometry
        key = "Geometry"
        file.create_group(key)
        self.geometry.save(file[key])

        # Save Velocity Grids
        key = "Velocity_Grids"
        file.create_group(key)
        self.sv.save(file[key])

        # Save Collisions
        key = "Collisions"
        file.create_group(key)
        self.coll.save(file[key])

        # Save Computation Parameters
        if "Scheme" not in file.keys():
            file.create_group("Scheme")
        # Save scheme
        self.scheme.save(file["Scheme"])

        if self.output_parameters is not None:
            #  noinspection PyUnresolvedReferences
            h5py_string_type = h5py.special_dtype(vlen=str)
            key = "Computation/Output_Parameters"
            file[key] = np.array(self.output_parameters,
                                 dtype=h5py_string_type).flatten()
            file[key].attrs["shape"] = self.output_parameters.shape

        file.close()
        return

    #####################################
    #            Verification           #
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
        self.check_parameters(file_address=self.file_address,
                              species=self.s,
                              time_grid=self.t,
                              position_grid=self.p,
                              species_velocity_grid=self.sv,
                              geometry=self.geometry,
                              output_parameters=self.output_parameters,
                              scheme=self.scheme,
                              complete_check=complete_check,
                              context=self)
        return

    @staticmethod
    def check_parameters(file_address=None,
                         species=None,
                         time_grid=None,
                         position_grid=None,
                         species_velocity_grid=None,
                         geometry=None,
                         output_parameters=None,
                         scheme=None,
                         complete_check=False,
                         context=None):
        r"""Sanity Check.

        Check integrity of given parameters and their interaction.

        Parameters
        ----------
        file_address : :obj:`str`, optional
        species : :obj:`Species`, optional
        time_grid : :obj:`Grid`, optional
        position_grid : :obj:`Grid`, optional
        species_velocity_grid : :obj:`SVGrid`, optional
        geometry: :class:`Geometry`, optional
        output_parameters : :obj:`~numpy.array` [:obj:`str`], optional
        scheme : :class:`Scheme`, optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be assigned (not None).
            If False, then unassigned parameters are ignored.
        context : :class:`Simulation`, optional
            The Simulation instance, allows checking related attributes.
        """
        if context is not None:
            assert isinstance(context, Simulation)

        # For complete check, assert that all parameters are assigned
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all([param is not None for param in locals().values()])

        # check all parameters, if set
        if file_address is not None:
            assert isinstance(file_address, str)
            pos_of_file_root = file_address.rfind("/") + 1
            file_root = file_address[pos_of_file_root:]
            file_directory = file_address[0:pos_of_file_root]
            # assert non empty root and directory
            assert file_directory != ""
            assert file_root != ""
            # assert the file directory exists and is a directory
            assert (os.path.exists(file_directory)
                    and os.path.isdir(file_directory)), \
                "No such directory: {}".format(file_directory)
            # assert write access to directory
            assert os.access(file_directory, os.W_OK), \
                "No write access to directory {}".format(file_directory)
            # Assert Validity of characters
            for char in bp_c.INVALID_CHARACTERS:
                assert char not in file_root, \
                    "The provided file root is invalid:\n" \
                    "It contains invalid characters: '{}'" \
                    "{}".format(char, file_root)
                if char == "/":  # ignore '/' for directory
                    continue
                assert char not in file_directory, \
                    "The provided file directory is invalid:\n" \
                    "It contains invalid characters: '{}'" \
                    "{}".format(char, file_directory)
            # Assert file is a simulation file
            if os.path.exists(file_directory + file_root + '.hdf5'):
                hdf5_file = h5py.File(file_directory + file_root + '.hdf5', 'r')
                assert hdf5_file.attrs["class"] == "Simulation"

        if species is not None:
            assert isinstance(species, bp.Species)
            species.check_integrity()

        if time_grid is not None:
            assert isinstance(time_grid, bp.Grid)
            time_grid.check_integrity(complete_check)
            assert time_grid.ndim == 1

        if position_grid is not None:
            assert isinstance(position_grid, bp.Grid)
            position_grid.check_integrity(complete_check)
            # Todo Remove this, when implementing 2D Transport
            if position_grid.ndim is not None \
                    and position_grid.ndim != 1:
                msg = "Currently only 1D Simulations are supported!"
                raise NotImplementedError(msg)

        if species_velocity_grid is not None:
            assert isinstance(species_velocity_grid, bp.SVGrid)
            species_velocity_grid.check_integrity(complete_check,
                                                  context)

        if geometry is not None:
            assert isinstance(geometry, bp.Geometry)
            geometry.check_integrity(complete_check,
                                     context)

        if output_parameters is not None:
            assert isinstance(output_parameters, np.ndarray)
            assert len(output_parameters.shape) == 2
            assert all([mom in bp_c.SUPP_OUTPUT
                        for mom in output_parameters.flatten()])

        if scheme is not None:
            scheme.check_integrity(complete_check)
        return

    def __eq__(self, other):
        if not isinstance(other, Simulation):
            return False
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            return False
        for (key, value) in self.__dict__.items():
            other_value = other.__dict__[key]
            if type(value) != type(other_value):
                return False
            if isinstance(value, np.ndarray):
                if np.any(value != other_value):
                    return False
            else:
                if value != other_value:
                    return False
        return True

    def __str__(self,
                write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance.
        """
        description = ''
        description += 'Simulation File = ' + self.file_address + '.hdf5\n'
        description += 'Species\n'
        description += '-------\n'
        description += '\t' + self.s.__str__().replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Time Data\n'
        description += '---------\n'
        time_str = self.t.__str__(write_physical_grids)
        description += '\t' + time_str.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Position-Space Data\n'
        description += '-------------------\n'
        position_str = self.p.__str__(write_physical_grids)
        description += '\t' + position_str.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Velocity-Space Data\n'
        description += '-------------------\n'
        velocity_str = self.sv.__str__(write_physical_grids)
        description += '\t' + velocity_str.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Geometry\n'
        description += '--------\n'
        geometry_str = self.geometry.__str__()
        description += '\t' + geometry_str.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Computation Scheme\n'
        description += '------------------\n\t'
        description += self.scheme.__str__().replace('\n', '\n\t')
        description += '\n'
        description += 'Animated Moments\n'
        description += '----------------\n\t'
        output_str = self.output_parameters.__str__().replace('\n', '\n\t')
        description += output_str + '\n'
        return description
