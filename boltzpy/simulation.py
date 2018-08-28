import boltzpy.constants as b_const
import boltzpy.species as b_spc
import boltzpy.specimen as b_spm
import boltzpy.grid as b_grd
import boltzpy.svgrid as b_svg
import boltzpy.rule as b_rul
import boltzpy.computation as b_run
import boltzpy.animation as b_ani

import boltzpy.helpers.file_addresses as h_file

import numpy as np
import os
import h5py
from datetime import datetime


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
        * :attr:`t.multi` denotes the number of calculation steps
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
    sv : :class:`SVGrid`
        Velocity-Space Grids of all Specimen.
    rule_arr : :obj:`~numpy.array` [:class:`Rule`]
        Array of the specified :class:`initialization rules <Rule>`.
    init_arr : :obj:`~numpy.array` [:obj:`int`]
        Links each :attr:`p`
        :class:`~boltzpy.Grid` point
        to its :class:`initialization rule <Rule>`.
        Contains the indices of the respective :class:`rules <Rule>`
    coll_select_scheme : :obj:`str`
        Selection scheme for the collision partners.
        Must be an element of
        :const:`~boltzpy.constants.SUPP_COLL_SELECTION_SCHEMES`
    coll_substeps : :obj:`int`
        Number of collision substeps per time step.
    order_os : :obj:`int`
        Approximation order of Operator Splitting.
        Must be in :const:`~boltzpy.constants.SUPP_ORDERS_OS`.
    order_coll : :obj:`int`
        Approximation order of approximation of the Collision Operator.
        Must be in
        :const:`~boltzpy.constants.SUPP_ORDERS_COLL`.
    order_transp : :obj:`int`
        Approximation order of transport solver.
        Must be in
        :const:`~boltzpy.constants.SUPP_ORDERS_TRANSP`.
    output_parameters : :obj:`~numpy.array` [:obj:`str`]
        Output/Results of the Simulation.
        Each element must be in :const:`~boltzpy.constants.SUPP_OUTPUT`.
        Must be a 2D array.
    """
    def __init__(self, file_address=None):
        # separate file directory and file root, using a helper function
        h_separate = h_file.split_address
        [self._file_directory, self._file_root] = h_separate(file_address)

        # Sanity check on (processed) file address
        self.check_parameters(file_address=self.file_address)
        del file_address    # not needed anymore, could lead to typos

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
            self.s = b_spc.Species.load(file[key])
        except KeyError:
            self.s = b_spc.Species()

        # load Time Grid
        try:
            key = "Time_Grid"
            self.t = b_grd.Grid.load(file[key])
        except KeyError:
            self.t = b_grd.Grid()
        self.t.dim = 1

        # load Position Grid
        try:
            key = "Position_Grid"
            self.p = b_grd.Grid().load(file[key])
        except KeyError:
            self.p = b_grd.Grid()

        # load Velocity Grids
        try:
            key = "Velocity_Grids"
            self.sv = b_svg.SVGrid.load(file[key])
            self.sv.setup(self.s)
        except KeyError:
            self.sv = b_svg.SVGrid()

        #################################
        #   Initialization Attributes   #
        #################################
        # load initialization rules
        try:
            hdf5_group = file["Initialization"]
            # Number of Rules is stored in group attributes
            n_rules = int(hdf5_group.attrs["Number of Rules"])
            # store rules in here iteratively
            self.rule_arr = np.empty(shape=(n_rules,), dtype=b_rul.Rule)
            # iteratively read the rules
            for rule_idx in range(n_rules):
                key = "Rule_" + str(rule_idx)
                self.rule_arr[rule_idx] = b_rul.Rule.load(hdf5_group[key])
        except KeyError:
            self.rule_arr = np.empty(shape=(0,), dtype=b_rul.Rule)

        # load initialization array
        try:
            key = "Initialization/Initialization Array"
            self.init_arr = file[key].value
        except KeyError:
            # default value -1 <=> no initialization rule assigned here
            self.init_arr = np.full(shape=self.p.size,
                                    fill_value=-1,
                                    dtype=int)

        ##############################
        #   Computation Attributes   #
        ##############################
        try:
            key = "Collision_Selection_Scheme"
            self.coll_select_scheme = file["Collisions"].attrs[key].value
        except KeyError:
            self.coll_select_scheme = 'Complete'
        # Todo replace this by Knudsen Number + adaptive RungeKutta
        try:
            key = "Collision_Substeps"
            self.coll_substeps = int(file["Computation"].attrs[key])
        except KeyError:
            self.coll_substeps = 1
        try:
            key = "Convergence_Order_Operator_Splitting"
            self.order_os = int(file["Computation"].attrs[key])
        except KeyError:
            self.order_os = 1
        try:
            key = "Convergence_Order_Transport"
            self.order_transp = int(file["Computation"].attrs[key])
        except KeyError:
            self.order_transp = 1
        try:
            key = "Convergence_Order_Collision_Operator"
            self.order_coll = int(file["Computation"].attrs[key])
        except KeyError:
            self.order_coll = 1
        try:
            key = "Computation/Output_Parameters"
            shape = file[key].attrs["shape"]
            self.output_parameters = file[key].value.reshape(shape)
        except KeyError:
            self.output_parameters = np.array([['Mass',
                                                'Momentum_X'],
                                               ['Momentum_X',
                                                'Momentum_Flow_X'],
                                               ['Energy',
                                                'Energy_Flow_X']])
        # # Submodules
        # # Todo self.description = ... String that describes the simulation
        # self.collision_relations = b_rel.CollisionRelations(self)
        # # Todo self.schemes, get from configuration
        # self.computation = b_cal.Calculation(self)
        # self.animation = b_ani.Animation(self)

        file.close()
        self.check_integrity(complete_check=False)
        return

    @property
    def file_address(self):
        """:obj:`str` :
        Full path of the :class:`Simulation` file.
        """
        return self._file_directory + self._file_root

    @property
    def n_rules(self):
        """:obj:`int` :
        Total number of :class:`initialization rules <Rule>` set up so far.
        """
        return self.rule_arr.size

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
        mass : int, optional
        collision_rate : :obj:`~numpy.array` [:obj:`float`] or :obj:`list` [:obj:`int`], optional
            Correlates to the collision probability between two specimen.
        color : :obj:`str`, optional
        """
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
        mass : int, optional
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
        self.t = b_grd.Grid(grid_form='rectangular',
                            grid_dimension=1,
                            grid_shape=np.array([number_time_steps]),
                            grid_spacing=step_size,
                            grid_multiplicator=calculations_per_time_step)
        return

    def setup_position_grid(self,
                            grid_dimension,
                            grid_shape,
                            grid_spacing):
        """Set up :attr:`p` and adjust :attr:`init_arr` to the new shape.
        See :class:`Grid() <Grid>`

        Parameters
        ----------
        grid_dimension : :obj:`int`
        grid_shape : :obj:`~numpy.array`  [:obj:`int`] or :obj:`list` [:obj:`int`]
        grid_spacing : :obj:`float`
        """
        if isinstance(grid_shape, list):
            assert all([isinstance(item, int) for item in grid_shape])
            grid_shape = np.array(grid_shape, dtype=int)
        self.p = b_grd.Grid(grid_form='rectangular',
                            grid_dimension=grid_dimension,
                            grid_shape=grid_shape,
                            grid_spacing=grid_spacing)
        # Update shape of initialization_array
        self.init_arr = np.full(shape=self.p.size,
                                fill_value=-1,
                                dtype=int)
        return

    # Todo Crosscheck offset -> very weird for boundaries / complex spaces
    # Todo                      is it a moving camera?
    def set_velocity_grids(self,
                           grid_dimension,
                           min_points_per_axis,
                           max_velocity,
                           grid_form='rectangular',
                           velocity_offset=None):
        """Set up :attr:`sv`.

        1. Generate a minimal Velocity :class:`Grid`.
        2. Use the minimal Grid as prototype in :meth:`SVGrid.setup`
           and setup the Velocity Grids for all :class:`Species`.

        Parameters
        ----------
        grid_dimension : :obj:`int`
        min_points_per_axis : :obj:`int`
        max_velocity : :obj:`float`
        grid_form : :obj:`str`, optional
        velocity_offset : :obj:`~numpy.array` [:obj:`float`], optional
        """
        self.sv = b_svg.SVGrid(grid_form=grid_form,
                               grid_dimension=grid_dimension,
                               min_points_per_axis=min_points_per_axis,
                               max_velocity=max_velocity,
                               velocity_offset=velocity_offset,
                               species_array=self.s)
        return

    def add_rule(self,
                 category,
                 rho,
                 drift,
                 temp,
                 name=None,
                 color=None):
        """Add a new :class:`initialization rule <Rule>` to :attr:`rule_arr`.

        Parameters
        ----------
        category : :obj:`str`
            Category of the :class:`P-Grid <boltzpy.Grid>` point.
            Must be in
            :const:`~boltzpy.constants.SUPP_GRID_POINT_CATEGORIES`.
        rho : :obj:`~numpy.array` [:obj:`float`]
        drift : :obj:`~numpy.array` [:obj:`float`]
        temp : :obj:`~numpy.array` [:obj:`float`]
        name : str, optional
            Displayed in the GUI to visualize the initialization.
        color : str, optional
            Displayed in the GUI to visualize the initialization.
        """
        b_rul.Rule.check_parameters(category=category,
                                    rho=rho,
                                    drift=drift,
                                    temp=temp)
        new_rule = b_rul.Rule(category,
                              rho,
                              drift,
                              temp,
                              name,
                              color)
        self.rule_arr = np.append(self.rule_arr, [new_rule])
        self.check_parameters(species=self.s,
                              species_velocity_grid=self.sv,
                              initialization_rules=self.rule_arr)
        return

    def choose_rule(self,
                    array_of_grid_point_indices,
                    rule_index):
        """Let the given :class:`P-Grid <boltzpy.Grid>` points
        be initialized with the specified
        :class:`initialization rule <Rule>`.

        Sets the :attr:`init_arr` entries
        of all  :class:`P-Grid <boltzpy.Grid>` points
        in *array_of_grid_point_indices* to *rule_index*.

        Parameters
        ----------
        array_of_grid_point_indices : :obj:`~numpy.array` [:obj:`int`]
            Contains flat indices of
            :class:`P-Grid <boltzpy.Grid>` points.
        rule_index : :obj:`int`
            Index of a :class:`initialization rule <Rule>`
            in :attr:`rule_arr`.
        """
        assert isinstance(array_of_grid_point_indices, np.ndarray)
        assert array_of_grid_point_indices.dtype == int
        assert np.min(array_of_grid_point_indices) >= 0
        assert np.max(array_of_grid_point_indices) < self.p.size
        assert isinstance(rule_index, int)
        assert 0 <= rule_index < self.n_rules

        for p in array_of_grid_point_indices:
            self.init_arr[p] = rule_index
        return

    ###########################
    #       Computation       #
    ###########################
    # Todo rework computation module
    def run_computation(self):
        """Compute the fully configured Simulation"""
        self.check_integrity()
        # Todo write hash function in Computation folder
        try:
            file = h5py.File(self.file_address + '.hdf5')
            hash = file["Computation"].attrs["Hash_Value"]
            # Todo define hashing method
            assert hash == self.__hash__()
            print("Computation was already done!")
            return
        except (KeyError, AssertionError):
            Calculation = b_run.Calculation(self)
            Calculation.run()
        return

    # Todo rework animation module
    def create_animation(self,
                         output_arr=None,
                         specimen_arr=None):
        """Create animated plot and saves it to disk.

        Sets up :obj:`matplotlib.animation.FuncAnimation` instance
        based on a figure with separate subplots
        for every moment in *output_arr*
        and separate lines for every specimen and moment.
        """
        # Todo Assert Computation ran successfully
        if output_arr is None:
            output_arr = self.output_parameters
        else:
            assert isinstance(output_arr, np.ndarray)
            assert output_arr.ndim == 2
            assert all([output in self.output_parameters.flatten()
                        for output in output_arr.flatten()])
        if specimen_arr is None:
            specimen_arr = self.s.specimen_arr
        else:
            assert isinstance(specimen_arr, np.ndarray)
            assert specimen_arr.ndim == 1
            assert all([isinstance(specimen, b_spm.Specimen)
                        for specimen in specimen_arr])
            assert all([specimen.name in self.s.names
                        for specimen in self.s])
        animation = b_ani.Animation(self)
        animation.animate(output_arr, specimen_arr)
        return

    # Todo choose proper parameters
    def create_snapshot(self,
                        time_step,
                        output_arr=None,
                        specimen_arr=None,
                        snapshot_name=None,
                        # Todo p_space -> cut of boundary effects,
                        # Todo color -> color some Specimen differently,
                        # Todo legend -> setup legend in the image?
                        ):
        """Creates a vector plot of the simulation
        at the desired time_step.
        Saves the file as *snapshot_name*.eps in the Simulation folder.

        Parameters
        ----------
        time_step : :obj:`int`
        output_arr : :obj:`~numpy.array` [:obj:`str`], optional
        specimen_arr : :obj:`~numpy.array` [:class:`~boltzpy.Specimen`], optional
        snapshot_name : :obj:`str`, optional
            File name of the vector image.
        """
        # Todo Reasonable asserts for time_step

        if output_arr is None:
            output_arr = self.output_parameters
        else:
            assert isinstance(output_arr, np.ndarray)
            assert all([output in self.output_parameters.flatten()
                        for output in output_arr])
        if specimen_arr is None:
            specimen_arr = self.s.specimen_arr
        else:
            assert isinstance(specimen_arr, np.ndarray)
            assert specimen_arr.ndim == 1
            assert all([isinstance(specimen, b_spm.Specimen)
                        for specimen in specimen_arr])
            assert all([specimen.name in self.s.names
                        for specimen in self.s])
        if snapshot_name is None:
            snapshot_name = (self.file_address + '_t={}.eps'.format(time_step))
        else:
            (file_dir,) = h_file.split_address(snapshot_name)
            assert os.path.isdir(file_dir)
            assert os.access(file_dir, os.W_OK)

        animation = b_ani.Animation(self)
        animation.snapshot(time_step,
                           output_arr,
                           specimen_arr,
                           snapshot_name)
        return

    #####################################
    #           Serialization           #
    #####################################
    # Todo Create __is_equal__ method, compare to default  params -> dont save
    def save(self, file_address=None):
        """Write all parameters of the :class:`Simulation` instance
        to its corresponding file or to the given HDF5 file.

        Parameters
        ----------
        file_address : :obj:`str`, optional
            Either a full path, a base file name or a file root.
            If a base file name or a file root is given,
            then the file is placed in the
            :attr:`~boltzpy.constants.DEFAULT_DIRECTORY`.
        """
        # Prepare file_address
        if file_address is None:
            file_address = self.file_address
        # separate file directory and file root, using a helper function
        h_separate = h_file.split_address
        [file_directory, file_root] = h_separate(file_address)
        file_address = file_directory + file_root
        # Don't overwrite non-Simulation files
        self.check_parameters(file_address=self.file_address)

        # Sanity Check before saving
        self.check_integrity(False)

        # Todo if sv and collision parameters are the same -> keep Collisions
        # Create new HDF5 file (delete old file, if any)
        file = h5py.File(file_address + ".hdf5", mode='w')
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

        # Save Velocity Grids
        key = "Velocity_Grids"
        file.create_group(key)
        self.sv.save(file[key])

        # Save initialization rules
        file.create_group("Initialization")
        hdf5_group = file["Initialization"]
        hdf5_group.attrs["Number of Rules"] = self.n_rules
        for (rule_idx, rule) in enumerate(self.rule_arr):
            hdf5_group.create_group("Rule_" + str(rule_idx))
            rule.save(hdf5_group["Rule_" + str(rule_idx)])

        # Save initialization Array
        key = "Initialization/Initialization Array"
        file[key] = self.init_arr

        # Save Computation Parameters
        if "Computation" not in file.keys():
            file.create_group("Computation")
        if self.coll_select_scheme is not None:
            key = "Collision_Selection_Scheme"
            file["Computation"].attrs[key] = self.coll_select_scheme
        # Todo replace this by Knudsen Number + adaptive RungeKutta
        if self.coll_substeps is not None:
            key = "Collision_Substeps"
            file["Computation"].attrs[key] = self.coll_substeps
        if self.order_os is not None:
            key = "Convergence_Order_Operator_Splitting"
            file["Computation"].attrs[key] = self.order_os
        if self.order_transp is not None:
            key = "Convergence_Order_Transport"
            file["Computation"].attrs[key] = self.order_transp
        if self.order_coll is not None:
            key = "Convergence_Order_Collision_Operator"
            file["Computation"].attrs[key] = self.order_coll
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
                              initialization_rules=self.rule_arr,
                              initialization_array=self.init_arr,
                              output_parameters=self.output_parameters,
                              coll_select_scheme=self.coll_select_scheme,
                              coll_substeps=self.coll_substeps,
                              order_os=self.order_os,
                              order_transp=self.order_transp,
                              order_coll=self.order_coll,
                              complete_check=complete_check)
        return

    @staticmethod
    def check_parameters(file_address=None,
                         species=None,
                         time_grid=None,
                         position_grid=None,
                         species_velocity_grid=None,
                         initialization_rules=None,
                         initialization_array=None,
                         output_parameters=None,
                         coll_select_scheme=None,
                         coll_substeps=None,
                         order_os=None,
                         order_transp=None,
                         order_coll=None,
                         complete_check=False):
        r"""Sanity Check.

        Check integrity of given parameters and their interaction.

        Parameters
        ----------
        file_address : :obj:`str`, optional
        species : :obj:`Species`, optional
        time_grid : :obj:`Grid`, optional
        position_grid : :obj:`Grid`, optional
        species_velocity_grid : :obj:`SVGrid`, optional

        initialization_rules : :obj:`~numpy.array` [:class:`Rule`], optional
        initialization_array : :obj:`~numpy.array` [:obj:`int`], optional
        output_parameters : :obj:`~numpy.array` [:obj:`str`], optional
        coll_select_scheme : :obj:`str`, optional
        coll_substeps : :obj:`int`, optional
        order_os : :obj:`int`, optional
        order_transp : :obj:`int`, optional
        order_coll : :obj:`int`, optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be assigned (not None).
            If False, then unassigned parameters are ignored.
        """
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
            for char in b_const.INVALID_CHARACTERS:
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
                hdf5_file = h5py.File(file_directory + file_root + '.hdf5')
                assert hdf5_file.attrs["class"] == "Simulation"

        if species is not None:
            assert isinstance(species, b_spc.Species)
            species.check_integrity()

        if time_grid is not None:
            assert isinstance(time_grid, b_grd.Grid)
            time_grid.check_integrity(complete_check)
            assert time_grid.dim == 1

        if position_grid is not None:
            assert isinstance(position_grid, b_grd.Grid)
            position_grid.check_integrity(complete_check)
            # Todo Remove this, when implementing 2D Transport
            if position_grid.dim is not None \
                    and position_grid.dim is not 1:
                msg = "Currently only 1D Simulations are supported!"
                raise NotImplementedError(msg)

        if species_velocity_grid is not None:
            assert isinstance(species_velocity_grid, b_svg.SVGrid)
            species_velocity_grid.check_integrity(complete_check)

        if position_grid is not None \
                and species_velocity_grid is not None:
            if position_grid.dim is not None \
                    and species_velocity_grid.dim is not None:
                assert species_velocity_grid.dim >= position_grid.dim

        if species is not None \
                and species_velocity_grid is not None:
            if species_velocity_grid.n_grids is not None:
                assert species_velocity_grid.n_grids == species.size

        if initialization_rules is not None:
            assert isinstance(initialization_rules, np.ndarray)
            assert initialization_rules.ndim == 1
            assert initialization_rules.dtype == 'object'
            for rule in initialization_rules:
                assert isinstance(rule, b_rul.Rule)
                rule.check_integrity(complete_check)
                if rule.rho is not None and species is not None:
                    assert rule.rho.shape == (species.size,)
                if (rule.drift is not None
                        and species is not None
                        and species_velocity_grid is not None):
                    assert rule.drift.shape == (species.size,
                                                species_velocity_grid.dim)
                if rule.temp is not None and species is not None:
                    assert rule.temp.shape == (species.size,)

        if initialization_array is not None:
            assert isinstance(initialization_array, np.ndarray)
            assert initialization_array.dtype == int
            assert np.min(initialization_array) >= -1
            # Todo Check this in unit tests,
            # Todo      compare before and after save + load
            if position_grid.size is not None:
                assert initialization_array.size == position_grid.size
            else:
                assert initialization_array.size == 1
            if initialization_rules is not None:
                assert (np.max(initialization_array)
                        < initialization_rules.size), \
                    'Undefined Rule! A P-Grid point is set ' \
                    'to be initialized by an undefined initialization rule.'
            if complete_check is True:
                assert np.min(initialization_array) >= 0, \
                    'Positional Grid is not properly initialized.' \
                    'Some Grid points have no initialization rule!'

        if output_parameters is not None:
            assert isinstance(output_parameters, np.ndarray)
            assert len(output_parameters.shape) is 2
            assert all([mom in b_const.SUPP_OUTPUT
                        for mom in output_parameters.flatten()])

        if coll_select_scheme is not None:
            assert isinstance(coll_select_scheme, str)
            selection_schemes = b_const.SUPP_COLL_SELECTION_SCHEMES
            assert coll_select_scheme in selection_schemes

        if coll_substeps is not None:
            assert isinstance(coll_substeps, int)
            assert coll_substeps >= 0

        if order_os is not None:
            assert isinstance(order_os, int)
            assert order_os in b_const.SUPP_ORDERS_OS
            if order_os != 1:
                raise NotImplementedError

        if order_coll is not None:
            assert isinstance(order_coll, int)
            assert order_coll in b_const.SUPP_ORDERS_COLL
            if order_coll != 1:
                raise NotImplementedError

        if order_transp is not None:
            assert isinstance(order_transp, int)
            assert order_transp in b_const.SUPP_ORDERS_TRANSP
            if order_transp != 1:
                raise NotImplementedError
        return

    def __str__(self,
                write_physical_grids=False):
        """Convert the instance to a string which describes all attributes."""
        description = ''
        description += 'Simulation File = ' + self.file_address + '.hdf5\n'
        description += 'Time = ' + str(datetime.now()) + '\n'
        description += '\n'
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
        description += 'Initialization Data\n'
        description += '-------------------\n'
        for (rule_idx, rule) in enumerate(self.rule_arr):
            rule_str = rule.__str__(rule_idx).replace('\n', '\n\t')
            description += '\t' + rule_str
            description += '\n'
        if write_physical_grids:
            description += '\tFlag-Grid of P-Space:\n\t\t'
            init_arr_str = self.init_arr.__str__().replace('\n', '\n\t\t')
            description += init_arr_str + '\n'
        description += '\n'
        description += 'Computation Data\n'
        description += '----------------\n\t'
        description += 'Collision Selection Scheme = ' \
                       '{}'.format(self.coll_select_scheme)
        description += '\n\t'
        description += 'Collision Steps per Time Step = ' \
                       '{}'.format(self.coll_substeps)
        description += '\n\t'
        description += 'Approximation Order of Operator Splitting = ' \
                       '{}'.format(self.order_os)
        description += '\n\t'
        description += 'Approximation Order of Collision Operator = ' \
                       '{}'.format(self.order_coll)
        description += '\n\t'
        description += 'Approximation Order of Transport Solver = ' \
                       '{}'.format(self.order_transp)
        description += '\n\t'
        description += 'Animated Moments:\n\t\t'
        output_str = self.output_parameters.__str__().replace('\n', '\n\t\t')
        description += output_str + '\n'
        return description
