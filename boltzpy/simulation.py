import os
import h5py
import numpy as np

import boltzpy.helpers.TimeTracker as h_tt
import boltzpy.AnimatedFigure as bp_af
import boltzpy.compute as bp_cp
import boltzpy.output as bp_o
import boltzpy.constants as bp_c
import boltzpy as bp


class Simulation(bp.BaseClass):
    r"""Handles all aspects of a single simulation.

    Each instance correlates to a single file
    in which all parameters and computation results are  stored.
    An instance can be completely restored from its file.



    .. todo::
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
    t : :class:`Grid`
        The Time Grid.
    geometry: :class:`Geometry`
        Describes the behaviour for all position points.
        Contains the :class:`initialization rules <Rule>`
    sv : :class:`SVGrid`
        Velocity-Space Grids of all Specimen.
    scheme : :class:`Scheme`
        Contains all computation scheme parameters.
    """

    def __init__(self,
                 t,
                 geometry,
                 sv,
                 coll,
                 scheme,
                 file_address=None,
                 log_state=False):
        # set file address (using a setter method)
        [self._file_directory, self._file_name] = ['', '']
        self.file_address = file_address

        self.t = t
        self.geometry = geometry
        self.sv = sv
        self.coll = coll
        self.scheme = scheme
        self.log_state = np.bool(log_state)
        self.check_integrity(complete_check=False)
        return

    # Todo remove this, replace usage by geometry
    @property
    def p(self):
        return self.geometry

    # Todo remove this, replace usage by geometry
    @property
    def init_arr(self):
        return self.geometry.init_array

    # Todo remove this, replace usage by geometry
    @property
    def rule_arr(self):
        return self.geometry.rules

    @property
    def default_directory(self):
        return __file__[:-21] + 'Simulations/'

    @property
    def file_address(self):
        """:obj:`str` :
        Full path of the :class:`Simulation` file.
        """
        return self._file_directory + self._file_name

    @file_address.setter
    def file_address(self, address):
        if address is None:
            self._file_directory = self.default_directory
            idx = 0
            self._file_name = str(idx) + ".hdf5"
            while os.path.exists(self.file_address):
                idx += 1
                self._file_name = str(idx) + ".hdf5"
        else:
            # separate file directory and file root
            begin_filename = address.rfind("/") + 1
            self._file_directory = address[0: begin_filename]
            self._file_name = address[begin_filename:]
            # if no directory given -> put it in the default directory
            if self._file_directory == '':
                self._file_directory = self.default_directory
            # remove hdf5 ending, if any
            if self._file_name[-5:] != '.hdf5':
                self._file_name = self._file_name + ".hdf5"
        self.check_parameters(file_address=self.file_address)
        return

    @property
    def file(self):
        return h5py.File(self.file_address, mode="r+")

    # Todo make this an array(obj) of tuples, remove specimen folders?
    @property
    def shape_of_results(self):
        output = np.empty(self.sv.specimen, dtype=dict)
        for s in self.sv.species:
            output[s] = {
                'particle_number': (self.t.size, self.p.size),
                'mean_velocity': (self.t.size, self.p.size, self.sv.ndim),
                'momentum': (self.t.size, self.p.size, self.sv.ndim),
                'momentum_flow': (self.t.size, self.p.size, self.sv.ndim),
                'temperature': (self.t.size, self.p.size),
                'energy': (self.t.size, self.p.size),
                'energy_flow': (self.t.size, self.p.size, self.sv.ndim)
            }
        return output

    @property
    def n_rules(self):
        """:obj:`int` :
        Total number of :class:`initialization rules <Rule>` set up so far.
        """
        return self.geometry.rules.size

    #####################################
    #            Computation            #
    #####################################
    # Todo write hash function in Computation folder
    #     file = h5py.File(self.file_address + '.hdf5')
    #     # hash = file["Computation"].attrs["Hash_Value"]
    #     # Todo define hashing method
    #     assert hash == self.__hash__()
    #     print("The saved results are up to date!"
    #           "A new computation is not necessary")
    #     return
    # else (KeyError, AssertionError):
    def compute(self,
                file_address=None):
        """Compute the fully configured Simulation"""
        self.check_integrity()
        if file_address is None:
            file_address = self.file_address
        # Save current state to a hdf file
        self.save(file_address)
        file = h5py.File(file_address, mode="r+")
        results = file["Results"]

        # Todo remove Data, move into Simulation
        # Generate Computation data
        data = bp.Data(self.file_address)
        data.check_stability_conditions()

        print('Start Computation:')
        results.attrs["t"] = 1
        time_tracker = h_tt.TimeTracker()
        # Todo this might be buggy, if data.tG changes
        # Todo e.g. in adaptive time schemes
        # Todo proposition: iterate over length?
        for (tw_idx, tw) in enumerate(data.tG[:, 0]):
            while data.t != tw:
                bp_cp.operator_splitting(data,
                                         self.geometry.transport,
                                         self.geometry.collision)
            self.write_results(data, tw_idx, results)
            file.flush()
            # print time estimate
            time_tracker.print(tw, data.tG[-1, 0])
        return

    def write_results(self, data, tw_idx, hdf_group):
        for s in self.sv.species:
            (beg, end) = self.sv.index_range[s]
            spc_state = data.state[..., beg:end]
            dv = self.sv.vGrids[s].physical_spacing
            mass = self.sv.masses[s]
            velocities = self.sv.vGrids[s].pG
            spc_group = hdf_group[str(s)]
            # particle_number
            particle_number = bp_o.particle_number(spc_state, dv)
            spc_group["particle_number"][tw_idx] = particle_number

            # mean velocity
            mean_velocity = bp_o.mean_velocity(spc_state,
                                               dv,
                                               velocities,
                                               particle_number)
            spc_group["mean_velocity"][tw_idx] = mean_velocity
            # temperature
            temperature = bp_o.temperature(spc_state,
                                           dv,
                                           velocities,
                                           mass,
                                           particle_number,
                                           mean_velocity)
            spc_group["temperature"][tw_idx] = temperature
            # momentum
            spc_group["momentum"][tw_idx] = bp_o.momentum(
                spc_state,
                dv,
                velocities,
                mass)
            # momentum flow
            spc_group["momentum_flow"][tw_idx] = bp_o.momentum_flow(
                spc_state,
                dv,
                velocities,
                mass)
            # energy
            spc_group["energy"][tw_idx] = bp_o.energy(
                spc_state,
                dv,
                velocities,
                mass)
            # energy flow
            spc_group["energy_flow"][tw_idx] = bp_o.energy_flow(
                spc_state,
                dv,
                velocities,
                mass)
            # complete distribution
            if self.log_state:
                spc_group["state"][tw_idx] = data.state[..., beg:end]
        # update index of current time step
        hdf_group.attrs["t"] = tw_idx + 1
        return

    #####################################
    #             Animation             #
    #####################################
    def animate(self, shape=(3, 2), moments=None):
        hdf_group = self.file["Results"]
        tmax = int(hdf_group.attrs["t"])
        figure = bp_af.AnimatedFigure(tmax=tmax)
        if moments is None:
            moments = ['particle_number',
                       'mean_velocity',
                       'momentum',
                       'momentum_flow',
                       'temperature',
                       'energy']
        else:
            assert len(moments) <= np.prod(shape)
        # xdata (geometry) is shared over all plots
        # Todo flatten() should NOT be necessary, fix with model/geometry
        xdata = (self.p.iG * self.p.delta).flatten()[1:-1]
        for (m, moment) in enumerate(moments):
            ax = figure.add_subplot(shape + (1 + m,),
                                    title=moment)
            for s in self.sv.species:
                spc_group = hdf_group[str(s)]
                if spc_group[moment].ndim == 2:
                    ydata = spc_group[moment][0:tmax, 1:-1]
                elif spc_group[moment].ndim == 3:
                    ydata = spc_group[moment][0:tmax, 1:-1, 0]
                else:
                    raise Exception
                ax.plot(xdata, ydata)
        figure.save(self.file_address[:-5] + '.mp4')
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(file_address):
        """Set up and return a :class:`Simulation` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        file_address : :obj:`str`, optional
            The full path to the simulation (hdf5) file.

        Returns
        -------
        self : :class:`Simulation`
        """
        assert isinstance(file_address, str)
        assert os.path.exists(file_address)
        # Open HDF5 file
        file = h5py.File(file_address, mode='r')
        assert file.attrs["class"] == "Simulation"

        t = bp.Grid.load(file["Time_Grid"])
        geometry = bp.Geometry.load(file["Geometry"])
        sv = bp.SVGrid.load(file["Velocity_Grids"])
        coll = bp.Collisions.load(file["Collisions"])
        scheme = bp.Scheme.load(file["Scheme"])
        log_state = np.bool(file.attrs["log_state"][()])

        self = Simulation(t, geometry, sv, coll, scheme, file_address, log_state)
        self.check_integrity(complete_check=False)
        return self

    def save(self, file_address=None):
        """Write all parameters of the :class:`Simulation` instance
        to a HDF5 file.

        Parameters
        ----------
        file_address : :obj:`str`, optional
            Is either a full path, a base file name or a file root.
            If it is a base file name or a file root,
            then the file is placed in the
            :attr:`~Simulation.default_directory`.
        """
        # Change Simulation.file_name, if file_address is given
        if file_address is None:
            file_address = self.file_address
        else:
            assert isinstance(file_address, str)
            if file_address != self.file_address:
                assert not os.path.exists(file_address)
        # Sanity Check before saving
        self.check_integrity(False)

        # Create new HDF5 file (deletes all old data, if any)
        file = h5py.File(file_address, mode='w')
        file.attrs["class"] = "Simulation"
        file.attrs["log_state"] = self.log_state

        key = "Time_Grid"
        file.create_group(key)
        self.t.save(file[key])
        key = "Geometry"
        file.create_group(key)
        self.geometry.save(file[key])
        key = "Velocity_Grids"
        file.create_group(key)
        self.sv.save(file[key])
        key = "Collisions"
        file.create_group(key)
        self.coll.save(file[key])
        key = "Scheme"
        file.create_group(key)
        self.scheme.save(file[key])

        key = "Results"
        file.create_group(key)
        # store index of current time step
        file[key].attrs["t"] = 0
        # set up separate subgroup for each species
        shapes = self.shape_of_results
        for s in self.sv.species:
            file[key].create_group(str(s))
            grp_spc = file[key][str(s)]
            # set up separate dataset for each moment
            for (name, shape) in shapes[s].items():
                grp_spc.create_dataset(name, shape=shape, dtype=float)
            if self.log_state:
                shape = (self.t.size, self.p.size, self.sv.vGrids[s].size)
                grp_spc.create_dataset("state", shape, dtype=float)

        # assert that the instance can be reconstructed from the save
        other = self.load(file_address)
        # if a different file name is given then, the check MUST fail
        # Todo implement proper __eq__ method
        if file_address == self.file_address:
            assert self == other
        else:
            assert not self == other
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
                              time_grid=self.t,
                              position_grid=self.p,
                              species_velocity_grid=self.sv,
                              geometry=self.geometry,
                              scheme=self.scheme,
                              complete_check=complete_check,
                              context=self)
        return

    @staticmethod
    def check_parameters(file_address=None,
                         time_grid=None,
                         position_grid=None,
                         species_velocity_grid=None,
                         geometry=None,
                         scheme=None,
                         complete_check=False,
                         context=None):
        r"""Sanity Check.

        Check integrity of given parameters and their interaction.

        Parameters
        ----------
        file_address : :obj:`str`, optional
        time_grid : :obj:`Grid`, optional
        position_grid : :obj:`Grid`, optional
        species_velocity_grid : :obj:`SVGrid`, optional
        geometry: :class:`Geometry`, optional
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

        if time_grid is not None:
            assert isinstance(time_grid, bp.Grid)
            time_grid.check_integrity()
            assert time_grid.ndim == 1

        if position_grid is not None:
            assert isinstance(position_grid, bp.Grid)
            position_grid.check_integrity()
            # Todo Remove this, when implementing 2D Transport
            if position_grid.ndim is not None \
                    and position_grid.ndim != 1:
                msg = "Currently only 1D Simulations are supported!"
                raise NotImplementedError(msg)

        if species_velocity_grid is not None:
            assert isinstance(species_velocity_grid, bp.SVGrid)
            species_velocity_grid.check_integrity()

        if geometry is not None:
            assert isinstance(geometry, bp.Geometry)
            geometry.check_integrity()

        if scheme is not None:
            scheme.check_integrity(complete_check)
        return

    def __str__(self,
                write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance.
        """
        description = ''
        description += 'Simulation File = ' + self.file_address + '\n'
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
        return description
