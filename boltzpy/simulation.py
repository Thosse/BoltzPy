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
                 log_state=False,
                 file=None):
        self.t = t
        self.geometry = geometry
        self.sv = sv
        self.coll = coll
        self.scheme = scheme
        self.log_state = np.bool(log_state)
        if file is None:
            idx = 0
            while True:
                idx += 1
                file_path = __file__[:-21] + 'Simulations/' + str(idx) + ".hdf5"
                if not os.path.exists(file_path):
                    break
            file = h5py.File(file_path, mode='w')
        self.file = file

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
    def compute(self,
                hdf_group=None):
        """Compute the fully configured Simulation"""
        self.check_integrity()
        if hdf_group is None:
            hdf_group = self.file
        assert isinstance(hdf_group, h5py.Group)
        # Save current state to the file
        self.save(hdf_group)
        results = hdf_group["Results"]
        file = hdf_group.file

        # Todo remove Data, move into Simulation
        # Generate Computation data
        data = bp.Data(hdf_group)
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
        figure.save(self.file.file.filename[:-5] + '.mp4')
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(file):
        """Set up and return a :class:`Simulation` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        file : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Simulation`
        """
        assert isinstance(file, h5py.Group)
        assert file.attrs["class"] == "Simulation"

        t = bp.Grid.load(file["Time_Grid"])
        geometry = bp.Geometry.load(file["Geometry"])
        sv = bp.SVGrid.load(file["Velocity_Grids"])
        coll = bp.Collisions.load(file["Collisions"])
        scheme = bp.Scheme.load(file["Scheme"])
        log_state = np.bool(file.attrs["log_state"][()])

        self = Simulation(t, geometry, sv, coll, scheme, log_state, file)
        self.check_integrity(complete_check=False)
        return self

    def save(self, hdf_group=None):
        """Write all parameters of the :class:`Simulation` instance
        to a HDF5 file.

        Parameters
        ----------
        hdf_group : :obj:`h5py.Group <h5py:Group>`
        """
        self.check_integrity(False)
        if hdf_group is None:
            hdf_group = self.file
        assert isinstance(hdf_group, h5py.Group)
        assert hdf_group.file.mode == "r+"
        # delete all current group content
        for key in hdf_group.keys():
            del hdf_group[key]

        hdf_group.attrs["class"] = "Simulation"
        hdf_group.attrs["log_state"] = self.log_state

        key = "Time_Grid"
        hdf_group.create_group(key)
        self.t.save(hdf_group[key])
        key = "Geometry"
        hdf_group.create_group(key)
        self.geometry.save(hdf_group[key])
        key = "Velocity_Grids"
        hdf_group.create_group(key)
        self.sv.save(hdf_group[key])
        key = "Collisions"
        hdf_group.create_group(key)
        self.coll.save(hdf_group[key])
        key = "Scheme"
        hdf_group.create_group(key)
        self.scheme.save(hdf_group[key])

        key = "Results"
        hdf_group.create_group(key)
        # store index of current time step
        hdf_group[key].attrs["t"] = 0
        # set up separate subgroup for each species
        shapes = self.shape_of_results
        for s in self.sv.species:
            hdf_group[key].create_group(str(s))
            grp_spc = hdf_group[key][str(s)]
            # set up separate dataset for each moment
            for (name, shape) in shapes[s].items():
                grp_spc.create_dataset(name, shape=shape, dtype=float)
            if self.log_state:
                shape = (self.t.size, self.p.size, self.sv.vGrids[s].size)
                grp_spc.create_dataset("state", shape, dtype=float)

        # assert that the instance can be reconstructed from the save
        other = self.load(hdf_group)
        # if a different file name is given then, the check MUST fail
        # Todo implement proper __eq__ method
        assert self.__eq__(other, ignore=["file"])
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
        self.check_parameters(time_grid=self.t,
                              position_grid=self.p,
                              species_velocity_grid=self.sv,
                              geometry=self.geometry,
                              scheme=self.scheme,
                              complete_check=complete_check,
                              context=self)
        return

    @staticmethod
    def check_parameters(time_grid=None,
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
        # Todo Test self.file and self.log_state
        # For complete check, assert that all parameters are assigned
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all([param is not None for param in locals().values()])

        # check all parameters, if set
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

    def __eq__(self, other, ignore=None,print_message=True):
        if ignore is None:
            ignore = ["file"]
        return super().__eq__(other, ignore, print_message)

    def __str__(self,
                write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance.
        """
        description = ''
        description += 'Simulation File = ' + self.file.file.filename + '\n'
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
