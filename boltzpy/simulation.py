import os
import h5py
import numpy as np

import boltzpy.helpers.TimeTracker as h_tt
import boltzpy.AnimatedFigure as bp_af
import boltzpy.compute as bp_cp
import boltzpy.output as bp_o
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

    Attributes
    ----------
    timing : :class:`Grid`
        The Time Grid.
    geometry: :class:`Geometry`
        Describes the behaviour for all position points.
        Contains the :class:`initialization rules <Rule>`
    model : :class:`Model`
        Velocity-Space Grids of all Specimen.
    file : :obj:`h5py.Group <h5py:Group>`
    log_state : :obj:`numpy.bool`
    order_operator_splitting : :obj:`int`
    order_transport : :obj:`int`
    order_collisions : :obj:`int`
    """

    def __init__(self,
                 timing,
                 geometry,
                 model,
                 coll,
                 file=None,
                 log_state=False,
                 order_operator_splitting=1,
                 order_transport=1,
                 order_collisions=1):
        self.timing = timing
        self.geometry = geometry
        self.model = model
        self.coll = coll
        self.log_state = np.bool(log_state)
        if file is None:
            idx = 0
            while True:
                idx += 1
                file_path = __file__[:-21] + 'Simulations/' + str(idx) + ".hdf5"
                if not os.path.exists(file_path):
                    break
            file = h5py.File(file_path, mode='w')
        self.order_operator_splitting = int(order_operator_splitting)
        self.order_transport = int(order_transport)
        self.order_collisions = int(order_collisions)
        self.file = file

        self.check_integrity(complete_check=False)
        return

    @property
    def results_shape(self):
        shapes = np.empty(self.model.specimen, dtype=dict)
        for s in self.model.species:
            shapes[s] = {
                'particle_number': (
                    self.timing.size,
                    self.geometry.size),
                'mean_velocity': (
                    self.timing.size,
                    self.geometry.size,
                    self.model.ndim),
                'momentum': (
                    self.timing.size,
                    self.geometry.size,
                    self.model.ndim),
                'momentum_flow': (
                    self.timing.size,
                    self.geometry.size,
                    self.model.ndim),
                'temperature': (
                    self.timing.size,
                    self.geometry.size),
                'energy': (
                    self.timing.size,
                    self.geometry.size),
                'energy_flow': (
                    self.timing.size,
                    self.geometry.size,
                    self.model.ndim),
                "state": (
                    self.timing.size,
                    self.geometry.size,
                    self.model.vGrids[s].size)}
            if not self.log_state:
                del shapes[s]["state"]
        return shapes

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
        results = hdf_group["results"]
        file = hdf_group.file

        # Todo remove Data, move into Simulation
        # Generate Computation data
        data = bp.Data(hdf_group)
        data.check_stability_conditions()

        print('Start Computation:')
        results.attrs["t"] = 1
        time_tracker = h_tt.TimeTracker()
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
        for s in self.model.species:
            (beg, end) = self.model.index_range[s]
            spc_state = data.state[..., beg:end]
            dv = self.model.vGrids[s].physical_spacing
            mass = self.model.masses[s]
            velocities = self.model.vGrids[s].pG
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
        hdf_group = self.file["results"]
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
        xdata = (self.geometry.iG * self.geometry.delta).flatten()[1:-1]
        for (m, moment) in enumerate(moments):
            ax = figure.add_subplot(shape + (1 + m,),
                                    title=moment)
            for s in self.model.species:
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

        timing = bp.Grid.load(file["timing"])
        geometry = bp.Geometry.load(file["geometry"])
        model = bp.Model.load(file["model"])
        coll = bp.Collisions.load(file["Collisions"])
        log_state = np.bool(file.attrs["log_state"][()])
        order_operator_splitting = file.attrs["order_operator_splitting"][()]
        order_transport = file.attrs["order_transport"][()]
        order_collisions = file.attrs["order_collisions"][()]

        self = Simulation(timing,
                          geometry,
                          model,
                          coll,
                          file,
                          log_state,
                          order_operator_splitting,
                          order_transport,
                          order_collisions)
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
        hdf_group.attrs["order_operator_splitting"] = self.order_operator_splitting
        hdf_group.attrs["order_transport"] = self.order_transport
        hdf_group.attrs["order_collisions"] = self.order_collisions

        self.timing.save(hdf_group.create_group("timing"))
        self.geometry.save(hdf_group.create_group("geometry"))
        self.model.save(hdf_group.create_group("model"))
        self.coll.save(hdf_group.create_group("Collisions"))

        key = "results"
        hdf_group.create_group(key)
        # store index of current time step
        hdf_group[key].attrs["t"] = 0
        # set up separate subgroup for each species
        shapes = self.results_shape
        for s in self.model.species:
            hdf_group[key].create_group(str(s))
            grp_spc = hdf_group[key][str(s)]
            # set up separate dataset for each moment
            for (name, shape) in shapes[s].items():
                grp_spc.create_dataset(name, shape, dtype=float)

        # assert that the instance can be reconstructed from the save
        other = self.load(hdf_group)
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
        self.check_parameters(time_grid=self.timing,
                              position_grid=self.geometry,
                              species_velocity_grid=self.model,
                              geometry=self.geometry,
                              complete_check=complete_check,
                              context=self)
        return

    @staticmethod
    def check_parameters(time_grid=None,
                         position_grid=None,
                         species_velocity_grid=None,
                         geometry=None,
                         complete_check=False,
                         context=None):
        r"""Sanity Check.

        Check integrity of given parameters and their interaction.

        Parameters
        ----------
       time_grid : :obj:`Grid`, optional
        position_grid : :obj:`Grid`, optional
        species_velocity_grid : :obj:`Model`, optional
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
            assert isinstance(species_velocity_grid, bp.Model)
            species_velocity_grid.check_integrity()

        if geometry is not None:
            assert isinstance(geometry, bp.Geometry)
            geometry.check_integrity()

    def __eq__(self, other, ignore=None, print_message=True):
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
        description += 'Timing\n'
        description += '---------\n'
        time_str = self.timing.__str__(write_physical_grids)
        description += '\t' + time_str.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Model\n'
        description += '-------------------\n'
        velocity_str = self.model.__str__(write_physical_grids)
        description += '\t' + velocity_str.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        description += 'Geometry\n'
        description += '--------\n'
        geometry_str = self.geometry.__str__()
        description += '\t' + geometry_str.replace('\n', '\n\t')
        description += '\n'
        description += '\n'
        return description
