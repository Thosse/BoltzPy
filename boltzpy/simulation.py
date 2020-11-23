import os
import h5py
import numpy as np

import boltzpy.helpers.TimeTracker as h_tt
import boltzpy.AnimatedFigure as bp_af
import boltzpy.compute as bp_cp
import boltzpy as bp


class Simulation(bp.BaseClass):
    r"""Handles all aspects of a single simulation.

    Each instance correlates to a single file
    in which all parameters and computation results are  stored.
    An instance can be completely restored from its file.



    .. todo::
        - Add Knudsen Number Attribute or Property?
            * Add method to get candidate for characteristic length

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
                 file=None,
                 log_state=False,
                 order_operator_splitting=1,
                 order_transport=1,
                 order_collisions=1):
        self.timing = timing
        self.geometry = geometry
        self.model = model
        self.log_state = np.bool(log_state)
        self.order_operator_splitting = int(order_operator_splitting)
        self.order_transport = int(order_transport)
        self.order_collisions = int(order_collisions)
        if file is None:
            idx = 0
            while True:
                idx += 1
                file_path = __file__[:-21] + 'Simulations/' + str(idx) + ".hdf5"
                if not os.path.exists(file_path):
                    break
            file = h5py.File(file_path, mode='w')
        self.file = file

        self.check_integrity()
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
                self.geometry.compute(data)
            self.write_results(data, tw_idx, results)
            file.flush()
            # print time estimate
            time_tracker.print(tw, data.tG[-1, 0])
        return

    # Todo this needs an overhaul
    def write_results(self, data, tw_idx, hdf_group):
        for s in self.model.species:
            idx_range = self.model.idx_range(s)
            spc_state = data.state[..., idx_range]
            spc_group = hdf_group[str(s)]
            # number_density
            number_density = self.model.number_density(spc_state, s)
            spc_group["particle_number"][tw_idx] = number_density

            # momentum
            momentum = self.model.momentum(spc_state, s)
            spc_group["momentum"][tw_idx] = momentum

            # mean velocity
            mean_velocity = self.model.mean_velocity(spc_state, s,
                                                     momentum=momentum)
            spc_group["mean_velocity"][tw_idx] = mean_velocity

            # temperature
            temperature = self.model.temperature(spc_state, s,
                                                 number_density=number_density)
            spc_group["temperature"][tw_idx] = temperature

            # momentum flow
            spc_group["momentum_flow"][tw_idx] = self.model.momentum_flow(spc_state, s)
            # energy
            spc_group["energy"][tw_idx] = self.model.energy_density(spc_state, s)
            # energy flow
            spc_group["energy_flow"][tw_idx] = self.model.energy_flow(spc_state, s)
            # complete distribution
            if self.log_state:
                spc_group["state"][tw_idx] = spc_state
        # update index of current time step
        hdf_group.attrs["t"] = tw_idx + 1
        return

    #####################################
    #             Animation             #
    #####################################
    # Todo give moments as tuple(arrays, ) (self.file...)
    def animate(self, tmin=None, tmax=None, shape=(3, 2), moments=None):
        hdf_group = self.file["results"]

        # chosse time frame
        if tmax is None:
            tmax = int(hdf_group.attrs["t"])
        if tmin is None:
            tmin = 0
        assert tmax >= tmin >= 0
        time_frame = np.s_[tmin: tmax]
        n_frames = tmax - tmin
        figure = bp_af.AnimatedFigure(tmax=n_frames,
                                      backend="agg")
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
                    ydata = spc_group[moment][time_frame, 1:-1]
                elif spc_group[moment].ndim == 3:
                    ydata = spc_group[moment][time_frame, 1:-1, 0]
                else:
                    raise Exception
                ax.plot(xdata, ydata)
        if n_frames == 1:
            figure.save(self.file.file.filename[:-5] + '.jpeg')
        else:
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
        log_state = np.bool(file.attrs["log_state"][()])
        order_operator_splitting = file.attrs["order_operator_splitting"][()]
        order_transport = file.attrs["order_transport"][()]
        order_collisions = file.attrs["order_collisions"][()]

        self = Simulation(timing,
                          geometry,
                          model,
                          file,
                          log_state,
                          order_operator_splitting,
                          order_transport,
                          order_collisions)
        return self

    def save(self, hdf_group=None):
        """Write all parameters of the :class:`Simulation` instance
        to a HDF5 file.

        Parameters
        ----------
        hdf_group : :obj:`h5py.Group <h5py:Group>`
        """
        self.check_integrity()
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
    def check_integrity(self):
        """Sanity Check."""
        assert isinstance(self.timing, bp.Grid)
        self.timing.check_integrity()
        assert self.timing.ndim == 1
        assert isinstance(self.geometry, bp.Geometry)
        self.geometry.check_integrity()
        assert isinstance(self.model, bp.Model)
        self.model.check_integrity()
        assert self.model.ndim >= self.geometry.ndim
        assert self.geometry.model_size == self.model.size
        for r in self.geometry.rules:
            assert r.specimen >= self.model.specimen
        assert isinstance(self.file, h5py.Group)
        assert isinstance(self.log_state, np.bool)
        assert isinstance(self.order_operator_splitting, int)
        assert self.order_operator_splitting == 1
        assert isinstance(self.order_collisions, int)
        assert self.order_collisions == 1
        assert isinstance(self.order_transport, int)
        assert self.order_transport == 1
        return

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
