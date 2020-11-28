import os
import h5py
import numpy as np

import boltzpy.helpers.TimeTracker as h_tt
import boltzpy as bp


class Simulation(bp.BaseClass):
    r"""Handles all aspects of a single simulation.

    Each instance correlates to a single file
    in which all parameters and computation results are  stored.
    An instance can be completely restored from its file.



    .. todo::
        - Add Knudsen Number Attribute or Property?
            * Add method to get candidate for characteristic length

    Parameters
    ----------
    timing : :class:`Grid`
        The Time Grid.
    geometry : :class:`Geometry`
        Describes the behaviour for all position points.
        Contains the :class:`initialization rules <Rule>`
    model : :class:`CollisionModel`
        Velocity-Space Grids of all Specimen.
    t : :obj:`numpy.int`
        Current time step.
    state : :obj:`~numpy.array` [:obj:`float`]
        Denotes the current state of the simulation.
    file : :obj:`h5py.Group <h5py:Group>`
    log_state : :obj:`numpy.bool`

    Attributes
    ----------
    dt : :obj:`float`
        Temporal step size.
    dp : :obj:`float`
        Positional step size.
    """
    def __init__(self,
                 timing,
                 geometry,
                 model,
                 t=None,
                 state=None,
                 file=None,
                 log_state=False):
        assert isinstance(timing, bp.Grid)
        self.timing = timing

        assert isinstance(geometry, bp.Geometry)
        self.geometry = geometry

        assert isinstance(model, bp.CollisionModel)
        self.model = model
        # store large model attrributes (vels, i_vels, spc_matrix) only once
        for (r, attr) in zip(geometry.rules, bp.BaseModel.shared_attributes()):
            assert np.all(getattr(r, attr) == getattr(model, attr))
            # all point to the same objects
            setattr(r, attr, getattr(model, attr))

        self.dt = self.timing.delta
        self.dp = self.geometry.delta

        self.t = np.int(0 if t is None else t)
        if state is None:
            self.state = self.geometry.initial_state
        else:
            self.state = np.array(state, dtype=float)
        self.interim = np.copy(self.state)
        # todo add results = dict(), use method set_results(hdf_group=None)
        #  default return dict of np.arrays
        #  this is a parameter

        self.log_state = np.bool(log_state)
        # todo this needs to be removed, as it creates empty files during tests
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

    @staticmethod
    def parameters():
        params = {"timing",
                  "geometry",
                  "model",
                  #"t",
                  # "state",
                  "log_state"}
        return params

    @staticmethod
    def attributes():
        attrs = Simulation.parameters()
        return attrs

    # todo add pressure
    @property
    def results_shape(self):
        shapes = np.empty(self.model.nspc, dtype=dict)
        for s in self.model.species:
            shapes[s] = {
                'number_density': (
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
                'energy_density': (
                    self.timing.size,
                    self.geometry.size),
                'energy_flow': (
                    self.timing.size,
                    self.geometry.size,
                    self.model.ndim),
                "state": (
                    self.timing.size,
                    self.geometry.size,
                    self.model.subgrids(s).size)}
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

        print('Start Computation:')
        results.attrs["t"] = 1
        time_tracker = h_tt.TimeTracker()
        for (tw_idx, tw) in enumerate(self.timing.iG[:, 0]):
            while self.t != tw:
                self.geometry.compute(self)
            self.write_results(tw_idx, results)
            file.flush()
            # print time estimate
            time_tracker.print(tw, self.timing.iG[-1, 0])
        # Todo this is only a temporary hack, do this properly!
        self.t = np.int(0)
        self.state = self.geometry.initial_state
        self.interim = self.state
        return

    # Todo this needs an overhaul
    def write_results(self, tw_idx, hdf_group):
        for s in self.model.species:
            idx_range = self.model.idx_range(s)
            spc_state = self.state[..., idx_range]
            spc_group = hdf_group[str(s)]

            number_density = self.model.cmp_number_density(spc_state, s)
            spc_group["number_density"][tw_idx] = number_density
            momentum = self.model.cmp_momentum(spc_state, s)
            spc_group["momentum"][tw_idx] = momentum
            mean_velocity = self.model.cmp_mean_velocity(
                spc_state,
                s,
                momentum=momentum)
            spc_group["mean_velocity"][tw_idx] = mean_velocity
            spc_group["temperature"][tw_idx] = self.model.cmp_temperature(
                spc_state,
                s,
                number_density=number_density,
                mean_velocity=mean_velocity)
            spc_group["momentum_flow"][tw_idx] = self.model.cmp_momentum_flow(spc_state, s)
            spc_group["energy_density"][tw_idx] = self.model.cmp_energy_density(spc_state, s)
            spc_group["energy_flow"][tw_idx] = self.model.cmp_energy_flow(spc_state, s)
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
        figure = bp.Plot.AnimatedFigure(tmax=n_frames,
                                        backend="agg")
        if moments is None:
            moments = ['number_density',
                       'mean_velocity',
                       'momentum',
                       'momentum_flow',
                       'temperature',
                       'energy_density']
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
    def save(self, hdf_group=None, write_all=False):
        if hdf_group is None:
            hdf_group = self.file
        bp.BaseClass.save(self, hdf_group, write_all)

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
        bp.BaseClass.check_integrity(self)
        assert isinstance(self.timing, bp.Grid)
        self.timing.check_integrity()
        assert self.timing.ndim == 1

        assert isinstance(self.geometry, bp.Geometry)
        self.geometry.check_integrity()
        if self.geometry.ndim != 1:
            raise NotImplementedError

        assert isinstance(self.model, bp.CollisionModel)
        self.model.check_integrity()
        assert self.model.ndim >= self.geometry.ndim
        assert self.geometry.model_size == self.model.nvels
        # all rules must be based on the same model
        for (r, attr) in zip(self.geometry.rules, bp.BaseModel.attributes()):
            assert np.all(getattr(r, attr) == getattr(self.model, attr))
        # all rules' shared_attributes must point towards the same object
        for (r, attr) in zip(self.geometry.rules, bp.BaseModel.shared_attributes()):
            assert getattr(r, attr) is getattr(self.model, attr)
            np.shares_memory(getattr(r, attr), getattr(self.model, attr))

        assert isinstance(self.file, h5py.Group)

        assert isinstance(self.log_state, np.bool)

        # check Courant-Friedrichs-Levy-Condition
        vels_norm = np.linalg.norm(self.model.vels)
        max_vels_norm = np.max(vels_norm)
        # Courant–Friedrichs–Lewy (CFL) condition
        assert max_vels_norm * (self.dt/self.dp) < 1/2
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
