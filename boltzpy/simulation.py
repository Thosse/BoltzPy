import os
import h5py
import numpy as np

import boltzpy.helpers.TimeTracker as h_tt
import boltzpy as bp


class Simulation(bp.BaseClass):
    r"""Handles all aspects of a single simulation.

    Parameters
    ----------
    timing : :class:`Grid`
        The Time Grid.
    geometry : :class:`Geometry`
        Describes the behaviour for all position points.
        Contains the :class:`initialization rules <Rule>`
    model : :class:`CollisionModel`
        Velocity-Space Grids of all Specimen.
    results : :obj:`h5py.Group`, optional
    log_state : :obj:`numpy.bool`

    Attributes
    ----------
    dt : :obj:`float`
        Temporal step size.
    dp : :obj:`float`
        Positional step size.
        t : :obj:`numpy.int`
        Current time step.
    state : :obj:`~numpy.array` [:obj:`float`]
        The current state of the simulation.
    state : :obj:`~numpy.array` [:obj:`float`]
        Stores interim values during computation (transport step)
    """
    def __init__(self,
                 timing,
                 geometry,
                 model,
                 results=None,
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

        self.t = np.int(0)
        self.state = self.geometry.initial_state
        self.interim = np.copy(self.state)
        if results is None:
            self.results = dict()
        else:
            self.results = results

        self.log_state = np.bool(log_state)

        self.check_integrity()
        return

    @staticmethod
    def parameters():
        params = {"timing",
                  "geometry",
                  "model",
                  "log_state"}
        return params

    @staticmethod
    def attributes():
        attrs = Simulation.parameters()
        return attrs

    @property
    def file(self):
        if isinstance(self.results, h5py.Group):
            return self.results.file
        else:
            raise AttributeError

    @property
    def shape_of_results(self):
        """Returns a dictionary of the shape of each result."""
        result = dict()
        for mom in {'number_density',
                    'temperature',
                    'energy_density',
                    'pressure'}:
            result[mom] = (self.timing.size,
                           self.geometry.size,
                           self.model.nspc)
        for mom in {'mean_velocity',
                    'momentum',
                    'momentum_flow',
                    'energy_flow'}:
            result[mom] = (self.timing.size,
                           self.geometry.size,
                           self.model.nspc,
                           self.model.ndim)
        if self.log_state:
            result["state"] = (self.timing.size,
                               self.geometry.size,
                               self.model.nvels)
        return result

    @staticmethod
    def default_file(directory=None, mode="w"):
        if directory is None:
            directory = __file__[:-21] + 'Simulations/'
        idx = 0
        while True:
            idx += 1
            file_path = directory + str(idx) + ".hdf5"
            if not os.path.exists(file_path):
                break
        return h5py.File(file_path, mode=mode)

    #####################################
    #            Computation            #
    #####################################
    def compute(self,
                hdf_group=None):
        """Compute the fully configured Simulation"""
        self.check_integrity()
        # Save current state to the file
        self.save(hdf_group)
        file = self.results.file

        print('Start Computation:')
        self.results.attrs["t"] = 1
        time_tracker = h_tt.TimeTracker()
        for (tw_idx, tw) in enumerate(self.timing.iG[:, 0]):
            while self.t != tw:
                self.geometry.compute(self)
            self.write_results(tw_idx)
            file.flush()
            # print time estimate
            time_tracker.print(tw, self.timing.iG[-1, 0])
        # Todo this is only a temporary hack, do this properly!
        self.t = np.int(0)
        self.state = self.geometry.initial_state
        self.interim = self.state
        return

    # Todo this needs an overhaul
    def write_results(self, tw_idx):
        for s in self.model.species:
            idx_range = self.model.idx_range(s)
            spc_state = self.state[..., idx_range]

            number_density = self.model.cmp_number_density(spc_state, s)
            self.results["number_density"][tw_idx, :, s] = number_density
            momentum = self.model.cmp_momentum(spc_state, s)
            self.results["momentum"][tw_idx, :, s] = momentum
            mean_velocity = self.model.cmp_mean_velocity(
                spc_state,
                s,
                momentum=momentum)
            self.results["mean_velocity"][tw_idx, :, s] = mean_velocity
            self.results["temperature"][tw_idx, :, s] = self.model.cmp_temperature(
                spc_state,
                s,
                number_density=number_density,
                mean_velocity=mean_velocity)
            self.results["momentum_flow"][tw_idx, :, s] = self.model.cmp_momentum_flow(spc_state, s)
            self.results["energy_density"][tw_idx, :, s] = self.model.cmp_energy_density(spc_state, s)
            self.results["energy_flow"][tw_idx, :, s] = self.model.cmp_energy_flow(spc_state, s)
            # complete distribution
        if self.log_state:
            self.results["state"][tw_idx] = self.state
        # update index of current time step
        self.results.attrs["t"] = tw_idx + 1
        return

    #####################################
    #             Animation             #
    #####################################
    # Todo give moments as tuple(arrays, ) (self.file...)
    def animate(self, tmin=None, tmax=None, shape=(3, 2), moments=None):
        # choose time frame
        if tmax is None:
            tmax = int(self.results.attrs["t"])
        if tmin is None:
            tmin = 0
        assert 0 <= tmin <= tmax <= int(self.results.attrs["t"])
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
                if self.results[moment].ndim == 3:
                    ydata = self.results[moment][time_frame, 1:-1, s]
                elif self.results[moment].ndim == 4:
                    ydata = self.results[moment][time_frame, 1:-1, s, 0]
                else:
                    raise Exception
                ax.plot(xdata, ydata)
        if n_frames == 1:
            figure.save(self.file.filename[:-5] + '.jpeg')
        else:
            figure.save(self.file.filename[:-5] + '.mp4')
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        # load all parameters
        self = bp.BaseClass.load(hdf5_group)
        self.results = hdf5_group["results"]
        return self

    def save(self, hdf5_group=None, attributes=None):
        self.check_integrity()
        hdf5_group = self.default_file() if hdf5_group is None else hdf5_group
        if attributes is None:
            attributes = self.parameters()
            attributes.add("results")

        # results must be saved separately, don't pass them to BaseClass
        save_results = "results" in attributes
        if save_results:
            attributes.remove("results")

        has_results = isinstance(self.results, h5py.Group)
        # load results temporarily, to avoid accidentally overwriting them
        if has_results:
            max_t = self.results.attrs["t"]
            self.results = {key: val[()]
                            for (key, val) in self.results.items()}
        # set default result values, for compute() to access
        elif save_results:
            max_t = 0
            for (moment, shape) in self.shape_of_results.items():
                self.results[moment] = np.zeros(shape, dtype=float)

        # save all attributes, except results
        bp.BaseClass.save(self, hdf5_group, attributes)

        # save results in group
        if save_results or has_results:
            hdf5_group.create_group("results")
            # store index of current time step
            hdf5_group["results"].attrs["t"] = max_t
            for (moment, values) in self.results.items():
                hdf5_group["results"][moment] = values
            self.results = hdf5_group["results"]
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

        # results should only be a dictionary, if freshly initialized
        if isinstance(self.results, dict):
            # todo assert len(self.results) == 0
            #  this collides with BaseClass.save calling check_integrity
            pass
        else:
            assert isinstance(self.results, h5py.Group)

        assert isinstance(self.log_state, np.bool)

        # check Courant-Friedrichs-Levy-Condition
        vels_norm = np.linalg.norm(self.model.vels)
        max_vels_norm = np.max(vels_norm)
        # Courant–Friedrichs–Lewy (CFL) condition
        assert max_vels_norm * (self.dt/self.dp) < 1/2
        return

    def __eq__(self, other, ignore=None, print_message=False):
        if ignore is None:
            ignore = ["t", "state", "interim"]
        return super().__eq__(other, ignore, print_message)

    def __str__(self,
                write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance.
        """
        description = ''
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
