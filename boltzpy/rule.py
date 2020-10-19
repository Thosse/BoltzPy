
import numpy as np
import h5py

import boltzpy as bp
import boltzpy.compute as bp_cp
import boltzpy.plot as bp_p
import boltzpy.initialization as bp_i
import boltzpy.output as bp_o


class Rule(bp.BaseClass):
    """Encapsulates the initialization methods
    and computational behaviour during the Simulation
    for all points, that are affected by this rule.

    Parameters
    ----------
    particle_number : :obj:`~numpy.array` [:obj:`float`]
    mean_velocity : :obj:`~numpy.array` [:obj:`float`]
    temperature : :obj:`~numpy.array` [:obj:`float`]
    affected_points : :obj:`list`[:obj:`int`]
    model : :obj:`~boltzpy.Model`, optional
        Used to construct the initial state,
        if no initial_state parameter is given.
    initial_state : :obj:`~numpy.array` [:obj:`float`], optional
    """
    def __init__(self,
                 particle_number,
                 mean_velocity,
                 temperature,
                 affected_points,
                 model=None,
                 initial_state=None):
        self.particle_number = np.array(particle_number, dtype=float)
        self.mean_velocity = np.array(mean_velocity, dtype=float)
        self.temperature = np.array(temperature, dtype=float)
        self.affected_points = np.array(affected_points, dtype=int)
        # Either initial_state is given as parameter
        if initial_state is not None:
            assert model is None
            self.initial_state = initial_state
        # or it is constructed based on the model
        else:
            assert model is not None
            self.initial_state = self.compute_initial_state(model)
        Rule.check_integrity(self)
        return

    @staticmethod
    def classes():
        return {'InnerPointRule': InnerPointRule,
                'ConstantPointRule': ConstantPointRule,
                'BoundaryPointRule': BoundaryPointRule,
                'HomogeneousPointRule': HomogeneousPointRule}

    @staticmethod
    def parameters():
        return {"particle_number",
                "mean_velocity",
                "temperature",
                "affected_points",
                "initial_state"}

    @staticmethod
    def attributes():
        attrs = Rule.parameters()
        attrs.update({"ndim",
                      "specimen"})
        return attrs

    @property
    def ndim(self):
        return self.mean_velocity.shape[1]

    @property
    def specimen(self):
        return self.mean_velocity.shape[0]

    def compute_initial_state(self, model):
        assert isinstance(model, bp.Model)
        assert self.ndim == model.ndim
        assert self.specimen == model.specimen

        initial_state = np.zeros(model.size, dtype=float)
        for s in model.species:
            mass = model.masses[s]
            velocities = model.vGrids[s].pG
            delta_v = model.vGrids[s].physical_spacing
            (beg, end) = model.index_offset[s:s+2]
            initial_state[beg:end] = bp_i.compute_initial_distribution(
                velocities,
                delta_v,
                mass,
                self.particle_number[s],
                self.mean_velocity[s],
                self.temperature[s])

        # Todo test initial state, moments should be matching (up to 10^-x)
        return initial_state

    def collision(self, data):
        """Executes single collision step for the :attr:`affected_points`.
           Reads data.state
           and writes the results in data.results"""
        raise NotImplementedError

    def transport(self, data):
        """Executes single transport step for the :attr:`affected_points`.

           This is a finite differences scheme.
           It computes the inflow and outflow and, if necessary,
           applies reflection or absorption.
           Reads data.state
           and writes the results in data.results"""
        raise NotImplementedError

    def plot(self,
             model,
             specimen,
             plot_object=None,
             **plot_style):
        """Plot the initial state of a single specimen using matplotlib 3D."""
        assert isinstance(model, bp.Model)
        assert isinstance(specimen, int)
        assert 0 <= specimen < self.specimen
        assert self.ndim == 2, (
            "3D Plots are only implemented for 2D velocity spaces")
        # show plot directly, if no object to store in is specified
        show_plot_directly = plot_object is None

        # Construct default plot object if None was given
        if plot_object is None:
            # Choose standard pyplot
            import matplotlib as mpl
            mpl.use('TkAgg')
            import matplotlib.pyplot as plt
            plot_object = plt

        # plot continuous maxwellian as a surface plot
        mass = model.masses[specimen]
        maximum_velocity = model.maximum_velocity
        plot_object = bp_p.plot_continuous_maxwellian(
            self.particle_number[specimen],
            self.mean_velocity[specimen],
            self.temperature[specimen],
            mass,
            -maximum_velocity,
            maximum_velocity,
            100,
            plot_object)

        # plot discrete distribution as a 3D bar plot
        beg, end = model.index_range[specimen]
        plot_object = bp_p.plot_discrete_distribution(
            self.initial_state[beg:end],
            model.vGrids[specimen].pG,
            model.vGrids[specimen].physical_spacing,
            plot_object,
        )
        if show_plot_directly:
            plot_object.show()
        return plot_object

    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`Rule` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Rule`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] in Rule.classes().keys()
        # choose derived class for new rule
        rule_class = Rule.classes()[hdf5_group.attrs["class"]]
        parameters = dict()
        for param in rule_class.parameters():
            parameters[param] = hdf5_group[param][()]
        return rule_class(**parameters)

    def save(self, hdf5_group, write_all=False):
        """Write the main parameters of the :class:`Rule` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        write_all : :obj:`bool`
            If True, write all attributes and properties to the file,
            even the unnecessary ones. Useful for testing,
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity()
        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = self.__class__.__name__
        attributes = self.attributes() if write_all else self.parameters()
        for attr in attributes:
            hdf5_group[attr] = self.__getattribute__(attr)
        return

    def check_integrity(self):
        """Sanity Check."""
        assert isinstance(self.particle_number, np.ndarray)
        assert self.particle_number.dtype == float
        assert self.particle_number.ndim == 1
        assert np.min(self.particle_number) > 0

        assert isinstance(self.mean_velocity, np.ndarray)
        assert self.mean_velocity.dtype == float
        assert self.mean_velocity.ndim == 2
        assert self.mean_velocity.shape[0] == self.particle_number.size
        assert self.mean_velocity.shape[1] in {2, 3}, (
            "Any drift has the same dimension as the velocity grid "
            "and must be in [2,3]."
            "mean_velocity.shape[1] = "
            "{}". format(self.mean_velocity.shape[1]))

        assert isinstance(self.temperature, np.ndarray)
        assert self.temperature.dtype == float
        assert self.temperature.ndim == 1
        assert np.min(self.temperature) > 0
        assert self.particle_number.size == self.temperature.size

        assert isinstance(self.affected_points, np.ndarray)
        assert self.affected_points.ndim == 1
        assert self.affected_points.dtype == int
        assert self.affected_points.size > 0
        assert np.min(self.affected_points) >= 0
        # if context is not None and context.geometry.shape is not None:
        #     assert np.max(affected_points) < context.geometry.size
        assert self.affected_points.size == len(set(self.affected_points)), (
            "Some points are affected twice be the same Rule:"
            "{}".format(self.affected_points))

        assert isinstance(self.initial_state, np.ndarray)
        assert self.initial_state.dtype == float
        assert not np.any(np.isnan(self.initial_state)), (
            "Initial state contains NAN values!"
        )
        assert np.all(self.initial_state >= 0), (
            "Minimal value = {}\n"
            "Initial values:\n"
            "{}".format(np.min(self.initial_state), self.initial_state)
        )
        return

    def __str__(self):
        """Convert the instance to a string, describing all attributes."""
        description = ''
        for (key, value) in self.__dict__.items():
            description += '{key}:\n\t{value}\n'.format(
                key=key,
                value=value.__str__().replace('\n', '\n\t'))
        return description


class InnerPointRule(Rule):
    def __init__(self,
                 particle_number,
                 mean_velocity,
                 temperature,
                 affected_points,
                 model=None,
                 initial_state=None):
        super().__init__(particle_number,
                         mean_velocity,
                         temperature,
                         affected_points,
                         model,
                         initial_state)
        return

    def collision(self, data):
        bp_cp.euler_scheme(data, self.affected_points)
        return

    def transport(self, data):
        if data.p_dim != 1:
            message = 'Transport is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        # simulate outflow
        data.result[self.affected_points, :] = bp_cp.transport_outflow_remains(
            data,
            self.affected_points
        )
        # simulate inflow
        data.result[self.affected_points, :] += bp_cp.transport_inflow_innerPoint(
            data,
            self.affected_points
        )
        return


class ConstantPointRule(Rule):
    def __init__(self,
                 particle_number,
                 mean_velocity,
                 temperature,
                 affected_points,
                 model=None,
                 initial_state=None):
        super().__init__(particle_number,
                         mean_velocity,
                         temperature,
                         affected_points,
                         model,
                         initial_state)
        return

    def collision(self, data):
        bp_cp.euler_scheme(data, self.affected_points)
        # Todo replace by bp_cp.no_collisions(data, self.affected_points)
        #  before that, implement proper initialization
        #  constant points should not change under collisions
        return

    def transport(self, data):
        pass


class BoundaryPointRule(Rule):
    def __init__(self,
                 particle_number,
                 mean_velocity,
                 temperature,
                 reflection_rate_inverse,
                 reflection_rate_elastic,
                 reflection_rate_thermal,
                 absorption_rate,
                 affected_points,
                 surface_normal=None,
                 incoming_velocities=None,
                 reflected_indices_inverse=None,
                 reflected_indices_elastic=None,
                 effective_particle_number=None,
                 model=None,
                 initial_state=None):
        self.reflection_rate_inverse = np.array(reflection_rate_inverse, dtype=float)
        self.reflection_rate_elastic = np.array(reflection_rate_elastic, dtype=float)
        self.reflection_rate_thermal = np.array(reflection_rate_thermal, dtype=float)
        self.absorption_rate = np.array(absorption_rate, dtype=float)

        # Either the incoming velocities and reflections are given as parameters
        if surface_normal is None:
            assert reflected_indices_inverse is not None
            assert reflected_indices_elastic is not None
            assert incoming_velocities is not None
            self.reflected_indices_inverse = np.array(reflected_indices_inverse,  dtype=int)
            self.reflected_indices_elastic = np.array(reflected_indices_elastic, dtype=int)
            self.incoming_velocities = np.array(incoming_velocities, dtype=int)
            self.effective_particle_number = np.array(effective_particle_number, dtype=float)
        # Or they must be generated based on the surface normal and the model
        else:
            assert model is not None
            assert isinstance(surface_normal, np.ndarray)
            assert surface_normal.dtype == int
            assert np.all(np.abs(surface_normal) <= 1), (
                "A surface normal must have entries from [-1, 0, 1]."
                "surface_normal = {}".format(surface_normal)
            )
            self.incoming_velocities = self.compute_incoming_velocities(model, surface_normal)
            self.reflected_indices_inverse = self.compute_reflected_indices_inverse(model)
            self.reflected_indices_elastic = self.compute_reflected_indices_elastic(model, surface_normal)
        super().__init__(particle_number,
                         mean_velocity,
                         temperature,
                         affected_points,
                         model,
                         initial_state)
        if effective_particle_number is None:
            self.effective_particle_number = np.array([
                bp_o.particle_number(
                    self.initial_state[np.newaxis,
                                       model.index_offset[s]: model.index_offset[s+1]],
                    model.vGrids[s].physical_spacing)
                for s in model.species])
        self.check_integrity()
        return

    @staticmethod
    def parameters():
        params = Rule.parameters()
        params.update({"reflection_rate_inverse",
                       "reflection_rate_elastic",
                       "reflection_rate_thermal",
                       "absorption_rate",
                       "incoming_velocities",
                       "reflected_indices_inverse",
                       "reflected_indices_elastic",
                       "effective_particle_number"})
        return params

    @staticmethod
    def attributes():
        attrs = BoundaryPointRule.parameters()
        attrs.update({"ndim",
                      "specimen"})
        return attrs

    @staticmethod
    def compute_incoming_velocities(model, surface_normal):
        # the incoming velocities are used to calculate the inflow during transport
        # we calculate the scalar product for each entry and check if its > 0
        # thus the velocity points towards the border
        incoming_velocities = np.where(model.iMG @ surface_normal > 0)[0]
        return incoming_velocities

    @staticmethod
    def compute_reflected_indices_inverse(model):
        reflected_indices_inverse = np.zeros(model.size, dtype=int)
        for (idx_v, v) in enumerate(model.iMG):
            spc = model.get_spc(idx_v)
            v_refl = -v
            idx_v_refl = model.get_idx(spc, v_refl)
            reflected_indices_inverse[idx_v] = idx_v_refl
        return reflected_indices_inverse

    @staticmethod
    def compute_reflected_indices_elastic(model, surface_normal):
        reflected_indices_elastic = np.zeros(model.size, dtype=int)
        # Todo only works in 1D Geometries
        assert np.sum(np.abs(surface_normal)) == 1, (
            "only works in 1D Geometries, "
            "doesn't even use surface normal currently")
        for (idx_v, v) in enumerate(model.iMG):
            spc = model.get_spc(idx_v)
            v_refl= np.copy(v)
            v_refl[0] = - v[0]
            idx_v_refl = model.get_idx(spc, v_refl)
            reflected_indices_elastic[idx_v] = idx_v_refl
        return reflected_indices_elastic

    def compute_initial_state(self, model):
        full_initial_state = super().compute_initial_state(model)
        # compute outgoing velocities, by relfecting incoming velocities
        outgoing_velocities = self.reflected_indices_inverse[self.incoming_velocities]
        # Set initial state to zero for all non-outgoing velocities
        initial_state = np.zeros(full_initial_state.shape)
        initial_state[outgoing_velocities] = full_initial_state[outgoing_velocities]
        return initial_state

    def collision(self, data):
        pass

    def transport(self, data):
        if data.p_dim != 1:
            message = 'Transport is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        # Simulate Outflowing
        data.result[self.affected_points, :] = bp_cp.transport_outflow_remains(
            data,
            self.affected_points
        )
        # Simulate Inflow
        inflow = bp_cp.transport_inflow_boundaryPoint(data,
                                                      self.affected_points,
                                                      self.incoming_velocities
                                                      )
        data.result[self.affected_points, :] += self.reflection(inflow,
                                                                data)
        return

    def reflection(self, inflow, data):
        reflected_inflow = np.zeros(inflow.shape, dtype=float)
        # compute each reflection separately for every species
        for s in range(data.n_spc):
            beg, end = data.v_range[s]
            inverse_inflow = self.reflection_rate_inverse[s] * inflow
            reflected_inflow[:, self.reflected_indices_inverse] += inverse_inflow

            elastic_inflow = self.reflection_rate_elastic[s] * inflow
            reflected_inflow[:, self.reflected_indices_elastic] += elastic_inflow

            thermal_inflow = bp_o.particle_number(
                self.reflection_rate_thermal[s] * inflow[..., beg:end],
                data.dv[s])
            thermal_factor = (thermal_inflow / self.effective_particle_number[s])
            reflected_inflow[..., beg:end] += (
                thermal_factor[:, np.newaxis]
                * self.initial_state[np.newaxis, beg:end]
            )
        return reflected_inflow

    def check_integrity(self, complete_check=True, context=None):
        super().check_integrity()
        assert np.all(self.mean_velocity == 0), (
            "BoundaryPointRules must have no drift! Drift ="
            "{}".format(self.mean_velocity)
        )

        rates = [self.reflection_rate_inverse,
                 self.reflection_rate_elastic,
                 self.reflection_rate_thermal,
                 self.absorption_rate]
        for rate in rates:
            assert isinstance(rate, np.ndarray)
            assert rate.dtype == float, (
                "Any reflection/absorption rate must be of type float. "
                "type(rate) = {}".format(type(rate))
            )
            assert rate.ndim == 1, (
                "All rates must be 1 dimensional arrays."
                "A single float for each species."
            )
            assert np.all(0 <= rate) and np.all(rate <= 1), (
                "Reflection/Absorption rates must be between 0 and 1. "
                "Rates = {}".format(rates)
            )
        assert len({len(rate) for rate in rates}) == 1, (
            "All rates must have the same length (number of species)."
            "Rates = {}".format(rates)
        )
        assert np.allclose(np.sum(rates, axis=0), 1.0, atol=1e-12), (
            "Reflection/Absorption rates must sum up to 1 for each species.\n"
            "Rates = {}\n"
            "Sums = {}".format(rates, np.sum(rates, axis=0))
        )

        for indices in [self.incoming_velocities,
                        self.reflected_indices_inverse,
                        self.reflected_indices_elastic]:
            assert isinstance(indices, np.ndarray)
            assert indices.dtype == int
            assert len(set(indices)) == indices.size, (
                "Index arrays must be unique indices!"
                "idx_array:\n{}".format(indices)
            )

        for reflection_indices in [self.reflected_indices_inverse,
                                   self.reflected_indices_elastic]:
            no_reflecion = np.arange(reflection_indices.size)
            reflect_twice = reflection_indices[reflection_indices]
            assert np.all(no_reflecion == reflect_twice), (
                    "Any Reflection applied twice, must return the original."
                    "idx_array[idx_array]:\n{}".format(reflect_twice))


class HomogeneousPointRule(Rule):
    """Implementation of a homogeneous Simulation.
    This means that no Transport happens in space.
    However, it is possible to provide a source term s,
    such that

    .. math::` \partial_t f + s = J[f,f]`

    Parameters
    ----------
    source_term : :obj:'~numpy.array'[:obj:'float']
    """
    def __init__(self,
                 particle_number,
                 mean_velocity,
                 temperature,
                 affected_points,
                 source_term=0.,
                 model=None,
                 initial_state=None):
        super().__init__(particle_number,
                         mean_velocity,
                         temperature,
                         affected_points,
                         model,
                         initial_state)
        self.source_term = np.array(source_term, dtype=float)
        return

    @staticmethod
    def parameters():
        params = Rule.parameters()
        params.update({"source_term"})
        return params

    def collision(self, data):
        bp_cp.collision_rkv4(data, self.affected_points)
        return

    def transport(self, data):
        """Implements the transport as a 4th order Runge-Kutta scheme.
        This is possible since there are no complex boundary conditions.
        """
        state = data.state[self.affected_points]
        data.result[self.affected_points] = state - self.source_term * data.dt
        return

    def check_integrity(self, complete_check=True, context=None):
        super().check_integrity()
        if self.source_term.size != 1:
            assert self.source_term.shape == self.initial_state.shape
