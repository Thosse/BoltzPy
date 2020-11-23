
import numpy as np

import boltzpy as bp
import boltzpy.compute as bp_cp


class Rule(bp.BaseClass):
    """Base Class for all Rules

    Contains methods for initialization and plotting.

    Parameters
    ----------
    particle_number : :obj:`~numpy.array` [:obj:`float`]
    mean_velocity : :obj:`~numpy.array` [:obj:`float`]
    temperature : :obj:`~numpy.array` [:obj:`float`]
    model : :obj:`~boltzpy.Model`, optional
        Used to construct the initial state,
        if no initial_state parameter is given.
    initial_state : :obj:`~numpy.array` [:obj:`float`], optional
    """
    def __init__(self,
                 particle_number,
                 mean_velocity,
                 temperature,
                 model=None,
                 initial_state=None):
        self.particle_number = np.array(particle_number, dtype=float)
        self.mean_velocity = np.array(mean_velocity, dtype=float)
        self.temperature = np.array(temperature, dtype=float)
        self.ndim = self.mean_velocity.shape[-1]
        self.specimen = self.particle_number.size
        # assert matching shapes
        assert self.particle_number.shape == (self.specimen,)
        assert self.mean_velocity.shape == (self.specimen, self.ndim)
        assert self.temperature.shape == (self.specimen,)
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
    def parameters():
        return {"particle_number",
                "mean_velocity",
                "temperature",
                "initial_state"}

    @staticmethod
    def attributes():
        attrs = Rule.parameters()
        attrs.update({"ndim",
                      "specimen"})
        return attrs

    def compute_initial_state(self, model):
        assert isinstance(model, bp.Model)
        assert self.ndim == model.ndim
        assert self.specimen == model.specimen
        initial_state = model.compute_initial_state(self.particle_number,
                                                    self.mean_velocity,
                                                    self.temperature)
        return initial_state

    def plot(self,
             model):
        """Plot the initial state of a single specimen using matplotlib 3D."""
        assert isinstance(model, bp.Model)
        assert self.ndim == 2, (
            "3D Plots are only implemented for 2D velocity spaces")

        fig = bp.Plot.AnimatedFigure()
        for s in model.species:
            ax = fig.add_subplot((1, model.specimen, s + 1), 3)
            idx_range = model.idx_range(s)
            vels = model.velocities[idx_range]
            ax.plot(vels[..., 0],
                    vels[..., 1],
                    self.initial_state[idx_range])
        fig.show()

        # Todo add Wireframe3D plot to animated figure
        # plot continuous maxwellian as a surface plot
        # mass = model.masses[specimen]
        # maximum_velocity = model.maximum_velocity
        # plot_object = bp_p.plot_continuous_maxwellian(
        #     self.particle_number[specimen],
        #     self.mean_velocity[specimen],
        #     self.temperature[specimen],
        #     mass,
        #     -maximum_velocity,
        #     maximum_velocity,
        #     100,
        #     plot_object)
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


class InhomogeneousRule(Rule):
    """Contains computational methods for spatial inhomogeneous points

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
        super().__init__(particle_number,
                         mean_velocity,
                         temperature,
                         model,
                         initial_state)
        self.affected_points = np.array(affected_points, dtype=int)
        InhomogeneousRule.check_integrity(self)
        return

    @staticmethod
    def parameters():
        params = Rule.parameters()
        params.update({"affected_points"})
        return params

    @staticmethod
    def attributes():
        attrs = InhomogeneousRule.parameters()
        attrs.update({"ndim",
                      "specimen"})
        return attrs

    def collision(self, data):
        """Executes single collision step for the :attr:`affected_points`.

        The collision step is implemented as an euler scheme."""
        coll = data.model.collision_operator(data.state[self.affected_points])
        data.state[self.affected_points] += data.dt * coll
        assert np.all(data.state[self.affected_points] >= 0)
        return

    def transport(self, data):
        """Executes single transport step for the :attr:`affected_points`.

           This is a finite differences scheme.
           It computes the inflow and outflow and, if necessary,
           applies reflection or absorption.
           Reads data.state
           and writes the results in data.results"""
        raise NotImplementedError

    def check_integrity(self, check_affected_points=True):
        """Sanity Check."""
        super(InhomogeneousRule, self).check_integrity()
        if check_affected_points:
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
        return


class InnerPointRule(InhomogeneousRule):
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


class ConstantPointRule(InhomogeneousRule):
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
        pass

    def transport(self, data):
        pass


class BoundaryPointRule(InhomogeneousRule):
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
        super(BoundaryPointRule, self).__init__(particle_number,
                                                mean_velocity,
                                                temperature,
                                                affected_points,
                                                model,
                                                initial_state)
        if effective_particle_number is None:
            self.effective_particle_number = np.array([
                model.number_density(
                    self.initial_state[np.newaxis, model.idx_range(s)], s)
                for s in model.species])
        self.check_integrity()
        return

    @staticmethod
    def parameters():
        params = InhomogeneousRule.parameters()
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
            v_refl = np.copy(v)
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
                                                                data.model)
        return

    def reflection(self, inflow, model):
        assert isinstance(model, bp.Model)
        reflected_inflow = np.zeros(inflow.shape, dtype=float)

        reflected_inflow += (np.dot(model.species_matrix, self.reflection_rate_inverse[:model.specimen])
                             * inflow[:, self.reflected_indices_inverse])
        reflected_inflow += (np.dot(model.species_matrix, self.reflection_rate_elastic[:model.specimen])
                             * inflow[:, self.reflected_indices_elastic])

        # compute each reflection separately for every species
        # Todo This is still wrong! faster velocities are depleted faster then slow vels
        #  thus slow vels accumulate, fast vels are reduced over time thus temperature is reduced
        for s in model.species:
            idx_range = model.idx_range(s)
            refl_thermal = (model.number_density(inflow, s)
                            / self.effective_particle_number[s]
                            * self.reflection_rate_thermal[s]
                            * self.initial_state[..., idx_range])
            reflected_inflow[..., idx_range] += refl_thermal
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


class HomogeneousRule(Rule):
    """Implementation of a homogeneous Simulation.
    This means that no Transport happens in space.
    However, it is possible to provide a source term s,
    such that

    Parameters
    ----------
    source_term : :obj:'~numpy.array'[:obj:'float']
    """
    def __init__(self,
                 particle_number,
                 mean_velocity,
                 temperature,
                 model,
                 initial_state=None,
                 source_term=0.0):
        super().__init__(particle_number,
                         mean_velocity,
                         temperature,
                         model,
                         initial_state)
        self.model = model
        self.source_term = np.zeros(self.initial_state.shape, dtype=float)
        self.source_term[...] = source_term
        return

    @staticmethod
    def parameters():
        params = Rule.parameters()
        params.update({"source_term"})
        return params

    @staticmethod
    def attributes():
        attrs = HomogeneousRule.parameters()
        attrs.update({"ndim",
                      "specimen"})
        return attrs

    def compute(self, dt=None, maxiter=5000, _depth=0):
        self.check_integrity()
        if dt is None:
            max_weight = np.max(self.model.collision_matrix)
            dt = 1 / (20 * max_weight)
        state = self.initial_state
        # store state at each timestep here
        result = np.zeros((maxiter,) + state.shape)
        result[0] = state
        # store the interim results of the runge-kutta scheme here
        # usually named k1,k2,k3,k4, only one is needed at the time
        rks_component = np.zeros(state.shape, float)
        # time offsets of the rks
        rks_offset = np.array([0, 0.5, 0.5, 1])
        # weights of each interim result / rks_component
        rks_weight = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
        # execute simulation
        for i in range(1, maxiter):
            state = result[i-1]
            result[i] = state
            # Runge Kutta steps
            for (offset, weight) in zip(rks_offset, rks_weight):
                interim_state = state + offset * rks_component
                coll = self.model.collision_operator(interim_state)
                rks_component = dt * (coll - self.source_term)
                result[i] = result[i] + weight * rks_component
            # break loop when reaching equilibrium
            if np.allclose(result[i] - result[i-1], 0, atol=1e-8, rtol=1e-8):
                return result[:i+1]
            # reduce dt, if NaN Values show up, at most 2 times!
            elif np.isnan(result[i]).any():
                if _depth < 3:
                    print("NaN-Value at i={}, set dt/100 = {}"
                          "".format(i, dt/100))
                    return self.compute(dt/100, maxiter, _depth + 1)
                else:
                    raise ValueError("NaN-Value at i={}".format(i))
        raise ValueError("No equilibrium established")

    def check_integrity(self):
        assert self.source_term.shape == self.initial_state.shape
        # source term must be orthogonal to all moments
        # otherwise no equilibrium can be established
        for s in self.model.species:
            number_density = self.model.number_density(self.source_term, s)
            assert np.isclose(number_density, 0)
        assert np.allclose(self.model.momentum(self.source_term), 0)
        assert np.allclose(self.model.energy_density(self.source_term), 0)
