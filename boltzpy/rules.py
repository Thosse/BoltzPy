
import numpy as np
import boltzpy as bp


class BaseRule(bp.BaseModel):
    """Base Class for all Rules

    Contains methods for initialization and plotting.

    Parameters
    ----------
    number_densities : :obj:`~numpy.array` [:obj:`float`]
    mean_velocities : :obj:`~numpy.array` [:obj:`float`]
    temperatures : :obj:`~numpy.array` [:obj:`float`]
    initial_state : :obj:`~numpy.array` [:obj:`float`], optional
    """
    def __init__(self,
                 number_densities,
                 mean_velocities,
                 temperatures,
                 masses,
                 shapes,
                 base_delta,
                 spacings=None,
                 initial_state=None,
                 **kwargs):
        bp.BaseModel.__init__(**locals())
        self.number_densities = np.array(number_densities, dtype=float)
        self.mean_velocities = np.array(mean_velocities, dtype=float)
        self.temperatures = np.array(temperatures, dtype=float)
        # assert matching shapes
        assert self.number_densities.shape == (self.nspc,)
        assert self.mean_velocities.shape == (self.nspc, self.ndim)
        assert self.temperatures.shape == (self.nspc,)
        # Either initial_state is given as parameter
        if initial_state is not None:
            self.initial_state = np.array(initial_state, dtype=float)
        else:
            self.initial_state = BaseRule.cmp_initial_state(self)
        BaseRule.check_integrity(self)
        return

    @staticmethod
    def parameters():
        params = bp.BaseModel.parameters()
        params.update({"number_densities",
                       "mean_velocities",
                       "temperatures",
                       "initial_state"})
        return params

    @staticmethod
    def attributes():
        attrs = bp.BaseModel.attributes()
        attrs.update(BaseRule.parameters())
        return attrs

    def cmp_initial_state(self,
                          number_densities=None,
                          mean_velocities=None,
                          temperatures=None):
        if number_densities is None:
            number_densities = self.number_densities
        if mean_velocities is None:
            mean_velocities = self.mean_velocities
        if temperatures is None:
            temperatures = self.temperatures
        initial_state = bp.BaseModel.cmp_initial_state(self,
                                                       number_densities,
                                                       mean_velocities,
                                                       temperatures)
        return initial_state

    # Todo rework this, is probably obsolete
    def plot(self,
             model):
        """Plot the initial state of a single specimen using matplotlib 3D."""
        assert self.ndim == 2, (
            "3D Plots are only implemented for 2D velocity spaces")

        fig = bp.Plot.AnimatedFigure()
        for s in model.species:
            ax = fig.add_subplot((1, model.nspc, s + 1), 3)
            idx_range = model.idx_range(s)
            vels = model.vels[idx_range]
            ax.plot(vels[..., 0],
                    vels[..., 1],
                    self.initial_state[idx_range])
        fig.show()

        # Todo add Wireframe3D plot to animated figure
        # plot continuous maxwellian as a surface plot
        # mass = model.masses[specimen]
        # maximum_velocity = model.maximum_velocity
        # plot_object = bp_p.plot_continuous_maxwellian(
        #     self.number_densities[specimen],
        #     self.mean_velocities[specimen],
        #     self.temperatures[specimen],
        #     mass,
        #     -maximum_velocity,
        #     maximum_velocity,
        #     100,
        #     plot_object)
        return

    def check_integrity(self):
        """Sanity Check."""
        bp.BaseModel.check_integrity(self)
        assert isinstance(self.number_densities, np.ndarray)
        assert self.number_densities.dtype == float
        assert self.number_densities.ndim == 1
        assert self.number_densities.shape == (self.nspc,)
        assert np.min(self.number_densities) >= 0

        assert isinstance(self.mean_velocities, np.ndarray)
        assert self.mean_velocities.dtype == float
        assert self.mean_velocities.ndim == 2
        assert self.mean_velocities.shape == (self.nspc, self.ndim)

        assert isinstance(self.temperatures, np.ndarray)
        assert self.temperatures.dtype == float
        assert self.temperatures.ndim == 1
        assert self.temperatures.shape == (self.nspc,)
        assert np.min(self.temperatures) > 0

        assert isinstance(self.initial_state, np.ndarray)
        assert self.initial_state.dtype == float
        assert self.initial_state.shape == (self.nvels, )
        assert not np.any(np.isnan(self.initial_state)), (
            "Initial state contains NAN values!"
        )
        assert np.all(self.initial_state >= 0), (
            "Minimal value = {}\n"
            "Initial values:\n"
            "{}".format(np.min(self.initial_state), self.initial_state)
        )
        return


class InhomogeneousRule(BaseRule):
    """Contains computational methods for spatial inhomogeneous points

    Parameters
    ----------
    number_densities : :obj:`~numpy.array` [:obj:`float`]
    mean_velocities : :obj:`~numpy.array` [:obj:`float`]
    temperatures : :obj:`~numpy.array` [:obj:`float`]
    affected_points : :obj:`list`[:obj:`int`]
    initial_state : :obj:`~numpy.array` [:obj:`float`], optional
    """
    def __init__(self,
                 number_densities,
                 mean_velocities,
                 temperatures,
                 affected_points,
                 masses,
                 shapes,
                 base_delta,
                 spacings=None,
                 initial_state=None,
                 **kwargs):
        BaseRule.__init__(**locals())
        self.affected_points = np.array(affected_points, dtype=int)
        InhomogeneousRule.check_integrity(self)
        return

    @staticmethod
    def parameters():
        params = BaseRule.parameters()
        params.update({"affected_points"})
        return params

    @staticmethod
    def attributes():
        attrs = BaseRule.attributes()
        attrs.update(InhomogeneousRule.parameters())
        return attrs

    def collision(self, sim):
        """Executes single collision step for the :attr:`affected_points`.

        The collision step is implemented as an euler scheme."""
        coll = sim.model.collision_operator(sim.state[self.affected_points])
        sim.state[self.affected_points] += sim.dt * coll
        assert np.all(sim.state[self.affected_points] >= 0)
        return

    def transport_outflow_remains(self, sim):
        result = ((1 - sim.flow_quota) * sim.state[self.affected_points])
        return result

    def transport_inflow(self, sim):
        raise NotImplementedError

    def transport(self, sim):
        """Executes single transport step for the :attr:`affected_points`.

           This is a finite differences scheme.
           It computes the inflow and outflow and, if necessary,
           applies reflection or absorption.
           Reads sim.state
           and writes the results into sim.interim"""
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
    def transport_inflow(self, sim):
        result = np.zeros((self.affected_points.size, self.nvels), dtype=float)
        neg_vels = np.where(self.vels[:, 0] < 0)[0]
        result[:, neg_vels] = sim.state[np.ix_(self.affected_points + 1,
                                               neg_vels)]
        pos_vels = np.where(self.vels[:, 0] > 0)[0]
        result[:, pos_vels] = (sim.state[np.ix_(self.affected_points - 1,
                                                pos_vels)])
        result *= sim.flow_quota[np.newaxis, :]
        return result

    def transport(self, sim):
        # simulate outflow
        sim.interim[self.affected_points, :] = self.transport_outflow_remains(sim)
        # simulate inflow
        sim.interim[self.affected_points, :] += self.transport_inflow(sim)
        return


class ConstantPointRule(InhomogeneousRule):
    def collision(self, sim):
        pass

    def transport_outflow_remains(self, sim):
        pass

    def transport_inflow(self, sim):
        pass

    def transport(self, sim):
        pass


class BoundaryPointRule(InhomogeneousRule):
    """Contains all data and computational methods rekated to boundary points.

    Parameters
    ----------
    surface_normal : :obj:`~numpy.array` [:obj:`int`]
        Describes the orientation of the boundary.
        Points outwards.
    refl_inverse : :obj:`~numpy.array` [:obj:`float`]
        Percentage of particles being inversely reflected.
    refl_elastic : :obj:`~numpy.array` [:obj:`float`]
        Percentage of particles being elastically reflected.
    refl_thermal : :obj:`list`[:obj:`int`]
        Percentage of particles being thermally reflected.
    refl_absorbs : :obj:`~numpy.array` [:obj:`float`], optional
        Percentage of particles being absorbed.

    Attributes
    ----------
    effective_number_densities : :obj:`~numpy.array` [:obj:`float`]
        The number densities of the initial state.
        Since the part of the distribution pointing inwardly is cut off,
        the real number densities are lower than the given parameters.
        This describes the numerically computed values.
        Extensively used for normalizing the initial state, in thermal reflections.
    _outflow_quota : :obj:`~numpy.array` [:obj:`float`]
        This is a factor to compute the outflowing mass
        from the thermal part of the state.
        This is precomputed, as it is used every transport step.
    """
    def __init__(self,
                 number_densities,
                 mean_velocities,
                 temperatures,
                 affected_points,
                 surface_normal,
                 refl_inverse,
                 refl_elastic,
                 refl_thermal,
                 refl_absorbs,
                 masses,
                 shapes,
                 base_delta,
                 spacings=None,
                 initial_state=None,
                 **kwargs):
        InhomogeneousRule.__init__(**locals())
        self.surface_normal = np.array(surface_normal, dtype=int)
        self.refl_inverse = np.array(refl_inverse, dtype=float)
        self.refl_elastic = np.array(refl_elastic, dtype=float)
        self.refl_thermal = np.array(refl_thermal, dtype=float)
        self.refl_absorbs = np.array(refl_absorbs, dtype=float)
        self.vels_in = self.cmp_vels_in()
        self.refl_idx_inverse = self.cmp_refl_idx_inverse()
        self.refl_idx_elastic = self.cmp_refl_idx_elastic()
        self.initial_state = self.cmp_initial_state()
        # This is shaped (1, nspc), for simpler multiplication in transport
        self.effective_number_densities = self.cmp_number_density(
            self.initial_state[np.newaxis, :],
            separately=True)

        # precomputed value for transport_outflow()
        self._outflow_quota = (self.cmp_number_density(np.abs(self.vels[:, 0])
                                                       * self.initial_state,
                                                       separately=True)
                               / self.effective_number_densities)
        self.check_integrity()
        return

    @staticmethod
    def parameters():
        params = InhomogeneousRule.parameters()
        params.update({"refl_inverse",
                       "refl_elastic",
                       "refl_thermal",
                       "refl_absorbs",
                       "surface_normal"})
        return params

    @staticmethod
    def attributes():
        attrs = BoundaryPointRule.parameters()
        attrs.update(InhomogeneousRule.attributes())
        attrs.update({"vels_in",
                      "refl_idx_inverse",
                      "refl_idx_elastic",
                      "effective_number_densities",
                      "_outflow_quota"})
        return attrs

    def cmp_vels_in(self):
        # the incoming velocities are used to calculate the inflow during transport
        # we calculate the scalar product for each entry and check if its > 0
        # thus the velocity points towards the border
        vels_in = np.where(self.i_vels @ self.surface_normal > 0)[0]
        return vels_in

    def cmp_refl_idx_inverse(self):
        refl_idx_inverse = np.zeros(self.nvels, dtype=int)
        for (idx_v, v) in enumerate(self.i_vels):
            spc = self.get_spc(idx_v)
            v_refl = -v
            idx_v_refl = self.get_idx(spc, v_refl)
            refl_idx_inverse[idx_v] = idx_v_refl
        return refl_idx_inverse

    def cmp_refl_idx_elastic(self):
        refl_idx_elastic = np.zeros(self.nvels, dtype=int)
        for (idx_v, v) in enumerate(self.i_vels):
            spc = self.get_spc(idx_v)
            v_refl = np.copy(v)
            v_refl[0] = - v[0]
            idx_v_refl = self.get_idx(spc, v_refl)
            refl_idx_elastic[idx_v] = idx_v_refl
        return refl_idx_elastic

    def cmp_initial_state(self,
                          number_densities=None,
                          mean_velocities=None,
                          temperatures=None):
        full_initial_state = super().cmp_initial_state(number_densities,
                                                       mean_velocities,
                                                       temperatures)
        # compute outgoing velocities, by relfecting incoming velocities
        outgoing_velocities = self.refl_idx_inverse[self.vels_in]
        # Set initial state to zero for all non-outgoing velocities
        initial_state = np.zeros(full_initial_state.shape)
        initial_state[outgoing_velocities] = full_initial_state[outgoing_velocities]
        return initial_state

    def collision(self, sim):
        pass

    def transport_outflow_remains(self, sim):
        state = sim.state[self.affected_points]
        # split state into thermal and nonthermal state,
        nd_total = self.cmp_number_density(state, separately=True)
        nd_thermal = (nd_total / self.effective_number_densities
                      * self.refl_thermal[np.newaxis, :])
        state_thermal = (self.get_array(nd_thermal)
                         * self.initial_state[np.newaxis, :])

        # Add nonthermal remains,
        # nonthermal state behaves normally, just as in InnerPointRules
        remains = ((1 - sim.flow_quota) * (state - state_thermal))
        # Add thermal remains,
        # thermal state must keep the initial temperature
        # faster velocities flow out faster thus the temperature lowers
        # thus keep the relative distribution, but update the number density
        nd_thermal_remains = nd_thermal * (1 - self._outflow_quota * sim.dt / sim.dp)
        thermal_remains = (self.get_array(nd_thermal_remains)
                           * self.initial_state[np.newaxis, :])
        remains += thermal_remains
        return remains

    def transport_inflow(self, sim):
        inflow = np.zeros((self.affected_points.size, self.nvels), dtype=float)

        neg_incomings_vels = np.where(self.vels[self.vels_in, 0] < 0)[0]
        neg_vels = self.vels_in[neg_incomings_vels]
        inflow[:, neg_vels] = sim.state[np.ix_(self.affected_points + 1, neg_vels)]

        pos_incomings_vels = np.where(self.vels[self.vels_in, 0] > 0)[0]
        pos_vels = self.vels_in[pos_incomings_vels]
        inflow[:, pos_vels] = sim.state[np.ix_(self.affected_points - 1, pos_vels)]
        return inflow * sim.flow_quota

    def transport(self, sim):
        # Simulate Outflowing
        sim.interim[self.affected_points, :] = self.transport_outflow_remains(sim)
        # Simulate Inflow
        inflow = self.transport_inflow(sim)
        sim.interim[self.affected_points, :] += self.reflection(inflow)
        return

    def reflection(self, inflow):
        reflected_inflow = np.zeros(inflow.shape, dtype=float)

        reflected_inflow += (self.get_array(self.refl_inverse)
                             * inflow[:, self.refl_idx_inverse])
        reflected_inflow += (self.get_array(self.refl_elastic)
                             * inflow[:, self.refl_idx_elastic])

        # compute each reflection separately for every species
        inflow_nd = self.cmp_number_density(inflow, separately=True)
        thermal_refl_nd = self.get_array(inflow_nd
                                         * self.refl_thermal
                                         / self.effective_number_densities)
        refl_thermal = thermal_refl_nd * self.initial_state[np.newaxis, :]
        reflected_inflow += refl_thermal
        return reflected_inflow

    def check_integrity(self, complete_check=True, context=None):
        InhomogeneousRule.check_integrity(self)
        assert np.all(self.mean_velocities == 0), (
            "BoundaryPointRules must have no drift! Drift ="
            "{}".format(self.mean_velocities)
        )
        assert np.all(np.abs(self.surface_normal) <= 1), (
            "A surface normal must have entries from [-1, 0, 1]."
            "surface_normal = {}".format(self.surface_normal))

        assert np.sum(np.abs(self.surface_normal)) == 1, (
            "Reflections only work in 1D Geometries, so far")

        rates = [self.refl_inverse,
                 self.refl_elastic,
                 self.refl_thermal,
                 self.refl_absorbs]
        for rate in rates:
            assert isinstance(rate, np.ndarray)
            assert rate.dtype == float
            assert rate.ndim == 1
            assert rate.shape == (self.nspc,)
            assert np.all(0 <= rate) and np.all(rate <= 1), (
                "Reflection/Absorption rates must be between 0 and 1. "
                "Rates = {}".format(rates))
        assert len({rate.size for rate in rates}) == 1, (
            "All rates must have the same length (number of species)."
            "Rates = {}".format(rates))
        assert np.allclose(np.sum(rates, axis=0), 1.0), (
            "Reflection/Absorption rates must sum up to 1 for each species.\n"
            "Rates = {}".format(rates))

        for vel_indices in [self.vels_in,
                            self.refl_idx_inverse,
                            self.refl_idx_elastic]:
            assert isinstance(vel_indices, np.ndarray)
            assert vel_indices.dtype == int
            assert len(set(vel_indices)) == vel_indices.size, (
                "Index arrays must be unique indices!"
                "idx_array:\n{}".format(vel_indices)
            )

        for reflection_indices in [self.refl_idx_inverse,
                                   self.refl_idx_elastic]:
            no_reflecion = np.arange(reflection_indices.size)
            reflect_twice = reflection_indices[reflection_indices]
            assert np.all(no_reflecion == reflect_twice), (
                    "Any Reflection applied twice, must return the original."
                    "idx_array[idx_array]:\n{}".format(reflect_twice))


class HomogeneousRule(BaseRule, bp.CollisionModel):
    """Implementation of a homogeneous Simulation.
    This means that no Transport happens in space.
    However, it is possible to provide a source term s,
    such that

    Parameters
    ----------
    source_term : :obj:'~numpy.array'[:obj:'float']
    """
    def __init__(self,
                 number_densities,
                 mean_velocities,
                 temperatures,
                 masses,
                 shapes,
                 base_delta,
                 spacings,
                 initial_state=None,
                 source_term=0.0,
                 **kwargs):
        BaseRule.__init__(**locals())
        bp.CollisionModel.__init__(**locals())
        self.source_term = np.zeros(self.initial_state.shape, dtype=float)
        self.source_term[...] = source_term
        self.check_integrity()
        return

    @staticmethod
    def parameters():
        params = BaseRule.parameters()
        params.update(bp.CollisionModel.parameters())
        params.update({"source_term"})
        return params

    @staticmethod
    def attributes():
        attrs = HomogeneousRule.parameters()
        attrs.update(bp.CollisionModel.attributes())
        return attrs

    def compute(self, dt=None, maxiter=5000, _depth=0):
        self.check_integrity()
        if dt is None:
            max_weight = np.max(self.collision_matrix)
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
                coll = self.collision_operator(interim_state)
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
        BaseRule.check_integrity(self)
        bp.CollisionModel.check_integrity(self)
        assert self.source_term.shape == self.initial_state.shape
        # source term must be orthogonal to all moments
        # otherwise no equilibrium can be established
        for s in self.species:
            number_density = self.cmp_number_density(self.source_term, s)
            assert np.isclose(number_density, 0)
        assert np.allclose(self.cmp_momentum(self.source_term), 0)
        assert np.allclose(self.cmp_energy_density(self.source_term), 0)
