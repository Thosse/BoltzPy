import h5py
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
        params = {**locals(), **kwargs}
        del params["kwargs"]
        bp.BaseModel.__init__(**params)
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
        params = {**locals(), **kwargs}
        del params["kwargs"]
        BaseRule.__init__(**params)
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
        params = {**locals(), **kwargs}
        del params["kwargs"]
        InhomogeneousRule.__init__(**params)
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
        params = {**locals(), **kwargs}
        del params["kwargs"]
        BaseRule.__init__(**params)
        bp.CollisionModel.__init__(**params)
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

    def gain_array(self,
                   relation_set=None,
                   initial_state=None):
        if relation_set is None:    # use all collision
            relation_set = slice(None)
        if initial_state is None:
            initial_state = self.initial_state
        assert initial_state.shape == (self.nvels,)
        relations = self.collision_relations[relation_set]
        colvels = initial_state[relations]
        gain_term = np.prod(colvels[:, [0, 2]], axis=-1)
        loss_term = np.prod(colvels[:, [1, 3]], axis=-1)
        # We must use the absolute matrix!
        # negative signs switch gain and loss terms
        collision_matrix = np.abs(self.collision_matrix[:, relation_set])
        gains = collision_matrix.dot(gain_term + loss_term)
        return gains

    def compute(self, dt, maxiter, hdf5_group=None,
                dataset_name="result", atol=1e-12, rtol=1e-12):
        self.check_integrity()
        # set up default hdf5 group, if none is given
        if hdf5_group is None:
            hdf5_group = h5py.File(bp.SIMULATION_DIR + "/tmp_data.hdf5", mode="w")
        assert dataset_name not in hdf5_group.keys()
        # create data set of variable length, increases in 10000 steps
        result = hdf5_group.create_dataset(
            dataset_name,
            (10000, self.nvels),
            maxshape=(maxiter, self.nvels))
        # store state at each timestep here
        new_state = self.initial_state
        result[0] = new_state

        # store the interim results of the runge-kutta scheme here
        # usually named k1,k2,k3,k4, only one is needed at the time
        rks_component = np.zeros(new_state.shape, float)
        # time offsets of the rks
        rks_offsets = np.array([0, 0.5, 0.5, 1])
        # weights of each interim result / rks_component
        rks_weights = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
        # execute simulation
        for i in range(1, maxiter):
            if i >= result.shape[0]:
                result.resize((result.shape[0] + 10000, self.nvels))
            old_state = new_state
            new_state = np.copy(old_state)
            # Runge Kutta steps
            for (rks_offset, rks_weight) in zip(rks_offsets, rks_weights):
                interim_state = old_state + rks_offset * rks_component
                coll = self.collision_operator(interim_state)
                # execute runge kutta substep
                new_state = new_state + rks_weight * dt * (coll + self.source_term)
            result[i] = new_state
            # break loop when reaching equilibrium
            if np.allclose(result[i] - result[i-1], 0, atol=atol, rtol=rtol):
                result.resize((i, self.nvels))
                result.flush()
                return result
            # check for instabilities or divergence
            assert np.all(result[i] >= 0), (
                "Instability detected at t={}. "
                "Reduce time step size or increase collision_factors, "
                "to compensate the source term!".format(i))
        raise ValueError("No equilibrium established")

    #################################
    #    Compute Folw Parameters    #
    #################################
    def cmp_viscosity(self,
                      dt,
                      maxiter=100000,
                      directions=None,
                      normalize=True,
                      hdf5_group=None):
        # Viscosities are always computed for centered maxwellians
        mean_vel = np.zeros(self.ndim)
        mom_func = self.mf_stress(mean_vel, directions, orthogonalize=True)
        # set up source term
        self.source_term = mom_func * self.initial_state
        # check, that source term is orthogonal on all moments
        self.check_integrity()
        # compute viscosity
        assert dt > 0 and maxiter > 0
        inverse_source_term = self.compute(dt, maxiter=maxiter, hdf5_group=hdf5_group)
        viscosity = np.sum(inverse_source_term[-1] * mom_func)

        if normalize:
            viscosity = viscosity / np.sum(mom_func**2 * self.initial_state)
        return viscosity

    def check_integrity(self):
        BaseRule.check_integrity(self)
        bp.CollisionModel.check_integrity(self)
        assert self.source_term.shape == self.initial_state.shape
        # source term must be orthogonal to all moments
        # otherwise no equilibrium can be established
        for s in self.species:
            assert np.isclose(self.cmp_number_density(self.source_term, s), 0)
        assert np.allclose(self.cmp_momentum(self.source_term), 0)
        assert np.isclose(self.cmp_energy_density(self.source_term), 0)
