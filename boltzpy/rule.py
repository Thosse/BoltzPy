
import numpy as np
import h5py

import boltzpy as bp
import boltzpy.constants as bp_c
import boltzpy.compute as bp_cp
import boltzpy.plot as bp_p
import boltzpy.initialization as bp_i
import boltzpy.momenta as bp_m


class Rule:
    """Encapsulates the initialization methods
    and computational behaviour during the Simulation
    for all points, that are affected by this rule.

    Parameters
    ----------
    initial_rho : :obj:`~numpy.array` [:obj:`float`]
    initial_drift : :obj:`~numpy.array` [:obj:`float`]
    initial_temp : :obj:`~numpy.array` [:obj:`float`]
    affected_points : :obj:`list`[:obj:`int`]
    velocity_grids : :obj:`~boltzpy.SVGrid`, optional
        Used to construct the initial state,
        if no initial_state parameter is given.
    initial_state : :obj:`~numpy.array` [:obj:`float`], optional

    Attributes
    ----------
    initial_rho : :obj:`~numpy.array` [:obj:`float`]
        Correlates to the initial amount of particles in
        :class:`P-Grid <boltzpy.Grid>` point.
    initial_drift : :obj:`~numpy.array` [:obj:`float`]
        Describes the mean velocity,
        i.e. the first moment (expectancy value) of the
        velocity distribution.
    initial_temp : :obj:`~numpy.array` [:obj:`float`]
        Correlates to the Energy,
        i.e. the second moment (variance) of the
        velocity distribution.
    affected_points : :obj:`~numpy.array`[:obj:`int`]
        Contains all indices of the space points, where this rule applies
    initial_state : :obj:`~numpy.array` [:obj:`float`]
        Initial state of the simulation model.
    """
    def __init__(self,
                 initial_rho,
                 initial_drift,
                 initial_temp,
                 affected_points,
                 velocity_grids=None,
                 species=None,
                 initial_state=None):
        self.check_parameters(initial_rho=initial_rho,
                              initial_drift=initial_drift,
                              initial_temp=initial_temp,
                              affected_points=affected_points)
        self.initial_rho = np.array(initial_rho, dtype=float)
        self.initial_drift = np.array(initial_drift, dtype=float)
        self.initial_temp = np.array(initial_temp, dtype=float)
        self.affected_points = np.array(affected_points, dtype=int)
        # Either initial_state is given as parameter
        if initial_state is not None:
            assert velocity_grids is None
            self.initial_state = initial_state
        # or it is constructed based on the velocity_grids
        else:
            assert velocity_grids is not None
            self.initial_state = self.compute_initial_state(velocity_grids,
                                                            species)
        Rule.check_integrity(self, complete_check=False)
        return

    @property
    def subclass(self):
        """
        :obj:`str`
            Used fore load() and save() to initialize as the correct subclass.
            Must be in :const:`~boltzpy.constants.SUPP_RULE_SUBCLASSES`.
        """
        raise NotImplementedError

    @staticmethod
    def get_subclass(behaviour_type):
        if behaviour_type == 'InnerPointRule':
            return InnerPointRule
        elif behaviour_type == 'ConstantPointRule':
            return ConstantPointRule
        elif behaviour_type == 'BoundaryPointRule':
            return BoundaryPointRule
        else:
            raise NotImplementedError

    @property
    def ndim(self):
        return self.initial_drift.shape[1]

    @property
    def number_of_specimen(self):
        return self.initial_drift.shape[0]

    def compute_initial_state(self, velocity_grids, species):
        assert isinstance(velocity_grids, bp.SVGrid)
        assert self.ndim == velocity_grids.ndim
        assert self.number_of_specimen == velocity_grids.number_of_grids

        initial_state = np.zeros(velocity_grids.size, dtype=float)
        for idx_spc in range(self.number_of_specimen):
            mass = species[idx_spc].mass
            velocities = velocity_grids.vGrids[idx_spc].pG
            delta_v = velocity_grids.vGrids[idx_spc].physical_spacing
            [beg, end] = velocity_grids.index_range[idx_spc]
            initial_state[beg:end] = bp_i.compute_initial_distribution(
                velocities,
                delta_v,
                mass,
                self.initial_rho[idx_spc],
                self.initial_drift[idx_spc],
                self.initial_temp[idx_spc])

        return initial_state

    #####################################
    #            Computation            #
    #####################################
    def collision(self, data):
        raise NotImplementedError

    def transport(self, data):
        """Executes single transport step for the :attr:`affected_points`.

           This is a finite differences scheme.
           It computes the inflow and outflow and, if necessary,
           applies reflection or absorption.
           The Computation reads data.state
           and writes the results in data.results"""
        raise NotImplementedError

    #####################################
    #           Visualization           #
    #####################################
    def plot(self,
             velocity_grid,
             species,
             index_of_specimen,
             plot_object=None,
             **plot_style):
        """Plot the initial state of a single specimen using matplotlib 3D.

        Parameters
        ----------
        plot_object : TODO Figure? matplotlib.pyplot?
        """
        assert isinstance(velocity_grid, bp.SVGrid)
        assert isinstance(index_of_specimen, int)
        assert 0 <= index_of_specimen < self.number_of_specimen
        assert self.ndim == 2, (
            "3D Plots are only implemented for 2D velocity spaces")
        # show plot directly, if no object to store in is specified
        show_plot_directly = plot_object is None

        # Construct default plot object if None was given
        if plot_object is None:
            # Choose standard pyplot
            import matplotlib.pyplot as plt
            plot_object = plt

        # plot continuous maxwellian as a surface plot
        mass = species[index_of_specimen].mass
        maximum_velocity = velocity_grid.maximum_velocity
        plot_object = bp_p.plot_continuous_maxwellian(
            self.initial_rho[index_of_specimen],
            self.initial_drift[index_of_specimen],
            self.initial_temp[index_of_specimen],
            mass,
            -maximum_velocity,
            maximum_velocity,
            100,
            plot_object)

        # plot discrete distribution as a 3D bar plot
        beg, end = velocity_grid.index_range[index_of_specimen]
        plot_object = bp_p.plot_discrete_distribution(
            self.initial_state[beg:end],
            velocity_grid.vGrids[index_of_specimen].pG,
            velocity_grid.vGrids[index_of_specimen].physical_spacing,
            plot_object,
        )
        if show_plot_directly:
            plot_object.show()
        return plot_object
    #####################################
    #           Serialization           #
    #####################################
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
        assert hdf5_group.attrs["class"] == "Rule"

        # choose derived class for new rule
        behaviour_type = hdf5_group.attrs["behaviour_type"]
        rule_class = bp.Rule.get_subclass(behaviour_type)

        # read parameters from file
        params = dict()
        for (key, value) in hdf5_group.items():
            params[key] = value[()]

        # construct rule
        self = rule_class(**params)
        self.check_integrity(False)
        return self

    def save(self, hdf5_group):
        """Write the main parameters of the :class:`Rule` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity(False)

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = "Rule"
        # save derived class of rule
        hdf5_group.attrs["behaviour_type"] = self.subclass

        # write attributes to file
        for (key, value) in self.__dict__.items():
            hdf5_group[key] = value
        return

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self, complete_check=True, context=None):
        """Sanity Check.
        Assert all conditions in :meth:`check_parameters`
        and the correct type of all attributes of the instance.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all attributes must be assigned (not None).
            If False, then unassigned attributes are ignored.
        context : :class:`Simulation`, optional
            The Simulation, which this instance belongs to.
            This allows additional checks.
        """
        assert isinstance(self.initial_rho, np.ndarray)
        assert isinstance(self.initial_drift, np.ndarray)
        assert isinstance(self.initial_temp, np.ndarray)
        assert isinstance(self.affected_points, np.ndarray)
        self.check_parameters(subclass=self.subclass,
                              initial_rho=self.initial_rho,
                              initial_drift=self.initial_drift,
                              initial_temp=self.initial_temp,
                              affected_points=self.affected_points,
                              initial_state=self.initial_state,
                              complete_check=complete_check,
                              context=context)
        return

    @staticmethod
    def check_parameters(subclass=None,
                         initial_rho=None,
                         initial_drift=None,
                         initial_temp=None,
                         affected_points=None,
                         velocity_grids=None,
                         initial_state=None,
                         complete_check=False,
                         context=None):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        subclass : :obj:`str`, optional
        initial_rho : :obj:`~numpy.array` [:obj:`float`], optional
        initial_drift : :obj:`~numpy.array` [:obj:`float`], optional
        initial_temp : :obj:`~numpy.array` [:obj:`float`], optional
        affected_points : :obj:`list`[:obj:`int`]
        velocity_grids : :obj:`~boltzpy.SVGrid`, optional
        initial_state : :obj:`~numpy.array` [:obj:`float`], optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        context : :class:`Simulation`, optional
            The Simulation, which this instance belongs to.
            This allows additional checks.
        """
        # For complete check, assert that all parameters are assigned
        assert isinstance(complete_check, bool)
        if complete_check is True:
            assert all(param_val is not None
                       for (param_key, param_val) in locals().items()
                       if param_key not in ["context", "velocity_grids"])
        # Set up basic constants
        if context is not None:
            assert isinstance(context, bp.Simulation)
            n_species = context.s.size
        else:
            n_species = None

        # check all parameters, if set
        if subclass is not None:
            assert isinstance(subclass, str)
            assert subclass in bp_c.SUPP_RULE_SUBCLASSES

        if initial_rho is not None:
            if isinstance(initial_rho, list):
                initial_rho = np.array(initial_rho, dtype=float)
            assert isinstance(initial_rho, np.ndarray)
            assert initial_rho.dtype == float
            assert initial_rho.ndim == 1
            assert np.min(initial_rho) > 0
            if n_species is not None:
                assert n_species == initial_rho.size
            else:
                n_species = initial_rho.size

        if initial_drift is not None:
            if isinstance(initial_drift, list):
                initial_drift = np.array(initial_drift, dtype=float)
            assert isinstance(initial_drift, np.ndarray)
            assert initial_drift.dtype == float
            assert initial_drift.ndim == 2
            if n_species is not None:
                assert initial_drift.shape[0] == n_species
            else:
                n_species = initial_drift.shape[0]
            assert initial_drift.shape[1] in [2, 3], (
                "Any drift has the same dimension as the velocity grid "
                "and must be in [2,3]."
                "initial_drift.shape[1] = {}". format(initial_drift.shape[1])
            )
            if context is not None and context.sv.ndim is not None:
                assert initial_drift.shape[1] == context.sv.ndim

        if initial_temp is not None:
            if isinstance(initial_temp, list):
                initial_temp = np.array(initial_temp, dtype=float)
            assert isinstance(initial_temp, np.ndarray)
            assert initial_temp.dtype == float
            assert initial_temp.ndim == 1
            assert np.min(initial_temp) > 0
            if n_species is not None:
                assert n_species == initial_temp.size
            else:
                n_species = initial_temp.size

        if affected_points is not None:
            if isinstance(affected_points, list):
                affected_points = np.array(affected_points, dtype=int)
            assert isinstance(affected_points, np.ndarray)
            assert affected_points.ndim == 1
            assert affected_points.dtype == int
            assert affected_points.size > 0
            assert np.min(affected_points) >= 0
            if context is not None and context.geometry.shape is not None:
                assert np.max(affected_points) < context.geometry.size
            assert affected_points.size == len(set(affected_points)), (
                "Some points are affected twice be the same Rule:"
                "{}".format(affected_points)
            )

        if velocity_grids is not None:
            assert isinstance(velocity_grids, bp.SVGrid)
            velocity_grids.check_integrity(context=context)

        if initial_state is not None:
            assert isinstance(initial_state, np.ndarray)
            assert initial_state.dtype == float
            assert np.all(initial_state >= 0)
            # Todo test initial state, moments should be matching (up to 10^-x)
        return

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        if type(self) != type(other):
            return False
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            return False
        for (key, value) in self.__dict__.items():
            other_value = other.__dict__[key]
            if type(value) != type(other_value):
                return False
            if isinstance(value, np.ndarray):
                if np.any(value != other_value):
                    return False
            else:
                if value != other_value:
                    return False
        return True

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
                 initial_rho=None,
                 initial_drift=None,
                 initial_temp=None,
                 affected_points=None,
                 velocity_grids=None,
                 species=None,
                 initial_state=None):
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         velocity_grids,
                         species,
                         initial_state)
        return

    @property
    def subclass(self):
        return 'InnerPointRule'

    #####################################
    #            Computation            #
    #####################################
    def collision(self, data):
        bp_cp.euler_scheme(data, self.affected_points)
        return

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
        data.result[self.affected_points, :] += bp_cp.transport_inflow_innerPoint(
            data,
            self.affected_points
        )
        return


class ConstantPointRule(Rule):
    def __init__(self,
                 initial_rho,
                 initial_drift,
                 initial_temp,
                 affected_points,
                 velocity_grids=None,
                 species=None,
                 initial_state=None):
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         velocity_grids,
                         species,
                         initial_state)
        return

    @property
    def subclass(self):
        return 'ConstantPointRule'

    #####################################
    #            Computation            #
    #####################################
    def collision(self, data):
        bp_cp.euler_scheme(data, self.affected_points)
        # Todo replace by bp_cp.no_collisions(data, self.affected_points)
        # before that, implement proper initialization
        return

    def transport(self, data):
        pass


# Todo This is not tested!
class BoundaryPointRule(Rule):
    def __init__(self,
                 initial_rho,
                 initial_drift,
                 initial_temp,
                 reflection_rate_inverse,
                 reflection_rate_elastic,
                 reflection_rate_thermal,
                 absorption_rate,
                 affected_points,
                 surface_normal=None,
                 incoming_velocities=None,
                 reflected_indices_inverse=None,
                 reflected_indices_elastic=None,
                 velocity_grids=None,
                 species=None,
                 initial_state=None):
        params = {key: value for (key, value) in locals().items()
                  if key not in ["self", "__class__"]}
        self.check_parameters(**params)
        self.reflection_rate_inverse = np.array(reflection_rate_inverse,
                                                dtype=float)
        self.reflection_rate_elastic = np.array(reflection_rate_elastic,
                                                dtype=float)
        self.reflection_rate_thermal = np.array(reflection_rate_thermal,
                                                dtype=float)
        self.absorption_rate = np.array(absorption_rate,
                                        dtype=float)
        # Either the incoming velocities and reflection indices
        # are given as parameters
        if surface_normal is None:
            assert reflected_indices_inverse is not None
            assert reflected_indices_elastic is not None
            assert incoming_velocities is not None
            self.reflected_indices_inverse = np.array(reflected_indices_inverse,
                                                      dtype=int)
            self.reflected_indices_elastic = np.array(reflected_indices_elastic,
                                                      dtype=int)
            self.incoming_velocities = np.array(incoming_velocities,
                                                dtype=int)
        # Or the incoming velocities and reflection indices
        # must be generated based on the surface normal and the velocity_grid
        else:
            self.incoming_velocities = self.compute_incoming_velocities(velocity_grids, surface_normal)
            self.reflected_indices_inverse = self.compute_reflected_indices_inverse(velocity_grids)
            self.reflected_indices_elastic = self.compute_reflected_indices_elastic(velocity_grids, surface_normal)
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         velocity_grids,
                         species,
                         initial_state)
        self.check_integrity()
        return

    @property
    def subclass(self):
        return 'BoundaryPointRule'

    @staticmethod
    def compute_incoming_velocities(velocity_grids, surface_normal):
        # the incoming velocities are used to calculate the inflow during transport
        # we calculate the scalar product for each entry and check if its > 0
        # thus the velocity points towards the border
        incoming_velocities = np.where(velocity_grids.iMG @ surface_normal > 0)[0]
        return incoming_velocities

    @staticmethod
    def compute_reflected_indices_inverse(velocity_grids):
        reflected_indices_inverse = np.zeros(velocity_grids.size, dtype=int)
        for (idx_v, v) in enumerate(velocity_grids.iMG):
            spc = velocity_grids.get_specimen(idx_v)
            v_refl = -v
            idx_v_refl = velocity_grids.find_index(spc, v_refl)
            reflected_indices_inverse[idx_v] = idx_v_refl
        return reflected_indices_inverse

    @staticmethod
    def compute_reflected_indices_elastic(velocity_grids, surface_normal):
        reflected_indices_elastic = np.zeros(velocity_grids.size, dtype=int)
        # Todo only works in 1D
        for (idx_v, v) in enumerate(velocity_grids.iMG):
            spc = velocity_grids.get_specimen(idx_v)
            v_refl = np.array([-1, 1]) * v
            idx_v_refl = velocity_grids.find_index(spc, v_refl)
            reflected_indices_elastic[idx_v] = idx_v_refl
        return reflected_indices_elastic

    def compute_initial_state(self, velocity_grids, species):
        full_initial_state = super().compute_initial_state(velocity_grids, species)
        # compute outgoing velocities, by relfecting incoming velocities
        outgoing_velocities = self.reflected_indices_inverse[self.incoming_velocities]
        # Set initial state to zero for all non-outgoing velocities
        initial_state = np.zeros(full_initial_state.shape)
        initial_state[outgoing_velocities] = full_initial_state[outgoing_velocities]
        return initial_state

    #####################################
    #            Computation            #
    #####################################
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
        for idx_spc in range(data.n_spc):
            beg, end = data.v_range[idx_spc]
            inverse_inflow = self.reflection_rate_inverse[idx_spc] * inflow
            reflected_inflow[:, self.reflected_indices_inverse] += inverse_inflow

            elastic_inflow = self.reflection_rate_elastic[idx_spc] * inflow
            reflected_inflow[:, self.reflected_indices_elastic] += elastic_inflow

            thermal_inflow = bp_m.particle_number(
                self.reflection_rate_thermal[idx_spc] * inflow[..., beg:end],
                data.dv[idx_spc])
            initial_particles = bp_m.particle_number(
                self.initial_state[np.newaxis, beg:end],
                data.dv[idx_spc])
            reflected_inflow[..., beg:end] += (
                thermal_inflow / initial_particles
                * self.initial_state[beg:end]
            )
        return reflected_inflow

    #####################################
    #            Verification           #
    #####################################
    def check_integrity(self, complete_check=True, context=None):
        assert isinstance(self.initial_rho, np.ndarray)
        assert isinstance(self.initial_drift, np.ndarray)
        assert isinstance(self.initial_temp, np.ndarray)
        assert isinstance(self.affected_points, np.ndarray)
        self.check_parameters(
            subclass=self.subclass,
            initial_rho=self.initial_rho,
            initial_drift=self.initial_drift,
            initial_temp=self.initial_temp,
            reflection_rate_inverse=self.reflection_rate_inverse,
            reflection_rate_elastic=self.reflection_rate_elastic,
            reflection_rate_thermal=self.reflection_rate_thermal,
            absorption_rate=self.absorption_rate,
            affected_points=self.affected_points,
            incoming_velocities=self.incoming_velocities,
            reflected_indices_inverse=self.reflected_indices_inverse,
            reflected_indices_elastic=self.reflected_indices_elastic,
            initial_state=self.initial_state,
            complete_check=complete_check,
            context=context)
        return

    @staticmethod
    def check_parameters(subclass=None,
                         initial_rho=None,
                         initial_drift=None,
                         initial_temp=None,
                         reflection_rate_inverse=None,
                         reflection_rate_elastic=None,
                         reflection_rate_thermal=None,
                         absorption_rate=None,
                         affected_points=None,
                         surface_normal=None,
                         incoming_velocities=None,
                         reflected_indices_inverse=None,
                         reflected_indices_elastic=None,
                         velocity_grids=None,
                         species=None,
                         initial_state=None,
                         complete_check=False,
                         context=None):
        Rule.check_parameters(
            subclass=subclass,
            initial_rho=initial_rho,
            initial_drift=initial_drift,
            initial_temp=initial_temp,
            affected_points=affected_points,
            velocity_grids=velocity_grids,
            initial_state=initial_state,
            complete_check=complete_check,
            context=context)
        if initial_drift is not None:
            if isinstance(initial_drift, list):
                initial_drift = np.array(initial_drift, dtype=float)
            assert np.all(initial_drift == 0), (
                "BoundaryPointRules must have no drift! Drift ="
                "{}".format(initial_drift)
            )

        rates = [reflection_rate_inverse,
                 reflection_rate_elastic,
                 reflection_rate_thermal,
                 absorption_rate]
        for rate in rates:
            if rate is not None:
                if isinstance(rate, list):
                    rate = np.array(rate, dtype=float)
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
        if all(rate is not None for rate in rates):
            assert len({len(rate) for rate in rates}) == 1, (
                "All rates must have the same length (number of species)."
                "Rates = {}".format(rates)
            )
            assert np.all(np.sum(rates, axis=0) == 1.0), (
                "Reflection/Absorption rates must sum up to 1 for each species."
                "Rates = {}".format(rates)
            )

        if surface_normal is not None:
            assert isinstance(surface_normal, np.ndarray)
            assert surface_normal.dtype == int
            assert np.all(np.abs(surface_normal) <= 1), (
                "A surface normal must have entries from [-1, 0, 1]."
                "surface_normal = {}".format(surface_normal)
            )

        for idx_array in [incoming_velocities,
                          reflected_indices_inverse,
                          reflected_indices_elastic]:
            if idx_array is not None:
                assert isinstance(idx_array, np.ndarray)
                assert idx_array.dtype == int
                assert len(set(idx_array)) == idx_array.size, (
                    "Index arrays must be unique indices!"
                    "idx_array:\n{}".format(idx_array)
                )

        for idx_array in [reflected_indices_inverse,
                          reflected_indices_elastic]:
            if idx_array is not None:
                assert np.all(idx_array[idx_array] == np.arange(idx_array.size)), (
                        "Any Reflection applied twice, must return the original."
                        "idx_array[idx_array]:\n{}".format(idx_array[idx_array])
                    )
