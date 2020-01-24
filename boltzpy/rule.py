
import numpy as np
import h5py
# TODO only temporary, replace with vectorized np method
import math

import boltzpy as bp
import boltzpy.constants as bp_c
import boltzpy.compute as bp_cp


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
            self.initial_state = self.compute_initial_state(velocity_grids)
        self.check_integrity(complete_check=False)
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

    # Todo (ST) use correct initialization form (mass in exponent)
    # Todo (LT) use a Newton Scheme for correct values
    def compute_initial_state(self, velocity_grids):
        assert isinstance(velocity_grids, bp.SVGrid)
        assert self.ndim == velocity_grids.ndim
        assert self.number_of_specimen == velocity_grids.number_of_grids
        initial_state = np.zeros(velocity_grids.size)

        for idx_spc in range(self.number_of_specimen):
            rho = self.initial_rho[idx_spc]
            drift = self.initial_drift[idx_spc]
            temp = self.initial_temp[idx_spc]
            [begin, end] = velocity_grids.index_range[idx_spc]
            v_grid = velocity_grids.iMG[begin:end]
            dv = velocity_grids.delta
            for (i_v, v) in enumerate(v_grid):
                # Physical Velocity
                pv = np.array(dv * v)
                diff_v = np.sum((pv - drift) ** 2)
                initial_state[begin + i_v] = rho * math.exp(-0.5 * (diff_v / temp))
            # Todo read into Rjasanov's script and do this correctly
            # Todo THIS IS CURRENTLY WRONG! ONLY TEMPORARY FIX
            # Adjust initialized values, to match configurations
            adj = initial_state[begin:end].sum()
            initial_state[begin:end] *= rho / adj
        return initial_state

    #####################################
    #            Computation            #
    #####################################
    def collision(self, data):
        raise NotImplementedError

    def transport(self, data):
        raise NotImplementedError

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
            assert initial_drift.shape[1] in [2, 3]
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
                if np.all(value != other_value):
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
                 initial_state=None):
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         velocity_grids,
                         initial_state)
        return

    @property
    def subclass(self):
        return 'InnerPointRule'

    def collision(self, data):
        bp_cp.euler_scheme(data, self.affected_points)
        return

    def transport(self, data):
        bp_cp.transport_fdm_inner(data, self.affected_points)
        return


class ConstantPointRule(Rule):
    def __init__(self,
                 initial_rho,
                 initial_drift,
                 initial_temp,
                 affected_points,
                 velocity_grids=None,
                 initial_state=None):
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         velocity_grids,
                         initial_state)
        return

    @property
    def subclass(self):
        return 'ConstantPointRule'

    def collision(self, data):
        bp_cp.euler_scheme(data, self.affected_points)
        # Todo replace by bp_cp.no_collisions(data, self.affected_points)
        # before that, implement proper initialization
        return

    def transport(self, data):
        bp_cp.no_transport(data, self.affected_points)
        return


# Todo add assertions
class BoundaryPointRule(Rule):
    def __init__(self,
                 initial_rho,
                 initial_temp,
                 reflection_rate_inverse,
                 reflection_rate_elastic,
                 reflection_rate_thermal,
                 absorption_rate,
                 surface_normal,    # TODO more complicated in 2d
                 affected_points,
                 velocity_grids=None,
                 initial_state=None):
        # BoundaryPointRules don't have a drift
        initial_drift = np.zeros((np.array(initial_rho).size,
                                  np.array(surface_normal).size),
                                 dtype=float)
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         velocity_grids,
                         initial_state)
        # Todo initial_state must be edited here, better to edit the method
        self.reflection_rate_inverse = reflection_rate_inverse
        self.reflection_rate_elastic = reflection_rate_elastic
        self.reflection_rate_thermal = reflection_rate_thermal
        self.absorption_rate = absorption_rate
        self.surface_normal = surface_normal
        return

    @property
    def subclass(self):
        return 'BoundaryPointRule'

    def collision(self, data):
        bp_cp.no_collisions(data, self.affected_points)
        return

    def transport(self, data):
        bp_cp.no_transport(data, self.affected_points)
        return

