
import numpy as np
import h5py
# TODO only temporary, replace with vectorized np method
import math

import boltzpy as bp
import boltzpy.constants as bp_c
import boltzpy.compute as bp_cp


class Rule:
    """Encapsulates all data to initialize a :class:`~boltzpy.Grid` point.

    A rule must be applied to every point of the
    :attr:`simulation.p <boltzpy.Simulation>`
    :class:`boltzpy.Grid`.
    It determines the points:

        * initial distribution in the velocity space
          based on :attr:`initial_rho`, :attr:`initial_drift`, and :attr:`initial_temp`.
        * behaviour (see :const:`~boltzpy.constants.SUPP_BEHAVIOUR_TYPES`)
          during the :mod:`computation`

    Parameters
    ----------
    initial_rho : :obj:`~numpy.array` [:obj:`float`]
    initial_drift : :obj:`~numpy.array` [:obj:`float`]
    initial_temp : :obj:`~numpy.array` [:obj:`float`]
    affected_points : :obj:`list`[:obj:`int`]
    initial_state : :obj:`~numpy.array` [:obj:`float`], optional
        Only given for testing purposes. Otherwise this is set by
        calling the :meth:`setup`

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
    # Todo test affected points
    # Todo test initial_state
    def __init__(self,
                 initial_rho=None,
                 initial_drift=None,
                 initial_temp=None,
                 affected_points=None,
                 initial_state=None):
        self.check_parameters(initial_rho=initial_rho,
                              initial_drift=initial_drift,
                              initial_temp=initial_temp,
                              affected_points=affected_points)
        self.initial_rho = initial_rho
        self.initial_drift = initial_drift
        self.initial_temp = initial_temp
        if affected_points is None:
            self.affected_points = np.empty((0,), dtype=int)
        else:
            self.affected_points = np.array(affected_points, dtype=int)
        # if initial state is given, generate the other parameters from it
        self.initial_state = initial_state
        self.check_integrity(complete_check=False)
        return

    @property
    def behaviour_type(self):
        """
        :obj:`str`
            Determines the behaviour during the simulation.
            Must be in :const:`~boltzpy.constants.SUPP_BEHAVIOUR_TYPES`.
        """
        raise NotImplementedError

    @staticmethod
    def child_class(behaviour_type):
        if behaviour_type == 'Inner Point':
            return InnerPointRule
        elif behaviour_type == 'Constant Point':
            return ConstantPointRule
        elif behaviour_type == 'Boundary Point':
            return BoundaryPointRule
        else:
            raise NotImplementedError

    @property
    def dimension(self):
        self.check_parameters(initial_rho=self.initial_rho,
                              initial_drift=self.initial_drift,
                              initial_temp=self.initial_temp)
        if self.initial_drift is None:
            return None
        else:
            return self.initial_drift.shape[1]

    @property
    def number_of_species(self):
        self.check_parameters(initial_rho=self.initial_rho,
                              initial_drift=self.initial_drift,
                              initial_temp=self.initial_temp)
        if self.initial_rho is not None:
            return self.initial_rho.size
        if self.initial_temp is not None:
            return self.initial_temp.size
        if self.initial_drift is not None:
            return self.initial_drift.shape[0]
        return None

    # Todo use this function to vecorize setup()
    # def density(self, velocity):
    #     pass

    @property
    def is_set_up(self):
        if any(attr is None for attr in self.__dict__.values()):
            return False
        else:
            self.check_integrity()
            return True

    @property
    def size_of_model(self):
        if self.initial_state is not None:
            return self.initial_state.size
        else:
            return None

    # Todo (ST) use correct initialization form (mass in exponent)
    # Todo (LT) use a Newton Scheme for correct values
    def setup(self, svgrid):
        assert isinstance(svgrid, bp.SVGrid)
        assert self.dimension == svgrid.ndim

        # setup initial state
        self.initial_state = np.zeros(svgrid.size)
        for idx_spc in range(self.number_of_species):
            rho = self.initial_rho[idx_spc]
            drift = self.initial_drift[idx_spc]
            temp = self.initial_temp[idx_spc]

            [begin, end] = svgrid.index_range[idx_spc]
            v_grid = svgrid.iMG[begin:end]
            dv = svgrid.delta
            for (i_v, v) in enumerate(v_grid):
                # Physical Velocity
                pv = np.array(dv * v)
                diff_v = np.sum((pv - drift) ** 2)
                self.initial_state[begin + i_v] = rho * math.exp(-0.5 * (diff_v / temp))
            # Todo read into Rjasanov's script and do this correctly
            # Todo THIS IS CURRENTLY WRONG! ONLY TEMPORARY FIX
            # Adjust initialized values, to match configurations
            adj = self.initial_state[begin:end].sum()
            self.initial_state[begin:end] *= rho / adj
        return

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
        # read parameters from file
        params = dict()
        # Todo Rename into Rho / Density
        if "Mass" in hdf5_group.keys():
            params["initial_rho"] = hdf5_group["Mass"][()]
        if "Mean Velocity" in hdf5_group.keys():
            params["initial_drift"] = hdf5_group["Mean Velocity"][()]
        if "Temperature" in hdf5_group.keys():
            params["initial_temp"] = hdf5_group["Temperature"][()]
        if "Affected Points" in hdf5_group.keys():
            params["affected_points"] = hdf5_group["Affected Points"][()]
        if "Initial State" in hdf5_group.keys():
            params["initial_state"] = hdf5_group["Initial State"][()]

        # choose derived class for new rule
        behaviour_type = hdf5_group["Behaviour Type"][()]
        rule_class = bp.Rule.child_class(behaviour_type)
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

        # write all set attributes to file
        if self.behaviour_type is not None:
            hdf5_group["Behaviour Type"] = self.behaviour_type
        if self.initial_rho is not None:
            hdf5_group["Mass"] = self.initial_rho
        if self.initial_drift is not None:
            hdf5_group["Mean Velocity"] = self.initial_drift
        if self.initial_temp is not None:
            hdf5_group["Temperature"] = self.initial_temp
        hdf5_group["Affected Points"] = self.affected_points
        if self.initial_state is not None:
            hdf5_group["Initial State"] = self.initial_state

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
        assert isinstance(self.affected_points, np.ndarray)
        self.check_parameters(behaviour_type=self.behaviour_type,
                              initial_rho=self.initial_rho,
                              initial_drift=self.initial_drift,
                              initial_temp=self.initial_temp,
                              affected_points=self.affected_points,
                              complete_check=complete_check,
                              context=context)
        return

    @staticmethod
    def check_parameters(behaviour_type=None,
                         initial_rho=None,
                         initial_drift=None,
                         initial_temp=None,
                         affected_points=None,
                         complete_check=False,
                         context=None):
        """Sanity Check.
        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        behaviour_type : :obj:`str`, optional
        initial_rho : :obj:`~numpy.array` [:obj:`float`], optional
        initial_drift : :obj:`~numpy.array` [:obj:`float`], optional
        initial_temp : :obj:`~numpy.array` [:obj:`float`], optional
        affected_points : :obj:`list`[:obj:`int`]
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
                       if param_key != "context")
        if context is not None:
            assert isinstance(context, bp.Simulation)

        # Set up basic constants
        if context is not None:
            n_species = context.s.size
        else:
            n_species = None

        # check all parameters, if set
        if behaviour_type is not None:
            assert isinstance(behaviour_type, str)
            assert behaviour_type in bp_c.SUPP_BEHAVIOUR_TYPES

        if initial_rho is not None:
            assert isinstance(initial_rho, np.ndarray)
            assert initial_rho.dtype == float
            assert initial_rho.ndim == 1
            assert np.min(initial_rho) > 0
            if n_species is not None:
                assert initial_rho.shape == (n_species,)
            n_species = initial_rho.shape[0]

        if initial_drift is not None:
            assert isinstance(initial_drift, np.ndarray)
            assert initial_drift.dtype == float
            assert initial_drift.ndim == 2
            if n_species is not None:
                assert initial_drift.shape[0] == n_species
            n_species = initial_drift.shape[0]
            assert initial_drift.shape[1] in [2, 3]
            if context is not None and context.sv.ndim is not None:
                assert initial_drift.shape[1] == context.sv.ndim

        if initial_temp is not None:
            assert isinstance(initial_temp, np.ndarray)
            assert initial_temp.dtype == float
            assert initial_temp.ndim == 1
            assert np.min(initial_temp) > 0
            if n_species is not None:
                assert initial_temp.shape == (n_species,)
            # n_species = initial_temp.shape[0]

        if affected_points is not None:
            if isinstance(affected_points, list):
                affected_points = np.array(affected_points, dtype=int)
            assert isinstance(affected_points, np.ndarray)
            assert affected_points.ndim == 1
            assert affected_points.dtype == int
            if affected_points.size != 0:
                assert np.min(affected_points) >= 0
            if context is not None and context.geometry.shape is not None:
                assert np.max(affected_points) < context.geometry.size
            assert affected_points.size == len(set(affected_points))

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

    def __str__(self, idx=None):
        """Convert the instance to a string, describing all attributes."""
        description = ''
        if idx is not None:
            description += 'Rule_{}:\n'.format(idx)
        description += 'Behaviour Type = ' + self.behaviour_type + '\n'
        description += 'Rho:\n\t'
        description += self.initial_rho.__str__() + '\n'
        description += 'Drift:\n\t'
        description += self.initial_drift.__str__().replace('\n', '\n\t') + '\n'
        description += 'Temperature: \n\t'
        description += self.initial_temp.__str__() + '\n'
        return description


class InnerPointRule(Rule):
    def __init__(self,
                 initial_rho=None,
                 initial_drift=None,
                 initial_temp=None,
                 affected_points=None,
                 initial_state=None):
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         initial_state)
        return

    @property
    def behaviour_type(self):
        return 'Inner Point'

    def collision(self, data):
        bp_cp.euler_scheme(data, self.affected_points)
        return

    def transport(self, data):
        bp_cp.transport_fdm_inner(data, self.affected_points)
        return


class ConstantPointRule(Rule):
    def __init__(self,
                 initial_rho=None,
                 initial_drift=None,
                 initial_temp=None,
                 affected_points=None,
                 initial_state=None):
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         initial_state)
        return

    @property
    def behaviour_type(self):
        return 'Constant Point'

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
                 reflection_rate_inverse,
                 reflection_rate_elastic,
                 reflection_rate_thermal,
                 reflection_temperature,
                 absorption_rate,
                 surface_normal,    # TODO more complicated in 2d
                 initial_rho=None,
                 initial_drift=None,
                 initial_temp=None,
                 affected_points=None,
                 initial_state=None):
        super().__init__(initial_rho,
                         initial_drift,
                         initial_temp,
                         affected_points,
                         initial_state)
        self.reflection_rate_inverse = reflection_rate_inverse
        self.reflection_rate_elastic = reflection_rate_elastic
        self.reflection_rate_thermal = reflection_rate_thermal
        self.reflection_temperature = reflection_temperature
        self.absorption_rate = absorption_rate
        self.surface_normal = surface_normal
        return

    @property
    def behaviour_type(self):
        return 'Boundary Point'

    def collision(self, data):
        bp_cp.no_collisions(data, self.affected_points)
        return

    def transport(self, data):
        bp_cp.no_transport(data, self.affected_points)
        return

