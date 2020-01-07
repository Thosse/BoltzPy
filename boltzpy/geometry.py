
import numpy as np
import h5py

import boltzpy as bp
import boltzpy.constants as bp_c


# TODO edit apply/affected points to be stored as dimensional tuples
class Geometry:
    r"""Describes the spatial geometry of the Simulation.


    Parameters
    ----------
    ndim : :obj:`int`
        The number of space dimensions.
        Must be in :const:`~boltzpy.constants.SUPP_GRID_DIMENSIONS`.
    shape : :obj:`tuple` [:obj:`int`]
        Shape of the space grid.
        Tuple of length :attr:`ndim`.
    rules : :obj:`~numpy.array` [:obj:`Rule`], optional
        List of Initialization :obj:`Rules<Rule>`

    Attributes
    ----------
    ndim : :obj:`int`
        The number of space dimensions.
        Must be in :const:`~boltzpy.constants.SUPP_GRID_DIMENSIONS`.
    shape : :obj:`tuple` [:obj:`int`]
        Shape of the space grid.
        Tuple of length :attr:`ndim`.
    rules : :obj:`~numpy.array` [:obj:`Rule`], optional
        List of Initialization :obj:`Rules<Rule>`

    """
    def __init__(self, ndim=None, shape=None, rules=None):
        # TODO move into check_integrity
        if ndim is not None:
            assert isinstance(ndim, int)
            assert ndim in bp_c.SUPP_GRID_DIMENSIONS
        if shape is not None:
            assert isinstance(shape, tuple)
            assert all(isinstance(line_size, int) for line_size in shape)
            assert all(line_size > 0 for line_size in shape)
        if ndim is not None and shape is not None:
            assert len(shape) == ndim
        if rules is not None:
            if isinstance(rules, list):
                rules = np.array(rules, dtype=bp.Rule)
            assert isinstance(rules, np.ndarray)
            assert rules.ndim == 1
            assert rules.dtype == bp.Rule
            for rule in rules:
                rule.check_integrity()

        self.ndim = ndim
        self.shape = shape
        if rules is not None:
            if isinstance(rules, list):
                self.rules = np.array(rules, dtype=bp.Rule)
            else:
                self.rules = rules
        else:
            self.rules = np.empty((0,), dtype=bp.Rule)
        return

    #: :obj:`dict` : Default ascii char, for terminal print
    DEFAULT_ASCII = {"Inner Point": 'o',
                     'Boundary Point': '#',
                     'Constant_IO_Point': '>',
                     'Time_Variant_IO_Point': '~',
                     None: '?'
                     }

    @property
    def size(self):
        if self.shape is not None:
            return np.prod(self.shape)
        else:
            return None

    @property
    def size_of_model(self):
        if self.rules.size == 0:
            return None
        sizes = [rule.size_of_model for rule in self.rules]
        # all model_sizes must be equal
        assert len(set(sizes)) == 1
        return sizes[0]

    # Todo Add test, that old indices are removed
    def add_rule(self,
                 behaviour_type,
                 initial_rho,
                 initial_drift,
                 initial_temp,
                 affected_points):
        """Add a new :class:`initialization rule <Rule>` to :attr:`rule_arr`.

        Parameters
        ----------
        behaviour_type : :obj:`str`
            Category of the :class:`P-Grid <boltzpy.Grid>` point.
            Must be in
            :const:`~boltzpy.constants.SUPP_BEHAVIOUR_TYPES`.
        initial_rho : :obj:`~numpy.array` [:obj:`float`]
        initial_drift : :obj:`~numpy.array` [:obj:`float`]
        initial_temp : :obj:`~numpy.array` [:obj:`float`]
        affected_points : :obj:`list` [:obj:`int`]
            Contains flat indices of
            :class:`P-Grid <boltzpy.Grid>` points.
        """
        # create a clean dictionary of parameters, without Nones
        parameters = {key: value
                      for (key, value) in locals().items()
                      if value is not None and key is not 'behaviour_type'}

        # choose derived class for new rule
        rule_class = bp.Rule.child_class(behaviour_type)
        rule_class.check_parameters(**parameters)

        # create rule and add to rules
        new_rule = rule_class(**parameters)
        self.rules = np.append(self.rules, [new_rule])
        self.apply_rule(rule=new_rule, affected_points=affected_points)
        self.check_integrity(complete_check=False)
        return

    @property
    def is_set_up(self):
        if any(attr is None for attr in self.__dict__.values()):
            return False
        if any(not rule.is_set_up for rule in self.rules):
            return False
        else:
            self.check_integrity()
            return True

    def setup(self, sv_grid):
        for rule in self.rules:
            rule.setup(sv_grid)
        return

    def apply_rule(self,
                   rule,
                   affected_points):
        """Add a new :class:`initialization rule <Rule>` to :attr:`rule_arr`.

        Parameters
        ----------
        rule : :obj:`~boltzpy.Rule`
        affected_points : :obj:`list` [:obj:`int`]
            Contains flat indices of
            :class:`P-Grid <boltzpy.Grid>` points.
        """
        assert isinstance(rule, bp.Rule)
        rule.check_integrity(complete_check=False)
        # remove previous occurrences
        for r in self.rules:
            occurrences = np.where(np.isin(r.affected_points, affected_points))
            r.affected_points = np.delete(r.affected_points, occurrences)

        # add points to rules affected_points
        rule.affected_points = np.append(rule.affected_points,
                                         affected_points)

        self.check_integrity(complete_check=False)
        return

    # Todo Only Temporary!
    @property
    def init_array(self):
        init_arr = np.full(self.shape, -1, dtype=int)
        for (idx_r, r) in enumerate(self.rules):
            for idx_p in r.affected_points:
                init_arr[idx_p] = idx_r
        return init_arr

    @property
    def initial_state(self):
        """Fully initiallized initial state.

        Returns
        -------
        state : :class:`~numpy.array` [:obj:`float`]
            The initialized PSV-Grid.
            Array of shape
            (:attr:`Simulation.p.size
            <boltzpy.Grid.size>`,
            :attr:`Simulation.sv.size
            <boltzpy.SVGrid.size>`).
        """
        if not self.is_set_up:
            return None
        shape = (self.size, self.size_of_model)
        state = np.zeros(shape=shape, dtype=float)
        for rule in self.rules:
            for p in rule.affected_points:
                state[p, :] = rule.initial_state
        # Todo state = state.reshape(shape + (model_size,))
        return state

    #####################################
    #            Computation            #
    #####################################
    def collision(self, data):
        for rule in self.rules:
            rule.collision(data)
        return

    def transport(self, data):
        for rule in self.rules:
            rule.transport(data)
        # update data.state (transport writes into data.result)
        data.state[...] = data.result[...]
        return

    #####################################
    #           Serialization           #
    #####################################
    @staticmethod
    def load(hdf5_group):
        """Set up and return a :class:`Geometry` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`

        Returns
        -------
        self : :class:`Geometry`
        """
        assert isinstance(hdf5_group, h5py.Group)
        assert hdf5_group.attrs["class"] == "Geometry"

        # read parameters from file
        params = dict()
        if "Dimensions" in hdf5_group.keys():
            params["ndim"] = int(hdf5_group["Dimensions"][()])
        if "Shape" in hdf5_group.keys():
            # cast into tuple of ints
            shape = hdf5_group["Shape"][()]
            params["shape"] = tuple(int(width) for width in shape)
        # load rules
        n_rules = 0
        if "Number of Rules" in hdf5_group.attrs.keys():
            n_rules = hdf5_group.attrs["Number of Rules"][()]
        params["rules"] = np.empty(shape=(n_rules,), dtype=bp.Rule)
        # iteratively read the rules
        for idx_rule in range(n_rules):
            key = "Rule_" + str(idx_rule)
            params["rules"][idx_rule] = bp.Rule.load(hdf5_group[key])

        self = Geometry(**params)
        self.check_integrity(complete_check=False)
        return self

    def save(self, hdf5_group):
        """Write the main parameters of the :obj:`Geometry` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity(complete_check=False)

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = "Geometry"

        # write all set attributes to file
        if self.ndim is not None:
            hdf5_group["Dimensions"] = self.ndim
        if self.shape is not None:
            hdf5_group["Shape"] = self.shape
        hdf5_group.attrs["Number of Rules"] = self.rules.size
        for (rule_idx, rule) in enumerate(self.rules):
            hdf5_group.create_group("Rule_" + str(rule_idx))
            rule.save(hdf5_group["Rule_" + str(rule_idx)])

        # check that the class can be reconstructed from the save
        other = Geometry.load(hdf5_group)
        assert self == other
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self,
                        complete_check=True,
                        context=None):
        """Sanity Check.

        Assert all conditions in :meth:`check_parameters`.

        Parameters
        ----------
        complete_check : :obj:`bool`, optional
            If True, then all attributes must be assigned (not None).
            If False, then unassigned attributes are ignored.
        context : :class:`Simulation`, optional
            The Simulation, which this instance belongs to.
            This allows additional checks.
        """
        self.check_parameters(ndim=self.ndim,
                              shape=self.shape,
                              rules=self.rules,
                              complete_check=complete_check,
                              context=context)
        return

    @staticmethod
    def check_parameters(ndim=None,
                         shape=None,
                         rules=None,
                         complete_check=False,
                         context=None):
        """Sanity Check.

        Checks integrity of given parameters and their interactions.

        Parameters
        ----------
        ndim : :obj:`int`, optional
        shape : :obj:`tuple` [:obj:`int`], optional
        rules : :obj:`~numpy.array` [:obj:`Rule`], optional
        complete_check : :obj:`bool`, optional
            If True, then all parameters must be set (not None).
            If False, then unassigned parameters are ignored.
        context : :class:`Simulation`, optional
            The Simulation, which this instance belongs to.
            This allows additional checks.
        """
        assert isinstance(complete_check, bool)
        # For complete check, assert that all parameters are assigned
        if complete_check is True:
            assert all(param_val is not None
                       for (param_key, param_val) in locals().items()
                       if param_key != "context")
        if context is not None:
            assert isinstance(context, bp.Simulation)

        # check all parameters, if set
        if ndim is not None:
            assert isinstance(ndim, int)
            assert ndim in bp_c.SUPP_GRID_DIMENSIONS

        if shape is not None:
            assert isinstance(shape, tuple)
            assert all(isinstance(width, int) for width in shape)
            assert (all(width >= 2 for width in shape)
                    or all(width == 1 for width in shape))
            if context is not None and context.p.shape is not None:
                assert shape == context.p.shape

        if rules is not None:
            assert isinstance(rules, np.ndarray)
            assert rules.dtype == bp.Rule
            assert rules.ndim == 1
            for rule in rules:
                assert isinstance(rule, bp.Rule)
                rule.check_integrity(complete_check=complete_check,
                                     context=context)
            # all points must be affected at most once
            affected_points = set()
            count_affected_points = 0
            for rule in rules:
                affected_points.update(rule.affected_points)
                count_affected_points += rule.affected_points.size
            assert len(affected_points) == count_affected_points
            if context is not None and context.p.size is not None:
                assert count_affected_points <= context.p.size
            # all points must be affected at least once
            if complete_check:
                assert len(affected_points) == np.prod(shape)
            # All rules must work on the same model
            size_of_model = set(rule.size_of_model
                                for rule in rules)
            assert len(size_of_model) in [0, 1]

        # check correct attribute relations
        if ndim is not None and shape is not None:
            assert len(shape) == ndim

        return

    def __eq__(self, other):
        if not isinstance(other, Geometry):
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
        """:obj:`str` :
        A human readable string which describes all attributes of the instance."""
        description = ''
        description += "Dimension = {}\n".format(self.ndim)
        description += "Shape = {}\n".format(self.shape)
        description += "Rules:\n\t"
        for (rule_idx, rule) in enumerate(self.rules):
            rule_str = rule.__str__(rule_idx).replace('\n', '\n\t')
            description += '\t' + rule_str
            description += '\n'
        return description
