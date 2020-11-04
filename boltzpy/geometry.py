
import numpy as np
import h5py

import boltzpy as bp


class Geometry(bp.Grid):
    r"""Describes the spatial geometry of the Simulation.


    Parameters
    ----------
    shape : :obj:`array_like` [:obj:`int`]
        Shape of the space grid.
    delta : :obj:`float`
        Smallest possible step size of the :obj:`Grid`.
    rules : :obj:`array_like` [:obj:`Rule`], optional
        List of Initialization :obj:`Rules<Rule>`
    """
    def __init__(self,  shape, delta, rules):
        rules = np.array(rules, dtype=bp.Rule)
        self.rules = rules
        super().__init__(shape,
                         delta,
                         spacing=1,
                         is_centered=False)
        return

    #: :obj:`dict` : Default ascii char, for terminal print
    DEFAULT_ASCII = {"Inner Point": 'o',
                     'Boundary Point': '#',
                     'Constant_IO_Point': '>',
                     'Time_Variant_IO_Point': '~',
                     None: '?'
                     }

    @property
    def affected_points(self):
        result = np.concatenate([r.affected_points for r in self.rules])
        return result

    @property
    def unaffected_points(self):
        possible_points = set(np.arange(self.size))
        affected_points = set(self.affected_points)
        return np.array(list(possible_points - affected_points))

    # Todo remove?
    @property
    def model_size(self):
        sizes = [rule.initial_state.size for rule in self.rules]
        assert len(set(sizes)) == 1
        return sizes[0]

    @staticmethod
    def parameters():
        return {"shape",
                "delta",
                "rules"}

    @staticmethod
    def attributes():
        attrs = Geometry.parameters()
        attrs.update(bp.Grid.attributes())
        attrs.update({"affected_points",
                      "unaffected_points",
                      "initial_state"})
        return attrs

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
            <boltzpy.Model.size>`).
        """
        shape = (self.size, self.model_size)
        state = np.zeros(shape=shape, dtype=float)
        for rule in self.rules:
            state[rule.affected_points, :] = rule.initial_state
        return state

    #####################################
    #            Computation            #
    #####################################
    def compute(self, data):
        """Executes a single time step, by operator splitting"""
        # executie s single transport step
        for rule in self.rules:
            rule.transport(data)
        # Todo this should be in the operator splitting
        # update data.state (transport writes into data.result)
        data.state[...] = data.result[...]
        assert np.all(data.state >= 0)
        # increase current_timestep counter
        data.t += 1
        # executie s single collision step
        for rule in self.rules:
            rule.collision(data)
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
        parameters = dict()
        for param in Geometry.parameters():
            # load rules separately
            if param == "rules":
                rules = np.empty(hdf5_group["rules"].attrs["size"],
                                 dtype=bp.Rule)
                for r in range(rules.size):
                    rules[r] = bp.Rule.load(hdf5_group["rules"][str(r)])
                parameters["rules"] = rules
            else:
                parameters[param] = hdf5_group[param][()]
        return Geometry(**parameters)

    def save(self, hdf5_group, write_all=False):
        """Write the main parameters of the :obj:`Geometry` instance
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
            # rules are saved separately in a subgroup
            if attr == "rules":
                hdf5_group.create_group("rules")
                hdf5_group["rules"].attrs["class"] = "Array"
                hdf5_group["rules"].attrs["size"] = self.rules.size
                for (idx_rule, rule) in enumerate(self.rules):
                    key_rule = str(idx_rule)
                    hdf5_group["rules"].create_group(key_rule)
                    rule.save(hdf5_group["rules"][key_rule])
            else:
                hdf5_group[attr] = self.__getattribute__(attr)
        # check that the class can be reconstructed from the save
        other = Geometry.load(hdf5_group)
        assert self == other
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check"""
        super().check_integrity()
        assert self.spacing == 1
        assert not self.is_centered

        assert isinstance(self.rules, np.ndarray)
        assert self.rules.dtype == bp.Rule
        assert self.rules.ndim == 1
        for rule in self.rules:
            assert isinstance(rule, bp.Rule)
            rule.check_integrity()
        # all points are affected at most once
        assert self.affected_points.size == len(set(self.affected_points)), (
                "Some points are affected by more than one rule:"
                "{}".format(self.affected_points))
        # all points are affected at least once
        assert self.unaffected_points.size == 0
        # rules affect only points in the geometry
        assert np.max(self.affected_points) < self.size
        # All rules must work on the same model(size)
        assert len({rule.initial_state.size for rule in self.rules}) < 2, (
            "Some rules have different model size!")
        return

    def __str__(self, write_physical_grids=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance."""
        description = super().__str__(write_physical_grids)
        for (key, value) in self.__dict__.items():
            if key == "rules":
                for (rule_idx, rule) in enumerate(self.rules):
                    description += "rules[{}]:\n".format(rule_idx)
                    rule_str = rule.__str__().replace('\n', '\n\t')
                    description += '\t' + rule_str
                    description += '\n'
                continue
            description += '{key}:\n\t{value}\n'.format(
                key=key,
                value=value.__str__().replace('\n', '\n\t'))
        return description
