
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
        result = list()
        for rule in self.rules:
            result += list(rule.affected_points)
        assert len(result) == len(set(result))
        return result

    @property
    def unaffected_points(self):
        possible_points = set(range(int(self.size)))
        affected_points = set(self.affected_points)
        return possible_points - affected_points

    @property
    def size_of_model(self):
        sizes = [rule.initial_state.size for rule in self.rules]
        # todo assert all rule.initial_state.size must be equal in check_params
        assert len(set(sizes)) == 1
        return sizes[0]

    def add_rule(self, new_rule):
        """Add a :class:`Rule` to :attr:`rules` array.

        Parameters
        ----------
        new_rule : :obj:`~boltzpy.Rule`
            The Rule object to append
        """
        assert isinstance(new_rule, bp.Rule)
        assert set(new_rule.affected_points).issubset(self.unaffected_points)
        self.rules = np.append(self.rules, [new_rule])
        self.check_integrity()
        return

    @property
    def is_set_up(self):
        return len(self.affected_points) == self.size

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
            <boltzpy.Model.size>`).
        """
        assert self.is_set_up
        shape = (self.size, self.rules[0].initial_state.size)
        state = np.zeros(shape=shape, dtype=float)
        for rule in self.rules:
            state[rule.affected_points, :] = rule.initial_state
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
        # Todo this should be in the operator splitting
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
        shape = tuple(int(width) for width in hdf5_group["shape"][()])
        delta = float(hdf5_group["delta"][()])
        # load rules iteratively
        rules = np.empty(shape=hdf5_group["rules"].attrs["size"],
                         dtype=bp.Rule)
        for pos_rule in range(rules.size):
            rules[pos_rule] = bp.Rule.load(hdf5_group["rules"][str(pos_rule)])
        # Initialize
        self = Geometry(shape, delta, rules)
        return self

    def save(self, hdf5_group):
        """Write the main parameters of the :obj:`Geometry` instance
        into the HDF5 group.

        Parameters
        ----------
        hdf5_group : :obj:`h5py.Group <h5py:Group>`
        """
        assert isinstance(hdf5_group, h5py.Group)
        self.check_integrity()

        # Clean State of Current group
        for key in hdf5_group.keys():
            del hdf5_group[key]
        hdf5_group.attrs["class"] = self.__class__.__name__

        # write all parameters
        hdf5_group["shape"] = self.shape
        hdf5_group["delta"] = self.delta
        # write rules
        hdf5_group.create_group("rules")
        hdf5_group["rules"].attrs["class"] = "Array"
        hdf5_group["rules"].attrs["size"] = self.rules.size
        # save all rules iteratively
        for (idx_rule, rule) in enumerate(self.rules):
            key_rule = str(idx_rule)
            hdf5_group["rules"].create_group(key_rule)
            rule.save(hdf5_group["rules"][key_rule])
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
        # all points must be affected at most once
        affected_points = list()
        for rule in self.rules:
            affected_points += list(rule.affected_points)
            assert len(affected_points) == len(set(affected_points)), (
                "Some points are affected by more than one rule:"
                "{}".format(affected_points)
            )
        # All rules must work on the same model
        assert len({(rule.initial_state.size for rule in self.rules)}) < 2, (
            "Some rules have different model size!"
        )
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
