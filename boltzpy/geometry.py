
import numpy as np

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
        rules = np.array(rules, dtype=bp.BaseRule)
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
    def compute(self, sim):
        """Executes a single time step, by operator splitting"""
        # execute a single transport step
        for rule in self.rules:
            rule.transport(sim)
        # update data.state (transport writes into data.result)
        (sim.state, sim.interim) = (sim.interim, sim.state)
        # executie s single collision step
        for rule in self.rules:
            rule.collision(sim)
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check"""
        bp.Grid.check_integrity(self)
        assert self.spacing == 1
        assert not self.is_centered

        assert isinstance(self.rules, np.ndarray)
        assert self.rules.dtype == bp.BaseRule
        assert self.rules.ndim == 1
        for rule in self.rules:
            assert isinstance(rule, bp.BaseRule)
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
