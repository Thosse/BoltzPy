from boltzmann.configuration import species as b_spc
from boltzmann.configuration import grid as b_grd
from boltzmann.configuration import svgrid as b_svg
from boltzmann.initialization import rule as b_rul

import numpy as np


class Initialization:
    """Handles initialization instructions and creates
    PSV-Grid and :attr:`p_flag`.

    * Collects initialization :attr:`rules`.
    * Categorizes each P-Grid point in :attr:`p_flag`,
      to specify its initial state and behavior in the simulation.
    * Creates the initialized PSV-Grids
      (:attr:`~boltzmann.calculation.Calculation.data`,
      :attr:`~boltzmann.calculation.Calculation.result`)
      for :class:`~boltzmann.calculation.Calculation`.

    .. todo::
        - Guarantee the Order of Initialization.rules (inner points first)
          Check if order needs to be changed, when adding new rule
          -> change values in p_flag.
        - @apply_rule: implement different 'shapes' to apply rules
          (e.g. a line with specified width,
          a ball with specified radius/diameter, ..).
        - @apply_rule: current mode (cuboid)
          -> switch between convex hull and span?
        - Figure out nice way to implement boundary points
        - sphinx: link PSV-Grid to Calculation.data
          and p_flag to Calculation.p_flag
          in Initialization-Docstring

    Attributes
    ----------
    p_flag : np.ndarray(int)
        Controls the behavior of each P-Grid point
        during calculation and its initial values.
        Let i_p be an index of P-Grid, then
        p_flag[i_p] is an index that refers to its
        initialization rule in :attr:`rules`
        Array of shape=(p.n[0],...,p.n[-2])
        and dtype=int.
    rules : array(:class:`~boltzmann.initialization.Rule`)
        Array of all defined :class:`Rules`, sorted by type/class or Rule.
        The Array is ordered as:
        [:class:`InnerPointRule`,
        :class:`BoundaryPointRule`,
        :class:`IOVariableRule`,
        :class:`IOConstantRule`].
    block_index : array(int)
        Marks the beginning/end of each rule-block in
        :attr:`rules`.
        For each Category-index i_c,
        the block_index[i_c] marks the first index i_r,
        such that rules[i_r] is a rule of category i_c.
        Note that block_index[4] marks the total number/length of rules.
        Array of shape=(5,), and dtype=int

    Notes
    -----
    Each P-Grid point fits
    into exactly one of the following Categories:

    * **Inner Point (Default)**:

      * both transport and collisions are applied normally

    * **Boundary Point**:

      * no collision step
      * additional reflection step after every transport step

    * **Constant Input/Output Point**:

      * no collision-step
      * no transport-step
      * Distribution is constant over the whole simulation.

    * **Time Variant Input/Output Point**:

      * no collision-step,
      * no transport-step,
      * Distribution is freshly initialized in every time step

    """
    SPECIFIED_CATEGORIES = ['Inner_Point',
                            'Boundary_Point',
                            'Constant_IO_Point',
                            'Time_Variant_IO_Point']

    def __init__(self,
                 species=b_spc.Species(),
                 p_grid=b_grd.Grid(),
                 sv_grid=b_svg.SVGrid):
        # Todo Check if the links are really necessary
        # Todo check if init or RuleArray needs those instances?
        self._s = species
        self._p = p_grid
        self._sv = sv_grid
        # Todo this needs to be an array of Rule Instances/Objects
        # Todo check if it actually does what it's supposed to
        self.rules = np.empty(shape=(0,), dtype=b_rul.Rule)
        self.block_index = np.zeros(shape=(5,),
                                    dtype=int)
        p_shape = tuple(self._p.n[0:-1])
        self.p_flag = np.full(shape=p_shape,
                              fill_value=-1,
                              dtype=int)

    def add_rule(self,
                 category,
                 rho_list,
                 drift_list,
                 temp_list,
                 TMP_OPTIONAL_STUFF=None,
                 name=''):
        """Adds a new initialization
        :class:`~boltzmann.initialization.Rule` for P-Grid points
        to the list of :attr:`rules`

        The added
        :class:`~boltzmann.initialization.Rule`
        initializes the velocity space of each specimen
        based on their conserved quantities
        mass (:attr:`rho`),
        mean velocity (:attr:`drift`)
        and temperature (:attr:`temp`).

        Parameters
        ----------
        category : str
            Specifies what subclass of Rule is to be added.
            (:class:`InnerPointRule`).
            Must be an Element of
            :const:`Initialization.SPECIFIED_CATEGORIES`
        rho_list : array_like
            List of the parameter rho, for each specimen.
            Rho correlates to the total weight/amount of particles in
            the area of the P-Grid point.,
        drift_list : array_like
            List of the parameter drift, for each specimen.
            Drift describes the mean velocity.
        temp_list : array_like
            List of the parameter temp, for each specimen.
            Temp describes the Temperature.
        name : str, optional
            Sets a name, for the points initialized with this rule.
        """
        assert category in Initialization.SPECIFIED_CATEGORIES
        # Depending on Rule Type -> Construct new_rule
        if category is Initialization.SPECIFIED_CATEGORIES[0]:
            category = 0
            new_rule = b_rul.Rule(category,
                                  rho_list,
                                  drift_list,
                                  temp_list,
                                  name)
        else:
            print('Unspecified Category: {}'
                  ''.format(category))
            assert False

        position = self.block_index[category]
        # Insert new rule into rules-array
        self.rules = np.insert(self.rules,
                               position,
                               [new_rule])
        # Adjust p_flag entries
        # TODO this is untested
        # TODO Check if this is done correctly, especially if rules are added
        # TODO that change the order
        for _val in np.nditer(self.p_flag,
                              op_flags=['readwrite']):
            if _val >= position:
                _val += 1
        # Adjust block_index
        for _val in np.nditer(self.block_index[1:],
                              op_flags=['readwrite']):
            if _val >= position:
                _val += 1
        return

    def check_integrity(self):
        assert self.rules.dtype == b_rul.Rule
        for (i_r, r) in enumerate(self.rules):
            r.check_integrity()
            assert r.cat
        length = len(Initialization.SPECIFIED_CATEGORIES)
        # A change of this length needs several changes in this module
        # and its submodules
        assert len(Initialization.SPECIFIED_CATEGORIES) is 4
        assert self.block_index.shape == (length+1,)
        # Todo assert shape of p_flag
        # p_shape == tuple(self._p.n[0:-1])
        assert np.min(self.p_flag) >= 0, 'Uninitialized P-Grid points'
        assert np.max(self.p_flag) <= self.block_index[-1], 'Undefined Rule'

    def print(self,
              physical_grid=False):
        print('=======INITIALIZATION=======')
        print('Number of Rules = '
              '{}'.format(self.rules.shape[0]))
        print('Block Indices = '
              '{}'.format(self.block_index))

        for i_c in range(4):
            print('Rules: {}'
                  ''.format(Initialization.SPECIFIED_CATEGORIES[i_c]))
            str_len = len(Initialization.SPECIFIED_CATEGORIES[i_c]) + 7
            print('-'*str_len)
            for r in self.rules[self.block_index[i_c]:
                                self.block_index[i_c+1]]:
                r.print()
        if physical_grid:
            print('Flag-Grid of P-Space:')
            print(self.p_flag)

    # def apply_init_rule(self,
    #                     rule,
    #                     p_min,
    #                     p_max,
    #                     p_flag):
    #     """Applies an (already defined) initialization rule on the P-Grid.
    #
    #     This methods changes the :attr:`p_flag` entry
    #     of the specified P-Grid points to the value of rule.
    #     This sets both their initial state and behaviour during
    #     :class:`~boltzmann.calculation.Calculation`
    #     to the ones stated in the indexed
    #     :attr:`~Initialization.rules`.
    #
    #     Parameters
    #     ----------
    #     rule : int
    #         Index of the to be applied rule in
    #         :attr:`arr`.
    #     p_min, p_max :  array_like(int)
    #         Indices of the boundaries on where to apply the rule.
    #         The rule is applied on all P-Grid points p, such that
    #         p_min[i] <= p[i] <= p_max[i]
    #         Indices are in vector form (indexing a non-flattened P-Grid).
    #     """
    #     assert 0 <= rule < self.arr.size
    #     if _p.dim is 1:
    #         assert type(p_min) is int
    #         assert type(p_max) is int
    #         p_flag[p_min:p_max+1] = rule
    #     elif self._p.dim is 2:
    #         assert type(p_min) in [list, np.ndarray]
    #         assert type(p_max) in [list, np.ndarray]
    #         assert np.array(p_min).size is self._p.dim
    #         assert np.array(p_max).size is self._p.dim
    #         self.p_flag[p_min[0]:p_max[0]+1,
    #                     p_min[1]:p_max[1]+1] = rule
    #     elif self._p.dim is 3:
    #         assert type(p_min) in [list, np.ndarray]
    #         assert type(p_max) in [list, np.ndarray]
    #         assert np.array(p_min).size is self._p.dim
    #         assert np.array(p_max).size is self._p.dim
    #         self.p_flag[p_min[0]:p_max[0]+1,
    #                     p_min[1]:p_max[1]+1,
    #                     p_min[2]:p_max[2]+1] = rule
    #     return
    #
    # def create_psv_grid(self):
    #     """Generates the initialized PSV-Grid
    #     (:attr:`~boltzmann.calculation.Calculation.data`,
    #     :attr:`~boltzmann.calculation.Calculation.result`)
    #     and :attr:`~boltzmann.calculation.Calculation.p_flag`
    #     for :class:`~boltzmann.calculation.Calculation`.
    #
    #
    #     returns
    #     ----------
    #     psv : np.ndarray(float)
    #         The initialized PSV-Grid.
    #         Array of shape=(p.n[0],...,p.n[-2], sv.index[-1])
    #         and dtype=float.
    #     p_flag : np.ndarray(int)
    #         Controls the behavior of each P-Grid point
    #         during calculation.
    #         Array of shape=(p.n[0],...,p.n[-2])
    #         and dtype=int.
    #     """
    #     self._s.print()
