from boltzmann.configuration import configuration as b_cnf
from boltzmann.initialization import rule as b_rul

import numpy as np
import math


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
        - Figure out nice way to implement boundary points
        - speed up init of psv grid <- ufunc's
        - @apply_rule: implement different 'shapes' to apply rules
          (e.g. a line with specified width,
          a ball with specified radius/diameter, ..).
          Switch between convex hull and span?
        - sphinx: link PSV-Grid to Calculation.data
          and p_flag to Calculation.p_flag
          in Initialization-Docstring

    Parameters
    ----------
    config : :class:`~boltzmann.configuration.Configuration`

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
                 config=b_cnf.Configuration()):
        self.config = config
        self.rules = np.empty(shape=(0,), dtype=b_rul.Rule)
        self.block_index = np.zeros(shape=(5,),
                                    dtype=int)
        p_shape = tuple(self.config.p.n)
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

        position = self.block_index[category+1]
        # Insert new rule into rules-array
        self.rules = np.insert(self.rules,
                               position,
                               [new_rule])
        # Adjust p_flag entries
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

    def apply_rule(self,
                   index_rule,
                   p_min,
                   p_max):
        """Applies an initialization rule from
        :attr:`~Initialization.rules`
        on the P-Grid.

        This methods changes the :attr:`p_flag` entry
        of the specified P-Grid points to the value of rule.
        This sets both their initial state and behaviour during
        :class:`~boltzmann.calculation.Calculation`
        to the ones specified in the respective
        :attr:`~Initialization.rules`.
        Note that the rule is applied on all P-Grid points p, such that
        p_min[i_p] <= p[i_p] < p_max[i_p]
        Indices are in vector form (indexing a non-flattened P-Grid).

        Parameters
        ----------
        index_rule : int
            Index of the to be applied rule in
            :attr:`rules`.
        p_min, p_max :  array_like(int)
            Indices of the boundary points.
            Mark the area where to apply the rule.

        """
        assert 0 <= index_rule < self.block_index[-1]
        dim = self.config.p.dim
        p_min = np.array(p_min)
        p_max = np.array(p_max)
        assert p_min.shape == (dim,)
        assert p_max.shape == (dim,)
        assert p_min.dtype == int
        assert p_max.dtype == int
        assert all(np.zeros(p_min.shape) <= p_min)
        assert all(p_min <= p_max)
        assert all(p_max <= self.config.p.n)

        if dim is 1:
            self.p_flag[p_min[0]:p_max[0]] = index_rule
        elif dim is 2:
            self.p_flag[p_min[0]:p_max[0],
                        p_min[1]:p_max[1]] = index_rule
        elif dim is 3:
            self.p_flag[p_min[0]:p_max[0],
                        p_min[1]:p_max[1],
                        p_min[2]:p_max[2]] = index_rule
        return

    def create_psv_grid(self):
        """Generates and returns the initialized PSV-Grid
        (:attr:`~boltzmann.calculation.Calculation.data`,
        :attr:`~boltzmann.calculation.Calculation.result`).

        Returns
        -------
        psv : np.ndarray(float)
            The initialized PSV-Grid.
            Array of shape=
            :attr:`~boltzmann.configuration.Configuration.p`.G.shape
            +
            :attr:`~boltzmann.configuration.Configuration.sv`.G.shape
            and dtype=float.
        """
        self.check_integrity()
        shape = (self.config.p.G.shape[0], self.config.sv.G.shape[0])
        # Todo Find nicer way to iterate over whole P-Space
        p_flat = self.p_flag.flatten()
        assert p_flat.shape == (shape[0],)
        psv = np.zeros(shape=shape, dtype=float)
        # set Velocity Grids for all specimen
        for i_p in range(p_flat.size):
            # get active rule
            r = self.rules[p_flat[i_p]]
            for i_s in range(self.config.s.n):
                rho = r.rho[i_s]
                temp = r.temp[i_s]
                begin = self.config.sv.index[i_s]
                end = self.config.sv.index[i_s+1]
                v_grid = self.config.sv.G[begin:end]
                for (i_v, v) in enumerate(v_grid[:, :]):
                    # Todo np.array(v) only for PyCharm Warning - Check out
                    diff_v = np.sum((np.array(v) - r.drift[i_s])**2)
                    psv[i_p, begin + i_v] = rho * math.exp(-0.5*(diff_v/temp))
                # Adjust initialized values, to match configurations
                # Todo read into Rjasanovs script and do this correctly
                # Todo THIS IS CURRENTLY WRONG! ONLY TEMPORARY FIX
                adj = psv[i_p, begin:end].sum()
                psv[i_p, begin:end] *= rho/adj
        return psv

    def check_integrity(self):
        assert self.rules.dtype == b_rul.Rule
        for (i_r, r) in enumerate(self.rules):
            r.check_integrity()
            assert i_r >= self.block_index[r.cat]
            assert i_r < self.block_index[r.cat+1]
        assert self.rules.shape == (self.block_index[-1],)
        length = len(Initialization.SPECIFIED_CATEGORIES)
        # A change of this length needs several changes in this module
        # and its submodules
        assert len(Initialization.SPECIFIED_CATEGORIES) is 4
        assert self.block_index.shape == (length+1,)
        assert self.p_flag.size == self.config.p.G.shape[0]
        assert self.p_flag.dtype == int
        assert np.min(self.p_flag) >= 0, 'Uninitialized P-Grid points'
        assert np.max(self.p_flag) <= self.rules.size - 1, 'Undefined Rule'
        return

    def print(self,
              physical_grid=False):
        print('=======INITIALIZATION=======')
        print('Number of Rules = '
              '{}'.format(self.rules.shape[0]))
        print('Block Indices = '
              '{}'.format(self.block_index))

        for i_c in range(4):
            print('Rules: {}'
                  ''.format(Initialization.SPECIFIED_CATEGORIES[i_c]+'s'))
            str_len = len(Initialization.SPECIFIED_CATEGORIES[i_c]) + 7
            print('-'*str_len)
            for r in self.rules[self.block_index[i_c]:
                                self.block_index[i_c+1]]:
                r.print(Initialization.SPECIFIED_CATEGORIES)
        if physical_grid:
            print('Flag-Grid of P-Space:')
            print(self.p_flag)
        return
