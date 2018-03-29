
from boltzmann.configuration import configuration as b_cnf
from boltzmann.initialization import rule as b_rul

import numpy as np
import math


class Initialization:
    """Handles initialization instructions and creates
    PSV-Grids and :attr:`p_flag`.

    * Collects initialization Rules in :attr:`rule_arr`.
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
    cnf : :class:`~boltzmann.configuration.Configuration`

    Notes
    -----
    Each P-Grid point fits
    into exactly one of the following Categories:

        * **Inner Point (Default)**:

          * both transport and collisions are applied normally

        * **Boundary Point**:

          * no collision step
          * additional reflection step after every transport step

        * **Ghost Boundary Point**:

          * for higher order transport
          * so far undetermined behaviour

        * **Constant Input/Output Point**:

          * no collision-step
          * no transport-step
          * Distribution is constant over the whole simulation.

        * **Time Variant Input/Output Point**:

          * no collision-step,
          * no transport-step,
          * Distribution is freshly initialized in every time step

    """
    def __init__(self,
                 cnf=b_cnf.Configuration()):
        self._cnf = cnf
        self._rule_arr = np.empty(shape=(0,), dtype=b_rul.Rule)
        n_categories = len(self.supported_categories)
        self._block_index = np.zeros(shape=(n_categories + 1,),
                                     dtype=int)
        p_shape = tuple(self.cnf.p.n)
        self._p_flag = np.full(shape=p_shape,
                               fill_value=-1,
                               dtype=int)
        return

    @property
    def cnf(self):
        """:obj:`~boltzmann.configuration.Configuration`:
        Points to the :obj:`~boltzmann.configuration.Configuration`."""
        return self._cnf

    @property
    def supported_categories(self):
        """List of all supported Categories of
        P-:class:`~boltzmann.configuration.Grid` points.
        Each Category behaves differently in
        :class:`~boltzmann.calculation.Calculation`."""
        supported_categories = ['Inner Point',
                                # 'Boundary Point',
                                # 'Ghost Boundary_Point',
                                # 'Constant_IO_Point',
                                # 'Time_Variant_IO_Point',
                                ]
        return supported_categories

    @property
    def block_index(self):
        """:obj:`~numpy.ndarray` of :obj:`int`:
        Marks the range of each Category block in :attr:`rule_arr`.

        For each Category c,
        :attr:`rule_arr`
        [:attr:`block_index` [c] : :attr:`block_index` [c+1]]
        denotes the range of Category c Rules (see :class:`Rule`)
        in :attr:`rule_arr`.
        Note that the last entry marks the total length of
        :attr:`rule_arr`."""
        return self._block_index

    @property
    def p_flag(self):
        """:obj:`~numpy.ndarray` of :obj:`int`:
        Controls the behavior of each
        P-:class:`~boltzmann.configuration.Grid` point
        during
        :class:`~boltzmann.calculation.Calculation`
        and its initial values.

        For each P-:class:`~boltzmann.configuration.Grid` point p,
        :attr:`p_flag` [p] is the index of its
        Initialization :class:`Rule` in :attr:`rule_arr`.
        """
        return self._p_flag

    @property
    def rule_arr(self):
        """:obj:`~numpy.ndarray` of :obj:`Rule` :
        Array of all specified construction Rules (see :class:`Rule` ).
        Sorted by Rule Category (see :attr:`supported_categories`).
        """
        return self._rule_arr

    #####################################
    #           Configuration           #
    #####################################
    def add_rule(self,
                 category,
                 rho_list,
                 drift_list,
                 temp_list,
                 TMP_OPTIONAL_STUFF=None,
                 name=''):
        """Adds a new initialization
        :class:`Rule` to  :attr:`rule_arr`.

        The added :class:`Rule`
        initializes the velocity space of each specimen
        (see :obj:`~boltzmann.configuration.Species`,
        :obj:`~boltzmann.configuration.SVGrid`)
        based on the conserved quantities

            * Mass (:attr:`Rule.rho`),
            * Mean Velocity (:attr:`Rule.drift`)
            * Temperature (:attr:`Rule.temp`)

        Parameters
        ----------
        category : :obj:`str`
            Specifies the Category
            of the new :obj:`Rule`.
            Must be an element of
            :attr:`supported_categories`.
        rho_list : :obj:`array` or :obj:`list`
            List of the parameter rho, for each specimen.
            See :attr:`Rule.rho`.
        drift_list : :obj:`array` or :obj:`list`
            List of the parameter drift, for each specimen.
            See :attr:`Rule.drift`.
        temp_list : :obj:`array` or :obj:`list`
            List of the parameter temp, for each specimen.
            See :attr:`Rule.temp`.
        name : :obj:`str`, optional
            Sets a name to this rule and the
            P-:class:`~boltzmann.configuration.Grid` points
            on which it's applied.
        """
        assert category in self.supported_categories
        # Construct new_rule -> Depending on Category
        if category == 'Inner Point':
            category = 0
            new_rule = b_rul.Rule(category,
                                  rho_list,
                                  drift_list,
                                  temp_list,
                                  name)
        else:
            print('Unspecified Category: {}'
                  ''.format(category))
            # Todo throw exception
            assert False

        pos_of_rule = self.block_index[category+1]
        # Insert new rule into rule array
        self._rule_arr = np.insert(self.rule_arr,
                                   pos_of_rule,
                                   [new_rule])
        # Adjust p_flag entries to new rule_arr
        for _val in np.nditer(self._p_flag,
                              op_flags=['readwrite']):
            if _val >= pos_of_rule:
                _val += 1
        # Adjust block_index to new rule_arr
        for _val in np.nditer(self._block_index[1:],
                              op_flags=['readwrite']):
            if _val >= pos_of_rule:
                _val += 1
        return

    def apply_rule(self,
                   index_rule,
                   p_min,
                   p_max):
        """Applies an initialization :obj:`Rule` from
        :attr:`rule_arr`
        to the specified points of the
        P-:class:`~boltzmann.configuration.Grid`.

        Sets the :attr:`p_flag` entry
        of the specified
        P-:class:`~boltzmann.configuration.Grid` points to index_rule.
        Both their initial values and behaviour during
        :class:`~boltzmann.calculation.Calculation`
        is defined by that :obj:`Rule`.

        Note that the :obj:`Rule` is applied to all
        P-:class:`~boltzmann.configuration.Grid` points p,
        such that
        all(p_min <= p < p_max).
        Indices are in vector form (not flattened).

        Parameters
        ----------
        index_rule : :obj:`int`
            Index of the to be applied :obj:`Rule` in
            :attr:`rule_arr`.
        p_min, p_max :  array_like of :obj:`int`
            Boundary points, given as multiples of positional step size
            (see :attr:`boltzmann.configuration.Grid.G`).
            Delimit/mark the area where to apply the rule.
        """
        assert 0 <= index_rule < self.block_index[-1]
        dim = self.cnf.p.dim
        p_min = np.array(p_min)
        p_max = np.array(p_max)
        assert p_min.shape == (dim,) and p_max.shape == (dim,)
        assert p_min.dtype == int and p_max.dtype == int
        assert all(np.zeros(p_min.shape) <= p_min)
        assert all(p_min <= p_max)
        assert all(p_max <= self.cnf.p.n)

        if dim is 1:
            self._p_flag[p_min[0]:p_max[0]] = index_rule
        elif dim is 2:
            self._p_flag[p_min[0]:p_max[0],
                         p_min[1]:p_max[1]] = index_rule
        elif dim is 3:
            self._p_flag[p_min[0]:p_max[0],
                         p_min[1]:p_max[1],
                         p_min[2]:p_max[2]] = index_rule
        return

    def create_psv_grid(self):
        """Generates and returns an initialized PSV-Grid
        (:attr:`~boltzmann.calculation.Calculation.data`,
        :attr:`~boltzmann.calculation.Calculation.result`).

        Checks the entry of :attr:`p_flag` and gets the
        initialization :obj:`Rule` for every P-Grid point p.
        Initializes the Distribution
        in the Velocity-Space of p as defined in the :obj:`Rule`.

        Returns
        -------
        psv : :class:`~numpy.ndarray` of :obj:`float`
            The initialized PSV-Grid.
            Array of shape
            (:attr:`~boltzmann.configuration.Configuration.p`.size,
            :attr:`~boltzmann.configuration.Configuration.sv`.size).
        """
        self.check_integrity()
        shape = (self.cnf.p.G.shape[0], self.cnf.sv.G.shape[0])
        # Todo Find nicer way to iterate over whole P-Space
        p_flat = self.p_flag.flatten()
        assert p_flat.size == self.cnf.p.size
        psv = np.zeros(shape=shape, dtype=float)
        # set Velocity Grids for all specimen
        for i_p in range(p_flat.size):
            # get active rule
            r = self.rule_arr[p_flat[i_p]]
            # Todo - simply call r_apply method?
            for i_s in range(self.cnf.s.n):
                rho = r.rho[i_s]
                temp = r.temp[i_s]
                begin = self.cnf.sv.index[i_s]
                end = self.cnf.sv.index[i_s+1]
                v_grid = self.cnf.sv.G[begin:end]
                dv = self.cnf.sv.d[i_s]
                for (i_v, v) in enumerate(v_grid):
                    # Physical Velocity
                    pv = dv * v
                    # Todo np.array(v) only for PyCharm Warning - Check out
                    diff_v = np.sum((np.array(pv) - r.drift[i_s])**2)
                    psv[i_p, begin + i_v] = rho * math.exp(-0.5*(diff_v/temp))
                # Todo read into Rjasanov's script and do this correctly
                # Todo THIS IS CURRENTLY WRONG! ONLY TEMPORARY FIX
                # Adjust initialized values, to match configurations
                adj = psv[i_p, begin:end].sum()
                psv[i_p, begin:end] *= rho/adj
        return psv

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check"""
        assert self.rule_arr.dtype == b_rul.Rule
        for (i_r, r) in enumerate(self.rule_arr):
            r.check_integrity()
            assert i_r >= self.block_index[r.cat]
            assert i_r < self.block_index[r.cat+1]
        assert self.rule_arr.shape == (self.block_index[-1],)
        length = len(self.supported_categories)
        # A change of this length needs several changes in this module
        # and its submodules
        assert len(self.supported_categories) is 1
        assert self.block_index.shape == (length+1,)
        assert self.p_flag.size == self.cnf.p.G.shape[0]
        assert self.p_flag.dtype == int
        assert np.min(self.p_flag) >= 0, 'Uninitialized P-Grid points'
        assert np.max(self.p_flag) <= self.rule_arr.size - 1, 'Undefined Rule'
        return

    def print(self, physical_grid=False):
        """Prints all Properties for Debugging Purposes

        If physical_grid is True, then :attr:`p_flag` is printed."""
        print('\n=======INITIALIZATION=======\n')
        print('Number of Rules = '
              '{}'.format(self.rule_arr.shape[0]))
        print('Block Indices = '
              '{}'.format(self.block_index))

        for (i_c, c) in enumerate(self.supported_categories):
            if self.block_index[i_c] < self.block_index[i_c+1]:
                print('\n{} Rules'
                      ''.format(self.supported_categories[i_c]))
                line_length = len(self.supported_categories[i_c]) + 6
                print('-'*line_length)
            for r in self.rule_arr[self.block_index[i_c]:
                                   self.block_index[i_c+1]]:
                r.print(self.supported_categories)
        if physical_grid:
            print('Flag-Grid of P-Space:')
            print(self.p_flag)
        return
