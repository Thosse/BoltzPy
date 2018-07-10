
from boltzmann.configuration import configuration as b_cnf
from boltzmann.initialization import rule as b_rul
import boltzmann.constants as b_const

import numpy as np

import math
import h5py


class Initialization:
    """Handles :class:`initialization instructions<Rule>`, creates
    PSV-Grids and :attr:`init_arr`.

    * Collects :class:`initialization rules <Rule>` in :attr:`rule_arr`.
    * Assigns each :class:`P-Grid <boltzmann.configuration.Grid>` point
      its :class:`initialization rule <Rule>`
      in :attr:`init_arr`.
    * Each :class:`initialization rule <Rule>`
      specifies the initial state and behavior in the simulation.
    * Creates the initialized PSV-Grids
      (:attr:`~boltzmann.calculation.Calculation.data` and
      :attr:`~boltzmann.calculation.Calculation.result`)
      for the :class:`Calculation <boltzmann.calculation.Calculation>`.

    .. todo::
        - Figure out nice way to implement boundary points
        - speed up init of psv grid <- ufunc's
        - @apply_rule: implement different 'shapes' to apply rules
          (e.g. a line with specified width,
          a ball with specified radius/diameter, ..).
          Switch between convex hull and span?
        - sphinx: link PSV-Grid to Calculation.data?
          link init_arr to Calculation.init_arr? No?
          in Initialization-Docstring

    Parameters
    ----------
    configuration : :class:`~boltzmann.configuration.Configuration`, optional

    Attributes
    ----------
    rule_arr : :obj:`~numpy.ndarray` [:class:`Rule`]
        Array of all specified :class:`initialization rules <Rule>` ).
        Rules are sorted by their 
        :const:`category <boltzmann.constants.SUPP_GRID_POINT_CATEGORIES>`.
    init_arr : :obj:`~numpy.ndarray` [:obj:`int`]
        Controls which :class:`Rule` applies to which
        :class:`P-Grid <boltzmann.configuration.Grid>` point

        For each :class:`P-Grid <boltzmann.configuration.Grid>` point p,
        :attr:`init_arr` [p] is the index of its
        :class:`initialization rule <Rule>` in :attr:`rule_arr`.
    """
    def __init__(self,
                 configuration=None):
        if configuration is not None:
            assert isinstance(configuration, b_cnf.Configuration)
            configuration.check_integrity(complete_check=False)
            self._cnf = configuration
        else:
            self._cnf = None

        self.rule_arr = np.empty(shape=(0,), dtype=b_rul.Rule)

        # Todo _block_index is ugly. This should be simpler
        # Todo block_index can probably be removed
        n_categories = len(b_const.SUPP_GRID_POINT_CATEGORIES)
        self._block_index = np.zeros(shape=(n_categories + 1,),
                                     dtype=int)
        # initialize initialization array for P-Grid
        # default value -1 means no initialization rule applies to the point
        self.init_arr = np.full(shape=self._cnf.p.size,
                                fill_value=-1,
                                dtype=int)
        # Todo be very careful, when adding more categories
        assert len(b_const.SUPP_GRID_POINT_CATEGORIES) is 1
        return

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
    def n_rules(self):
        """:obj:`int` :
        Total number of :class:`Rules <Rule>` set up so far.
        """
        return self.rule_arr.size

    #####################################
    #           Configuration           #
    #####################################
    # Todo check here and in Rule that all conserved quantities are np.ndarrays
    # Todo => Don't allow lists
    def add_rule(self,
                 category,
                 rho,
                 drift,
                 temp,
                 name=None,
                 color=None):
        """Adds a new :class:`initialization rules <Rule>`
        to  :attr:`rule_arr`.

        Parameters
        ----------
        category : :obj:`str`
            Category of the :class:`P-Grid <boltzmann.configuration.Grid>`
            point. Must be in
            :const:`~boltzmann.constants.SUPP_GRID_POINT_CATEGORIES`.
        rho : :obj:`~numpy.ndarray` [:obj:`float`]
        drift : :obj:`~numpy.ndarray` [:obj:`float`]
        temp : :obj:`~numpy.ndarray` [:obj:`float`]
        name : str, optional
            Displayed in the GUI to visualize the initialization.
        color : str, optional
            Displayed in the GUI to visualize the initialization.
        """
        # Todo move into check parameters method
        # Assert correct types
        assert isinstance(rho, np.ndarray)
        assert isinstance(drift, np.ndarray)
        assert isinstance(temp, np.ndarray)
        # Assert conserved quantities have correct shape
        assert np.array(rho).shape == (self._cnf.s.n,)
        assert np.array(drift).shape == (self._cnf.s.n, self._cnf.sv.dim)
        assert np.array(temp).shape == (self._cnf.s.n,)
        # Construct new_rule -> Depending on Category
        i_cat = b_const.SUPP_GRID_POINT_CATEGORIES.index(category)
        new_rule = b_rul.Rule(category,
                              rho,
                              drift,
                              temp,
                              name,
                              color)

        # Put new rule in right position
        # rule array is ordered in blocks of rule_categories
        pos_of_rule = self.block_index[i_cat+1]
        # Insert new rule into rule array
        self.rule_arr = np.insert(self.rule_arr,
                                  pos_of_rule,
                                  [new_rule])
        # Adjust init_arr entries to new rule_arr indices
        for (idx, val) in enumerate(self.init_arr):
            if val >= pos_of_rule:
                self.init_arr[idx] += 1
        # Adjust block_index entries to new rule_arr indices
        # Todo this should be more pythonic
        n_categories = len(b_const.SUPP_GRID_POINT_CATEGORIES)
        for idx in range(i_cat+1, n_categories + 1):
            self._block_index[idx] += 1
        return

    def apply_rule(self,
                   array_of_grid_point_indices,
                   rule_index):
        """The specified
        :class:`P-Grid <boltzmann.configuration.Grid>` points
        will be initialized with the specified
        :class:`initialization rule <Rule>`.

        This is done by setting the :attr:`init_arr` entries
        of the  :class:`P-Grid <boltzmann.configuration.Grid>`
        points in *array_of_grid_point_indices* to *rule_index*.

        Parameters
        ----------
        array_of_grid_point_indices : :obj:`~numpy.ndarray` [:obj:`int`]
            Indices (1D) of all :class:`P-Grid <boltzmann.configuration.Grid>`
            points on which the :class:`initialization rule <Rule>`
            should be applied.
        rule_index : :obj:`int`
            Index of the :class:`initialization rule <Rule>`
            in :attr:`rule_arr`.
        """
        assert isinstance(array_of_grid_point_indices, np.ndarray)
        assert array_of_grid_point_indices.dtype == int
        assert np.min(array_of_grid_point_indices) >= 0
        assert np.max(array_of_grid_point_indices) < self._cnf.p.size

        assert isinstance(rule_index, int)
        assert 0 <= rule_index < self.n_rules

        for p in array_of_grid_point_indices:
            self.init_arr[p] = rule_index
        return

    # Todo link PSV Grid to Calculation.data -> explain what psv grid is there
    # Todo Alternatively rename psv-Grid into data_array
    def create_psv_grid(self):
        """Generates and returns an initialized PSV-Grid
        (:attr:`~boltzmann.calculation.Calculation.data`,
        :attr:`~boltzmann.calculation.Calculation.result`).

        Checks the entries of :attr:`init_arr` and uses that
        :class:`initialization rule <Rule>` to initialize the
        :class:`P-Grid <boltzmann.configuration.Grid>` points
        velocity Grid.

        Returns
        -------
        psv : :class:`~numpy.ndarray` [:obj:`float`]
            The initialized PSV-Grid.
            Array of shape
            (:attr:`cnf.p.size
            <boltzmann.configuration.Grid.size>`,
            :attr:`cnf.sv.size
            <boltzmann.configuration.SVGrid.size>`).
        """
        self.check_integrity()
        assert self.init_arr.size == self._cnf.p.size
        psv = np.zeros(shape=(self._cnf.p.size, self._cnf.sv.size),
                       dtype=float)
        # set Velocity Grids for all specimen
        for (i_p, i_rule) in enumerate(self.init_arr):
            # get active rule
            r = self.rule_arr[i_rule]
            # Todo - simply call r_apply method?
            for i_s in range(self._cnf.s.n):
                rho = r.rho[i_s]
                temp = r.temp[i_s]
                [begin, end] = self._cnf.sv.range_of_indices(i_s)
                v_grid = self._cnf.sv.iMG[begin:end]
                dv = self._cnf.sv.vGrids[i_s].d
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
        assert isinstance(self.rule_arr, np.ndarray)
        assert self.rule_arr.ndim == 1
        assert self.n_rules == self.block_index[-1]
        for (i_r, rule) in enumerate(self.rule_arr):
            assert isinstance(rule, b_rul.Rule)
            rule.check_integrity()
            # assert right shape of conserved quantities
            assert rule.rho.shape == (self._cnf.s.n,)
            assert rule.drift.shape == (self._cnf.s.n, self._cnf.sv.dim)
            assert rule.temp.shape == (self._cnf.s.n,)
            # Todo this could be removed if block_index is obsolete
            assert i_r >= self.block_index[rule.i_cat]
            assert i_r < self.block_index[rule.i_cat+1]

        assert self.init_arr.size == self._cnf.p.size
        assert self.init_arr.dtype == int
        assert np.min(self.init_arr) >= 0, \
            'Positional Grid is not properly initialized.' \
            'Some Grid points have no initialization rule!'
        assert np.max(self.init_arr) < self.n_rules, \
            'Undefined Rule! A P-Grid point is set ' \
            'to be initialized by an undefined initialization rule. ' \
            'Either add the respective rule, or choose an existing rule.'

        n_categories = len(b_const.SUPP_GRID_POINT_CATEGORIES)
        assert self.block_index.size == n_categories + 1
        return

    # Todo change into __str__ method
    def print(self, physical_grid=False):
        """Prints all Properties for Debugging Purposes

        If physical_grid is True, then :attr:`init_arr` is printed."""
        print('\n=======INITIALIZATION=======\n')
        print('Number of Rules = '
              '{}'.format(self.rule_arr.shape[0]))
        print('Block Indices = '
              '{}'.format(self.block_index))

        for (i_c, category) in enumerate(b_const.SUPP_GRID_POINT_CATEGORIES):
            if self.block_index[i_c] < self.block_index[i_c+1]:
                print('\n{} Rules'
                      ''.format(category))
                line_length = len(category) + 6
                print('-'*line_length)
            for r in self.rule_arr[self.block_index[i_c]:
                                   self.block_index[i_c+1]]:
                r.print(b_const.SUPP_GRID_POINT_CATEGORIES)
        if physical_grid:
            print('Flag-Grid of P-Space:')
            print(self.init_arr)
        return
