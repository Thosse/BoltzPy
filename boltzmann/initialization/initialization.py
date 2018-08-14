
from boltzmann import simulation as b_sim
from boltzmann.initialization import rule as b_rul

import numpy as np
import os
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
        - Add former block_index functionality for boundary points again
            * sort rule_arr and init_arr
            * set up reflection methods -> depends on position
                -> multiplies number of boundary rules
            * move into initialization module

    Parameters
    ----------
    simulation : :class:`~boltzmann.Simulation`
    file_address : :obj:`str`, optional
        Can be either a full path, a base file name or a file root.

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
    def __init__(self, simulation, file_address=None):
        # Todo simulation.check_integrity(complete_check=False)
        assert isinstance(simulation, b_sim.Simulation)
        self._sim = simulation

        # Assert write access rights and that file exists
        if file_address is None:
            file_address = self._sim.file_address
        else:
            assert os.path.exists(file_address), \
                "File does not exist: {}".format(file_address)
            assert os.access(file_address, os.W_OK), \
                "No write access to {}".format(file_address)

        # Open HDF5 file
        # Todo Assert it is a simulation file!
        if os.path.exists(file_address):
            file = h5py.File(file_address, mode='r')
        else:
            file = h5py.File(file_address, mode='w-')

        ######################
        #   Initialization   #
        ######################
        # load initialization rules
        try:
            self.rule_arr = np.empty(shape=(0,), dtype=b_rul.Rule)
            n_rules = int(file["Initialization"].attrs["Number of Rules"])
            for rule_idx in range(n_rules):
                key = "Initialization/Rule_{}".format(rule_idx)
                # Todo make Rule load static!
                rule = b_rul.Rule()
                rule.load(file[key])
                self.rule_arr = np.append(self.rule_arr, [rule])
        except KeyError:
            self.rule_arr = np.empty(shape=(0,), dtype=b_rul.Rule)

        # load initialization array
        # default value -1 <=> no initialization rule applies to the point
        try:
            key = "Initialization/Initialization Array"
            self.init_arr = file[key].value
        except KeyError:
            self.init_arr = np.full(shape=self._sim.configuration.p.size,
                                    fill_value=-1,
                                    dtype=int)
        self.check_integrity()
        return

    @property
    def n_rules(self):
        """:obj:`int` :
        Total number of :class:`Rules <Rule>` set up so far.
        """
        return self.rule_arr.size

    #####################################
    #           Configuration           #
    #####################################
    # Todo Write add and edit methods? For easier GUI?
    def add_rule(self,
                 category,
                 rho,
                 drift,
                 temp,
                 name=None,
                 color=None):
        """Add a new :class:`initialization rule <Rule>`
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
        assert np.array(rho).shape == (self._sim.configuration.s.n,)
        assert np.array(drift).shape == (self._sim.configuration.s.n,
                                         self._sim.configuration.sv.dim)
        assert np.array(temp).shape == (self._sim.configuration.s.n,)

        new_rule = b_rul.Rule(category,
                              rho,
                              drift,
                              temp,
                              name,
                              color)
        self.rule_arr = np.append(self.rule_arr, [new_rule])
        return

    def apply_rule(self,
                   array_of_grid_point_indices,
                   rule_index):
        """Mark :class:`P-Grid <boltzmann.configuration.Grid>` points
        to be initialized with the specified
        :class:`initialization rule <Rule>`.

        Sets the :attr:`init_arr` entries
        of all  :class:`P-Grid <boltzmann.configuration.Grid>` points
        in *array_of_grid_point_indices* to *rule_index*.

        Parameters
        ----------
        array_of_grid_point_indices : :obj:`~numpy.ndarray` [:obj:`int`]
            Contains flat indices of
            :class:`P-Grid <boltzmann.configuration.Grid>` points.
        rule_index : :obj:`int`
            Index of a :class:`initialization rule <Rule>`
            in :attr:`rule_arr`.
        """
        assert isinstance(array_of_grid_point_indices, np.ndarray)
        assert array_of_grid_point_indices.dtype == int
        assert np.min(array_of_grid_point_indices) >= 0
        assert (np.max(array_of_grid_point_indices)
                < self._sim.configuration.p.size)
        assert isinstance(rule_index, int)
        assert 0 <= rule_index < self.n_rules

        for p in array_of_grid_point_indices:
            self.init_arr[p] = rule_index
        return

    # Todo move into initialization module
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
        assert self.init_arr.size == self._sim.configuration.p.size
        psv = np.zeros(shape=(self._sim.configuration.p.size,
                              self._sim.configuration.sv.size),
                       dtype=float)
        # set Velocity Grids for all specimen
        for (i_p, i_rule) in enumerate(self.init_arr):
            # get active rule
            r = self.rule_arr[i_rule]
            # Todo - simply call r_apply method?
            for i_s in range(self._sim.configuration.s.n):
                rho = r.rho[i_s]
                temp = r.temp[i_s]
                [begin, end] = self._sim.configuration.sv.range_of_indices(i_s)
                v_grid = self._sim.configuration.sv.iMG[begin:end]
                dv = self._sim.configuration.sv.vGrids[i_s].d
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
    #           Serialization           #
    #####################################
    def save(self, file_address=None):
        """Writes all parameters of the :class:`Initialization` instance
        to the given HDF5-file.

        Parameters
        ----------
        file_address : str, optional
            Full path to a :class:`~boltzmann.Simulation` HDF5-file.
        """
        self.check_integrity()
        if file_address is None:
            file_address = self._sim.file_address
        else:
            assert os.path.exists(file_address), \
                "File does not exist: {}".format(file_address)
            assert os.access(file_address, os.W_OK), \
                "No write access to {}".format(file_address)

        # Open file
        file = h5py.File(file_address, mode='a')

        # Clear currently saved Initialization, if any
        if "Initialization" in file.keys():
            del file["Initialization"]

        # Create and open empty "Initialization" group
        file.create_group("Initialization")
        file_i = file["Initialization"]

        # Save all Rules
        file_i.attrs["Number of Rules"] = self.n_rules
        for (i_r, rule) in enumerate(self.rule_arr):
            file_i.create_group("Rule_{}".format(i_r))
            hdf5_subgroup = file_i["Rule_{}".format(i_r)]
            rule.save(hdf5_subgroup)

        # Save Initialization Array
        file_i["Initialization Array"] = self.init_arr
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check"""
        assert isinstance(self.rule_arr, np.ndarray)
        assert self.rule_arr.ndim == 1
        for (i_r, rule) in enumerate(self.rule_arr):
            assert isinstance(rule, b_rul.Rule)
            rule.check_integrity()
            # assert right shape of conserved quantities
            assert rule.rho.shape == (self._sim.configuration.s.n,)
            assert rule.drift.shape == (self._sim.configuration.s.n,
                                        self._sim.configuration.sv.dim)
            assert rule.temp.shape == (self._sim.configuration.s.n,)

        assert isinstance(self.init_arr, np.ndarray)
        # Todo Check this in unit tests, compare before and after save + load
        if self._sim.configuration.p.size is not None:
            assert self.init_arr.size == self._sim.configuration.p.size
        else:
            assert self.init_arr.size == 1
        assert self.init_arr.dtype == int
        assert np.min(self.init_arr) >= -1
        # Todo Check this, before Calculation -> complete check
        # assert np.min(self.init_arr) >= 0, \
        #     'Positional Grid is not properly initialized.' \
        #     'Some Grid points have no initialization rule!'
        assert np.max(self.init_arr) < self.n_rules, \
            'Undefined Rule! A P-Grid point is set ' \
            'to be initialized by an undefined initialization rule. ' \
            'Either add the respective rule, or choose an existing rule.'
        return

    # Todo change into __str__ method
    def print(self, physical_grid=False):
        """Prints all Properties for Debugging Purposes

        If physical_grid is True, then :attr:`init_arr` is printed."""
        print('\n=======INITIALIZATION=======\n')
        print('Number of Rules = '
              '{}'.format(self.rule_arr.shape[0]))

        print('\nInitialization Rules')
        print('-' * 20)
        for rule in self.rule_arr:
            rule.print()
        if physical_grid:
            print('Flag-Grid of P-Space:')
            print(self.init_arr)
        return
