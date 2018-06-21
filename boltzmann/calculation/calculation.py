
from . import collisions as b_col
from . import output_function as b_opf

import numpy as np

from time import time


class Calculation:
    """Manages calculation process

    For equivalent, straightforward, but less performant Code
    see the CalculationTest Class!

    ..todo::
        - enable writing of complete results (not moments) to sim file, for unittests
        - decide on t_arr:
            * is it an integer, or array(int)
            * for higher order transport -> multiple entries?
        - Properly implement calculation( switches for Orders, vectorized))
        - Implement Operator Splitting of Order 2 (easy)
        - directly use p_flag? shrink it down
          (only 1 flag(==0) for inner points necessary)?
          Use p_flag as an array of pointers, that point to their
          calculation function (depending on their type)
        - implement complete output, for testing
        - Implement complex Geometries for P-Grid:

            * Each P-Grid Point has a list of Pointers to its Neighbours
            * For each Velocity there is a list of Pointers to these Pointers
              (Which Neighbours to use for Transport-Calculation)

        - Implement adaptive P-Grid:
            * The P-Grid is given as an array of the P-Gris Points - Array_0
            * Each P-Grid Point has several separate lists of Pointers:

              * List_0 contains Pointers to the Neighbours in the P-Grid.
              * List_1,... are empty in the beginning

            * If a finer Grid is necessary (locally), then:

              * Add a Array with the additional points - Array_1
              * The Entries of the additional Points are interpolated
                from the neighbours in the coarse grid
              * Fill List_1 with Pointers to the new,
                finer neighbouring points in Array_1
              * Note that it's necessary to check for some points, if they've
                been added already, by neighbouring points
                (for each Neighbour in List_0, check List_1).
                In that case only add the remaining points to Array_1 and
                let the Pointer in List_1 point to the existing point in
                Array_1
              * In Order to fulfill the boundary conditions it is necessary
                to add "Ghost-Boundary-Points" to the neighbouring points
                of the finer Grid.
                There is a PHD-Thesis about this. -> Find it
              * Those Ghost points are only interpolated
                from their Neighbours
                there is no extra transport & collision step necessary
                for them
              * When moving to a finer Grid, the time-step is halved
                for both transport and collision on all Grids.
                This is not that bad, as for Collisions only the number of
                sub-collision-steps is halved, by equal weight/time_step
                (as long as this is possible).
                => Only the transport step gets more intense
              * there should be additional p_flags for each point, denoting,
                if its a point with 'sub'points or a neighbouring point
                of a finer area,with 'Ghost-Boundary-Points'
              * The total number of Grid-Refinement-Levels should be bound

    Parameters
    ----------
    cnf : :class:`~boltzmann.configuration.Configuration`
    ini : :class:`~boltzmann.initialization.Initialization`
    """
    def __init__(self,
                 cnf,
                 ini):
        # Visible Properties
        self._cnf = cnf
        self._cols = b_col.Collisions(self._cnf)
        self._data = ini.create_psv_grid()
        self._result = np.copy(self.data)
        self._p_flag = ini.p_flag
        self._f_out = b_opf.OutputFunction(cnf)
        self._t_cur = cnf.t.G[0]
        self._cal_time = time()     # to estimate remaining time
        # t_arr: np.ndarray(int)
        # t_arr is used in the ** transport step **.
        # Each t_arr[i_v, _] denotes an index difference in P - Space,
        # such that data[p + t_arr[i_v, _], i_v] is used
        # for the calculation of result[p, i_v].
        # Todo self.t_arr = np.zeros((0,), dtype=int)
        # t_w : np.ndarray(float)
        # t_w[i_t, :] denotes the weight for t_arr[i_t, :]
        # in the transport step
        # Todo self.t_w = np.zeros((0,), dtype=float)
        return

    @property
    def cnf(self):
        """:obj:`~boltzmann.configuration.Configuration`:
        Points at the Configuration"""
        return self._cnf

    @property
    def data(self):
        """:obj:`~numpy.ndarray` of :obj:`float`:
        Current state of the simulation

        At time step :attr:`t_cur`, position p and velocity v:

            data[p, v] = f(:attr:`t_cur`, p, v).

        Array of shape
        (:attr:`cnf`.
        :attr:`~boltzmann.configuration.Configuration.p`.
        :attr:`~boltzmann.configuration.Grid.size`,
        :attr:`cnf`.
        :attr:`~boltzmann.configuration.Configuration.sv`.
        :attr:`~boltzmann.configuration.SVGrid.index` [-1]).
        """
        return self._data

    @property
    def p_flag(self):
        """:obj:`~numpy.ndarray` of :obj:`int`:
        Currently p_flag does nothing.

        In the future:

        For each P-:class:`~boltzmann.configuration.Grid` point :obj:`p`,
        :attr:`p_flag` [:obj:`p`] describes its category
        (see
        :attr:`~boltzmann.initialization.Initialization.supported_categories`).

        :attr:`p_flag` controls the behaviour of each
        P-:class:`~boltzmann.configuration.Grid` point
        during :class:`Calculation`.
        For each different value in :attr:`p_flag`
        a custom function is generated.
        """
        return self._p_flag

    @property
    def f_out(self):
        """:obj:`~boltzmann.calculation.OutputFunction`:
        Handles generation and saving of interim results
        """
        return self._f_out

    @property
    def t_cur(self):
        """:obj:`int`:
        The current time step.
        """
        return self._t_cur

    #####################################
    #            Calculation            #
    #####################################
    def run(self):
        """Starts the Calculation and writes the interim results
        to the disk
        """
        assert self.check_stability_conditions()
        self._cal_time = time()
        print('Calculating...          ',
              end='\r')

        for (i_w, t_w) in enumerate(self._cnf.t.G):
            while self.t_cur != t_w:
                self._calculate_time_step()
            # generate Output and write it to disk
            self.f_out.apply(self)

        print("Calculating...Done                      \n"
              "Time taken =  {} seconds"
              "".format(round(time() - self._cal_time, 3)))
        return

    def _calculate_time_step(self):
        """Executes a single time step
        and prints an estimate of the remaining time to the terminal"""
        # executing time step
        self._calculate_transport_step()
        for _ in range(self._cnf.coll_substeps):
            self._calculate_collision_step()
        self._t_cur += 1
        # executing time step
        self._print_time_estimate()
        return

    def _print_time_estimate(self):
        """Prints an estimate of the remaining time to the terminal"""
        remaining_steps = self._cnf.t.G[-1] - self.t_cur
        est_step_duration = (time() - self._cal_time) / self.t_cur
        estimated_time = round(remaining_steps * est_step_duration, 1)
        print('Calculating...{}'
              ''.format(estimated_time),
              end='\r')
        return

    def _calculate_collision_step(self):
        """Executes a single collision step on complete P-Grid"""
        for p in range(self._cnf.p.size):
            u_c0 = self._data[p, self._cols.collision_arr[:, 0]]
            u_c1 = self._data[p, self._cols.collision_arr[:, 1]]
            u_c2 = self._data[p, self._cols.collision_arr[:, 2]]
            u_c3 = self._data[p, self._cols.collision_arr[:, 3]]
            col_factor = (np.multiply(u_c0, u_c2) - np.multiply(u_c1, u_c3))
            self._data[p] += self._cols.mat.dot(col_factor)
        return

    def _calculate_transport_step(self):
        """Executes single collision step on complete P-Grid"""
        if self._cnf.p.dim != 1:
            message = 'Transport is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        dt = self._cnf.t.d
        dp = self._cnf.p.d
        for s in range(self._cnf.s.n):
            [beg, end] = self._cnf.sv.range_of_indices(s)
            dv = self._cnf.sv.vGrids[s].d
            # Todo removal of boundaries only temporary, until rules for input/output points or boundary points are set
            for p in range(1, self._cnf.p.size-1):
                for v in range(beg, end):
                    pv = dv * self._cnf.sv.SVG[v]
                    if pv[0] <= 0:
                        new_val = ((1 + pv[0]*dt/dp) * self.data[p, v]
                                   - pv[0]*dt/dp * self.data[p+1, v])
                    elif pv[0] > 0:
                        new_val = ((1 - pv[0]*dt/dp) * self.data[p, v]
                                   + pv[0]*dt/dp * self.data[p-1, v])
                    else:
                        continue
                    self._result[p, v] = new_val
        self._data[...] = self._result[...]
        return

    def check_stability_conditions(self):
        """Checks Courant-Friedrichs-Levy Condition

        Returns
        -------
        bool
            True, if all conditions are satisfied.
            False, otherwise."""
        # check Courant-Friedrichs-Levy-Condition
        max_v = np.linalg.norm(self._cnf.sv.boundaries, axis=1).max()
        dt = self._cnf.t.d
        dp = self._cnf.p.d
        cfl_condition = max_v * (dt/dp) < 1/2
        return cfl_condition
