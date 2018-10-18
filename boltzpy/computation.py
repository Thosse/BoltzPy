
import boltzpy as b_sim
import boltzpy.collisions.collision_relations as b_rel
import boltzpy.initialization as b_ini
import boltzpy.output as b_out

import numpy as np
from time import time


class Calculation:
    r"""Manages computation process

    For equivalent, straightforward, but less performant Code
    see the CalculationTest Class!

    ..todo::
        - enable writing of complete results (not moments) to sim file,
          for unittests
        - decide on t_arr:
            * is it an integer, or array(int)
            * for higher order transport -> multiple entries?
            * replace t_arr, and t_w by sparse matrix,
              such that transport is simple multiplication?
        - Properly implement computation( switches for Orders, vectorized))
        - Implement Operator Splitting of Order 2 (easy)
        - directly use p_flag? shrink it down
          (only 1 flag(==0) for inner points necessary)?
          Use p_flag as an array of pointers, that point to their
          computation function (depending on their type)
        - Implement complex Geometries for P-Grid:

            * Each P-Grid Point has a list of Pointers to its Neighbours
              (8 in 2D, 26 in 3D)
            * For each Velocity there is a list of Pointers
              to these Pointers
              (Which Neighbours to use for Transport-Calculation)

        - Implement adaptive P-Grid:
            * The P-Grid is given as an array of the P-Gris Points
              -> Array_0
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
    simulation : :class:`~boltzpy.Simulation`

    Attributes
    ----------
    data : :obj:`~np.ndarray` [:obj:`float`]
        Current state of the simulation, such that
        :math:`f(t_{cur}, p, v) = data[i_p, i_v]`.
        Array of shape
        (:attr:`cnf.p.size <boltzpy.Grid.size>`,
        :attr:`cnf.sv.size <boltzpy.SVGrid.size>`).
    f_out : :class:`~boltzpy.output.OutputFunction`
        Handles generation and saving of interim results
    t_cur : :obj:`int`
        The current time step / index.
        If t_cur is in :attr:`cnf.t.iG <boltzpy.Grid>`
        The current data is written to the sim file.
    """
    def __init__(self, simulation):
        assert isinstance(simulation, b_sim.Simulation)
        # Todo simulation.check_integrity(complete_check=False)
        self._sim = simulation
        self._cols = b_rel.CollisionRelations(self._sim)

        # Todo possibly better to do with np.full
        if (self._sim.p.size is None
                or self._sim.sv.size is None):
            self.data = None
            # Todo _result might be unnecessary
            self._result = None
        else:
            self.data = np.empty(shape=(self._sim.p.size,
                                        self._sim.sv.size),
                                 dtype=float)
            self._result = np.copy(self.data)

        # Todo _p_flag might be unnecessary,
        # Todo only necessary to set up transport step
        # Todo possibly useful to decide if to do collision step in position
        # self._p_flag = ini.p_flag
        self.f_out = b_out.OutputFunction(self._sim)
        if self._sim.t.iG is None:
            self.t_cur = None
        else:
            self.t_cur = self._sim.t.iG[0, 0]
        self._cal_time = time()     # to estimate remaining time
        # t_arr: np.ndarray(int)
        # t_arr is used in the ** transport step **.
        # Each t_arr[i_v, _] denotes an index difference in P - Space,
        # such that data[p + t_arr[i_v, _], i_v] is used
        # for the computation of result[p, i_v].
        # Todo self.t_arr = np.zeros((0,), dtype=int)
        # t_w : np.ndarray(float)
        # t_w[i_t, :] denotes the weight for t_arr[i_t, :]
        # in the transport step
        # Todo self.t_w = np.zeros((0,), dtype=float)
        return

    # @property
    # def p_flag(self):
    #     """:obj:`~numpy.ndarray` of :obj:`int`:
    #     Currently p_flag does nothing.
    #
    #     In the future:
    #
    #     For each P-:class:`~boltzpy.Grid` point :obj:`p`,
    #     :attr:`p_flag` [:obj:`p`] describes its category
    #     (see
    #     :attr:`~boltzpy.initialization.Initialization.supported_categories`).
    #
    #     :attr:`p_flag` controls the behaviour of each
    #     P-:class:`~boltzpy.Grid` point
    #     during :class:`Calculation`.
    #     For each different value in :attr:`p_flag`
    #     a custom function is generated.
    #     """
    #     return self._p_flag

    #####################################
    #            Calculation            #
    #####################################
    def run(self, hdf_group_name):
        """Starts the Calculation and writes the interim results
        to the disk
        """
        # Todo Add check_integrity / stability conditions?
        assert self.check_stability_conditions()
        # Todo Move this part into setup block?
        # Initialize PSV-Grids
        self.data = b_ini.create_psv_grid(self._sim)
        self._result = np.copy(self.data)

        # Generate collision relations
        # Todo coll_select_scheme = "free_flow"
        if self._sim.coll_substeps != 0:
            self._cols.setup()

        # Prepare Output functions
        self.f_out.output_function(hdf_group_name=hdf_group_name)

        # set start time
        self.t_cur = self._sim.t.iG[0, 0]

        self._cal_time = time()
        print('Calculating...          ',
              end='\r')

        for (i_w, t_w) in enumerate(self._sim.t.iG[:, 0]):
            while self.t_cur != t_w:
                self._calculate_time_step()
                self._print_time_estimate()
            # generate Output and write it to disk
            self.f_out.apply(self)

        time_taken_in_seconds = int(time() - self._cal_time)
        # large number of spaces necessary to overwrite the old terminal lines
        print('Calculating... Done' + 40*' ' + '\n'
              'Time taken = ' + self._format_time(time_taken_in_seconds))
        return

    def _calculate_time_step(self):
        """Executes a single time step"""
        # executing time step
        self._calculate_transport_step()
        for _ in range(self._sim.coll_substeps):
            self._calculate_collision_step()
        assert np.all(self.data > 0)
        self.t_cur += 1
        return

    def _print_time_estimate(self):
        """Prints an estimate of the remaining time to the terminal"""
        rem_steps = self._sim.t.iG[-1, 0] - self.t_cur
        est_step_duration = (time() - self._cal_time) / self.t_cur
        est_time_in_seconds = int(rem_steps * est_step_duration)
        print('Calculating... '
              + self._format_time(est_time_in_seconds)
              + ' remaining',
              end='\r')
        return

    @staticmethod
    def _format_time(t_in_seconds):
        (days, t_in_seconds) = divmod(t_in_seconds, 86400)
        (hours, t_in_seconds) = divmod(t_in_seconds, 3600)
        (minutes, t_in_seconds) = divmod(t_in_seconds, 60)
        t_string = '{:2d}d {:2d}h {:2d}m {:2d}s'.format(days,
                                                        hours,
                                                        minutes,
                                                        t_in_seconds)
        return t_string

    def _calculate_collision_step(self):
        """Executes a single collision step on complete P-Grid"""
        for p in range(self._sim.p.size):
            u_c0 = self.data[p, self._cols.collision_arr[:, 0]]
            u_c1 = self.data[p, self._cols.collision_arr[:, 1]]
            u_c2 = self.data[p, self._cols.collision_arr[:, 2]]
            u_c3 = self.data[p, self._cols.collision_arr[:, 3]]
            col_factor = (np.multiply(u_c0, u_c2) - np.multiply(u_c1, u_c3))
            self.data[p] += self._cols.mat.dot(col_factor)
        return

    def _calculate_transport_step(self):
        """Executes single collision step on complete P-Grid"""
        if self._sim.p.dim != 1:
            message = 'Transport is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)

        dt = self._sim.t.d
        dp = self._sim.p.d
        offset = self._sim.sv.offset
        for s in range(self._sim.s.size):
            [beg, end] = self._sim.sv.idx_range(s)
            dv = self._sim.sv.vGrids[s].d
            # Todo removal of boundaries (p in range(1, ... -1))
            # Todo is only temporary,
            # Todo until rules for input/output points
            # Todo or boundary points are set
            for p in range(1, self._sim.p.size-1):
                for v in range(beg, end):
                    pv = dv * self._sim.sv.iMG[v] + offset
                    if pv[0] <= 0:
                        new_val = ((1 + pv[0]*dt/dp) * self.data[p, v]
                                   - pv[0]*dt/dp * self.data[p+1, v])
                    elif pv[0] > 0:
                        new_val = ((1 - pv[0]*dt/dp) * self.data[p, v]
                                   + pv[0]*dt/dp * self.data[p-1, v])
                    else:
                        continue
                    self._result[p, v] = new_val
        self.data[...] = self._result[...]
        return

    def check_stability_conditions(self):
        """Checks Courant-Friedrichs-Levy Condition

        Returns
        -------
        bool
            True, if all conditions are satisfied.
            False, otherwise."""
        # check Courant-Friedrichs-Levy-Condition
        max_v = np.linalg.norm(self._sim.sv.boundaries,
                               axis=1).max()
        dt = self._sim.t.d
        dp = self._sim.p.d
        cfl_condition = max_v * (dt/dp) < 1/2
        return cfl_condition
