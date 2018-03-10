
from boltzmann.configuration import configuration as b_cnf
from boltzmann.initialization import initialization as b_ini
from . import output_function as b_opf

import numpy as np
from time import time


class Calculation:
    """Manages calculation process based on minimal set of porameters.

    This class focuses purely on performance
    and readily sacrifices readability.
    For equivalent, more understandable, but less performant Code
    see the CalculationTest Class!

    ..todo::
        - decide on t_arr:
            * is it an integer, or array(int)
            * for higher order transport -> multiple entries?
        - Properly implement calculation( switches for Orders, vectorized))
        - Definition of t_arr is only temporary
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
              * Fill List_1 with Pointers to the new, finer neighbouring points
                in Array_1
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
              * Those Ghost points are only interpolated from their Neighbours
                there is no extra transport & collision step necessary for them
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

    Attributes
    ----------
    t_cur : int
        Current time step.
    """
    def __init__(self,
                 cnf=b_cnf.Configuration(),
                 ini=b_ini.Initialization()):
        # Visible Properties
        self._data = ini.create_psv_grid()
        self._result = np.copy(self.data)
        self._p_flag = ini.p_flag
        self._f_out = b_opf.OutputFunction(cnf)
        # Private Attributes
        # self._c_arr = cnf.cols.collision_arr
        # self._c_weight = cnf.cols.weight_arr
        # Public Attributes
        self.t_cur = cnf.t.G[0]
        self._t = cnf.t.G
        # Todo replace cnf by subset of necessary attributes?
        self._cnf = cnf
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

        # Todo Remove output, replace by appending current results
        # Todo to a file
        output_shape = (cnf.t.size, cnf.p.size, cnf.s.n)
        self.output_mass = np.zeros(output_shape, dtype=float)

        return

    # Todo Edit docstring (attr links)
    @property
    def data(self):
        """:obj:`~numpy.ndarray` of :obj:`float`:
        State of the simulation.

        At time step t, position p and velocity v:

            data[p, v] = f(t, p, v).

        Both Collisions and Transport steps
        read their data from :attr:`data`
        and write the results to :attr:`_result`.
        Afterwards :attr:`data` and :attr:`_result` are either synchronized
        (data[:] =  _results[:])
        or swapped (data, _results = _results, data).
        Array of shape
        ( :attr:`boltzmann.Configuration.p`.size,
        :attr:`boltzmann.Configuration.sv.index` [-1]).
        """
        return self._data

    @property
    def p_flag(self):
        """:obj:`~numpy.ndarray` of :obj:`int`:
        For each P-:class:`~boltzmann.configuration.Grid` point :obj:`p`,
        :attr:`p_flag` [:obj:`p`] describes the category of :obj:`p`
        (see :attr:`~boltzmann.initialization.Initialization.supported_categories`).

        :attr:`p_flag` controls the behaviour of each
        P-:class:`~boltzmann.configuration.Grid` point
        during :class:`Calculation`.
        For each different value in :attr:`p_flag`
        a custom function is generated.
        """
        return self._p_flag

    # Todo Compare to output_func docs and recheck, docstring
    @property
    def f_out(self):
        """:obj:`~boltzmann.calculation.OutputFunction`:
        Processes :attr:`data`
        in the specified intervals
        (see :attr:`~boltzmann.configuration.Configuration.t`)
        and returns the results.
        In the future it directly stores the results on HDD.
        """
        return self._f_out

    def run(self):
        self.check_conditions()
        cal_time = time()
        # Todo revise the function -> write do HDD
        tmp_result = np.zeros(shape=(self._cnf.t.size,
                                     self._cnf.s.n,
                                     self._cnf.p.size),
                              dtype=float)
        for (i_write, t_write) in enumerate(self._t):
            while self.t_cur != t_write:
                self.calc_time_step()
                self.t_cur += 1
            # Todo replace write_mass with  write_results method
            # self.write_results()
            tmp_result[i_write, ...] = self.f_out.apply(self.data)
        print("Calculation - Done\n"
              "Time taken =  {} seconds"
              "".format(round(time() - cal_time, 3)))
        return tmp_result

    def calc_time_step(self):
        self.calc_transport_step()
        for _ in range(self._cnf.n_collision_steps_per_time_step):
            self.calc_collision_step()
        return

    def calc_collision_step(self):
        # Todo Check this again, done in a hurry!
        # Todo removal of boundaries only temporary
        for p in range(1, self._cnf.p.size-1):
            for [i_col, col] in enumerate(self._cnf.cols.collision_arr):
                d_col = (self.data[p, col[0]] * self.data[p, col[2]]
                         - self.data[p, col[1]] * self.data[p, col[3]])
                d_col *= self._cnf.cols.weight_arr[i_col] * self._cnf.t.d
                self._result[p, col[0]] -= d_col
                self._result[p, col[2]] -= d_col
                self._result[p, col[1]] += d_col
                self._result[p, col[3]] += d_col
        self._data = np.copy(self._result)
        return

    def calc_transport_step(self):
        # Todo Check this again, done in a hurry!
        if self._cnf.p.dim is not 1:
            print('So far only 1-D Problems are implemented!')
            assert False
        dt = self._cnf.t.d
        dp = self._cnf.p.d
        for s in range(self._cnf.s.n):
            beg = self._cnf.sv.index[s]
            end = self._cnf.sv.index[s+1]
            dv = self._cnf.sv.d * self._cnf.sv.multi
            # Todo removal of boundaries only temporary
            for p in range(1, self._cnf.p.size-1):
                for v in range(beg, end):
                    pv = dv * self._cnf.sv.G[v]
                    if pv[0] < 0:
                        d_trp = ((1 + pv[0]*dt/dp) * self.data[p, v]
                                 - pv[0]*dt/dp * self.data[p+1, v])
                    elif pv[0] > 0:
                        d_trp = ((1 - pv[0]*dt/dp) * self.data[p, v]
                                 + pv[0]*dt/dp * self.data[p-1, v])
                    else:
                        continue
                    self._result[p, v] = d_trp
        self._data = np.copy(self._result)
        return

    def write_results(self):
        self.f_out.apply(self.data)

    def check_conditions(self):
        # check Courant-Friedrichs-Levy-Condition
        max_v = self._cnf.sv.get_max_v()
        dt = self._cnf.t.d
        dp = self._cnf.p.d
        cfl = max_v * (dt/dp)
