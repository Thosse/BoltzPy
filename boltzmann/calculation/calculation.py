from boltzmann.configuration import configuration as b_cnf
from boltzmann.initialization import initialization as b_ini
from . import output_function as b_opf

import numpy as np
from time import time


class Calculation:
    """
    Manages calculation process based on minimal set of parameters.
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

        - Implement complex Geometries for P-Grid:

            * Each P-Grid Point has a list of Pointers to its Neighbours
            * For each Velocity there is a list of Pointers to these Pointers
              (Which Neighbours to use for Transport-Calculation)

        - Implement adaptive P-Grid:
            * The P-Grid is given as an array of the P-Gris Points - Array_0
            * Each P-Grid Point has several seperate lists of Pointers:

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
              * In Order to fullfill the boundary conditions it is necessary
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
    data, result : np.ndarray
        State of the simulation.
        After time step t, data[i_p, i_v] == f(t, p.G[i_p], sv.G[i_v]).
        Both Collisions and Transport steps
        read their data from *data*
        and write the results to *result*.
        Afterwards data and result are either synchronized
        (data[:] =  results[:])
        or swapped (data, results = results, data).
        Array of shape=(p.n[-1], sv.index[-1])
        and dtype=float.

    p_flag : np.ndarray(int)
        Let i_p be an index of P-Space, then
        p_flag[i_p] describes whether
        i_p is an inner point, boundary point, or input/output point.
        This controls the behaviour of this P-Grid point
        during calculation.
        For each different value in p_flag a custom sub-function is generated.
    c_arr : np.ndarray(int)
        c_arr[i_c, :] is an array of 4 indices of the SVGrid,
        which describe a single collision i_c.
        The ordering is as follows:

            | c_arr[_, 0] = v_pre_collision of Specimen 1
            | c_arr[_, 1] = v_post_collision of Specimen 1
            | c_arr[_, 2] = v_pre_collision of Specimen 2
            | c_arr[_, 3] = v_post_collision of Specimen 2

    c_w : np.ndarray(float)
        c_w[i_c] denotes the weight for c_arr[i_c, :]
        in the collision step.
    t_arr : np.ndarray(int)
        t_arr is used in the **transport step**.
        Each t_arr[i_v, _] denotes an index difference in P-Space,
        such that data[p + t_arr[i_v, _], i_v]
        is used for the calculation of result[p, i_v].
    t_w : np.ndarray(float)
        t_w[i_t, :] denotes the weight for t_arr[i_t, :]
        in the transport step
    f_out : :class:`~boltzmann.calculation.OutputFunction`
        Processes :attr:`~boltzmann.calculation.Calculation.result`
        in the specified intervalls and stores results on HDD.
    cnf : :class:`~boltzmann.configuration.Configuration`
    t_cur : int
        Current time step.
    """
    def __init__(self,
                 cnf=b_cnf.Configuration(),
                 ini=b_ini.Initialization(),
                 moments=list()):
        self.data = ini.create_psv_grid()
        self.result = np.copy(self.data)
        # self.p_flag = ini.p_flag
        # Todo replace cnf by subset of necessary attributes?
        self.cnf = cnf
        # self.c_arr = cnf.cols.i_arr
        # self.c_w = cnf.cols.weight
        # self.t_arr = np.zeros((0,), dtype=int)
        # self.t_w = np.zeros((0,), dtype=float)
        # Todo moments is only temporary an extra parameter
        self.f_out = b_opf.OutputFunction(moments, cnf)
        # Todo Remove write_mass, replace by appending current results
        # Todo to a file
        write_shape = (self.cnf.t.size, self.cnf.p.size, self.cnf.s.n)
        self.write_mass = np.zeros(write_shape, dtype=float)
        self.t_cur = 0
        return

    def run(self):
        self.check_conditions()
        cal_time = time()
        for t_write in range(self.cnf.t.size):
            while self.t_cur != self.cnf.t.G[t_write]:
                self.calc_time_step()
                self.t_cur += 1
            # Todo replace write_mass with  write_results method
            # self.write_results()
            self.write_mass[t_write, ...] = self.f_out.apply(self.data)
        print("Calculation - Done\n"
              "Time taken =  {} seconds"
              "".format(round(time() - cal_time, 3)))
        return

    def calc_time_step(self):
        self.calc_transport_step()
        # Todo Add attribute to store collisons_per_calc_step
        tmp_cols_per_calc = 5
        for _ in range(tmp_cols_per_calc):
            self.calc_collision_step()
        return

    def calc_collision_step(self):
        # Todo Check this again, done in a hurry!
        self.result = np.copy(self.data)
        # Todo removal of boundaries only temporary
        for p in range(1, self.cnf.p.size-1):
            for [i_col, col] in enumerate(self.cnf.cols.i_arr):
                d_col = (self.data[p, col[0]] * self.data[p, col[2]]
                         - self.data[p, col[1]] * self.data[p, col[3]])
                d_col *= self.cnf.cols.weight[i_col] * self.cnf.t.d
                self.result[p, col[0]] -= d_col
                self.result[p, col[2]] -= d_col
                self.result[p, col[1]] += d_col
                self.result[p, col[3]] += d_col
        self.switch_data_results()
        return

    def calc_transport_step(self):
        # Todo Check this again, done in a hurry!
        self.result = np.copy(self.data)
        if self.cnf.p.dim is not 1:
            print('So far only 1-D Problems are implemented!')
            assert False
        dt = self.cnf.t.d
        dp = self.cnf.p.d
        for s in range(self.cnf.s.n):
            beg = self.cnf.sv.index[s]
            end = self.cnf.sv.index[s+1]
            dv = self.cnf.sv.d * self.cnf.sv.multi
            # Todo removal of boundaries only temporary
            for p in range(1, self.cnf.p.size-1):
                for v in range(beg, end):
                    pv = dv * self.cnf.sv.G[v]
                    if pv[0] < 0:
                        d_trp = ((1 + pv[0]*dt/dp) * self.data[p, v]
                                 - pv[0]*dt/dp * self.data[p+1, v])
                    elif pv[0] > 0:
                        d_trp = ((1 - pv[0]*dt/dp) * self.data[p, v]
                                 + pv[0]*dt/dp * self.data[p-1, v])
                    else:
                        continue
                    self.result[p, v] = d_trp
        self.switch_data_results()
        return

    def write_results(self):
        self.f_out.apply(self.data)

    def switch_data_results(self):
        self.data, self.result = self.result, self.data
        return

    def check_conditions(self):
        # check Courant-Friedrichs-Levy-Condition
        max_v = self.cnf.sv.get_max_v()
        dt = self.cnf.t.d
        dp = self.cnf.p.d
        cfl = max_v * (dt/dp)
        assert cfl < 1/2
