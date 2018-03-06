from boltzmann.configuration import configuration as b_cnf
from boltzmann.initialization import initialization as b_ini
from . import output_function as b_opf

import numpy as np


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
        - fix documentation of cols (new line for each v_pre and v_post)
        - Definition of t_arr is only temporary
        - directly use p_flag? or shrink it down
          (only 1 flag(==0) for inner points necessary)
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
              * Fill List_1 with Pointers to the new, finer neighbouring points
                in Array_1
              * Note that it's necessary to check for some points, if they've
                been added already, by neighbouring points
                (for each Neighbour in List_0, check List_1).
                In that case only add the remaining points to Array_1 and
                let the Pointer in List_1 point to the existing point in
                Array_1
              * In Order to fullfill the boundary conditions it is necessary
                to add "Ghost-Boundary-Points" to the boundary of the finer
                Grid. There is a PHD-Thesis about this. -> Find it
              * The total number of Grid-Refinement-Levels is


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
        For each value in p_flag a custom sub-function is generated.
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
    """
    def __init__(self,
                 cnf=b_cnf.Configuration(),
                 ini=b_ini.Initialization(),
                 moments=list()):
        self.data = ini.create_psv_grid()
        self.result = np.copy(self.data)
        self.p_flag = ini.p_flag
        self.c_arr = cnf.cols.i_arr
        self.c_w = cnf.cols.weight
        self.t_arr = np.zeros((0,), dtype=int)
        self.t_w = np.zeros((0,), dtype=float)
        # Todo moments is only temporary here
        self.f_out = b_opf.OutputFunction(moments, cnf)
        return
