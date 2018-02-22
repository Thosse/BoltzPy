import numpy as np


class Calculation:
    """
    Manages calculation process based on minimal set of parameters.
    This class focuses purely on performance
    and readily sacrifices readability.
    For equivalent, more understandable, but less performant Code
    see the CalculationTest Class!

    ..todo::
        - fix documentation of cols (new line for each v_pre and v_post)
        - Definition of t_arr is only temporary

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
    c_w : np.ndarray(float)
        c_w[i_c] denotes the weight for c_arr[i_c, :]
        in the collision step
    p_flag : np.ndarray(int)
        Let i_p be an index of P-Space, then
        p_flag[i_p] describes whether
        i_p is an inner point, boundary point, or input/output point.
        This controls the behaviour of the calculation in this point.
        For each value in p_flag a custom sub-function is generated.
    """
    def __init__(self):
        self.data = np.zeros((0,), dtype=float)
        self.result = np.zeros((0,), dtype=float)
        self.c_arr = np.zeros((0, 4), dtype=int)
        self.c_w = np.zeros((0,), dtype=float)
        self.t_arr = np.zeros((0,), dtype=int)
        self.t_w = np.zeros((0,), dtype=float)
        return
