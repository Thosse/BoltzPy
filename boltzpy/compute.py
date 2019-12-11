r""""..todo::

    - decide on tG:
        * is it an integer, or array(int)
        * for higher order transport -> multiple entries?
        * replace tG, and t_w by sparse matrix,
          such that transport is simple multiplication?
    - Properly implement computation( switches for Orders, vectorized))
    - Implement Operator Splitting of Order 2 (easy)
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
    - implement p-local time-adaptive scheme?
        * split collision in multiple collision steps, to keep stability
"""
import numpy as np


##################################
#       Operator Splitting       #
##################################
def operator_splitting(data, func_transport, func_collision):
    """Executes a single time step"""
    # executing time step
    func_transport(data)
    func_collision(data)
    assert np.all(data.state > 0)
    data.t += 1
    return


#################################
#           Transport           #
#################################
def fdm_first_order(data, affected_points):
    """Executes single collision step on complete P-Grid"""
    if data.p_dim != 1:
        message = 'Transport is currently only implemented ' \
                  'for 1D Problems'
        raise NotImplementedError(message)

    dt = data.dt
    dp = data.dp
    offset = data.velocity_offset
    # Todo is this case separation necessary?
    for (spc, [beg, end]) in enumerate(data.v_range):
        for p in affected_points:
            # # Todo there should be a faster way -> vectorize, keep old function for testing
            for v in range(beg, end):
                pv = data.vG[v] + offset
                if pv[0] <= 0:
                    new_val = ((1 + pv[0] * dt / dp) * data.state[p, v]
                               - pv[0] * dt / dp * data.state[p + 1, v])
                elif pv[0] > 0:
                    new_val = ((1 - pv[0] * dt / dp) * data.state[p, v]
                               + pv[0] * dt / dp * data.state[p - 1, v])
                else:
                    continue
                data.result[p, v] = new_val
    return


def no_transport(data, affected_points):
    """No Transport occurs here"""
    return


##################################
#           Collisions           #
##################################
# Todo this needs the col_mat, make sure this is the case
def euler_scheme(data, affected_points):
    """Executes a single collision step on complete P-Grid"""
    for p in affected_points:
        u_c0 = data.state[p, data.col[:, 0]]
        u_c1 = data.state[p, data.col[:, 1]]
        u_c2 = data.state[p, data.col[:, 2]]
        u_c3 = data.state[p, data.col[:, 3]]
        col_factor = (np.multiply(u_c0, u_c2) - np.multiply(u_c1, u_c3))
        data.state[p] += data.col_mat.dot(col_factor)
    return


def no_collisions(data, affected_points):
    """No Collisions are done here"""
    return
