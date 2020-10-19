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
    assert np.all(data.state >= 0)
    data.t += 1
    return


#################################
#           Transport           #
#################################
def transport_outflow_remains(data, affected_points):
    # Todo make this an attribute of data? Or just pv = VG + offset?
    outflow_percentage = (np.abs(data.vG[:, 0] + data.velocity_offset[0])
                          * data.dt
                          / data.dp)
    result = ((1 - outflow_percentage) * data.state[affected_points, :])
    return result


def transport_inflow_innerPoint(data, affected_points):
    # # Todo move this into data.pv or something, is often needed
    pv = data.vG + data.velocity_offset
    inflow_percentage = (data.dt / data.dp * np.abs(pv[:, 0]))
    result = np.zeros((affected_points.size, data.vG.shape[0]), dtype=float)

    neg_vels = np.where(pv[:, 0] < 0)[0]
    result[:, neg_vels] = (inflow_percentage[neg_vels]
                           * data.state[np.ix_(affected_points + 1,
                                               neg_vels)]
                           )

    pos_vels = np.where(pv[:, 0] > 0)[0]
    result[:, pos_vels] = (inflow_percentage[pos_vels]
                           * data.state[np.ix_(affected_points - 1,
                                               pos_vels)]
                           )
    return result


# Todo test that this is equivalent to innerPoint, with all velocities allowed
def transport_inflow_boundaryPoint(data,
                                   affected_points,
                                   incoming_velocities):
    # # Todo move this into data.pv or something, is often needed
    pv = (data.vG + data.velocity_offset)[:, :]
    inflow_percentage = (data.dt / data.dp * np.abs(pv[:, 0]))
    result = np.zeros((affected_points.size, data.vG.shape[0]), dtype=float)

    neg_incomings_vels = np.where(pv[incoming_velocities, 0] < 0)[0]
    neg_vels = incoming_velocities[neg_incomings_vels]
    result[:, neg_vels] = (inflow_percentage[neg_vels]
                           * data.state[np.ix_(affected_points + 1, neg_vels)])

    pos_incomings_vels = np.where(pv[incoming_velocities, 0] > 0)[0]
    pos_vels = incoming_velocities[pos_incomings_vels]
    result[:, pos_vels] = (inflow_percentage[pos_vels]
                           * data.state[np.ix_(affected_points - 1, pos_vels)])
    return result


def fdm_first_order(data, affected_points):
    """Executes single collision step on complete P-Grid"""
    if data.p_dim != 1:
        message = 'Transport is currently only implemented ' \
                  'for 1D Problems'
        raise NotImplementedError(message)

    dt = data.dt
    dp = data.dp
    offset = data.velocity_offset
    v_size = data.v_range[-1, 1]
    for p in affected_points:
        for v in range(v_size):
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


def transport_fdm_inner(data, affected_points):
    """Executes single transport step for a set of inner points.

    This is a finite differences scheme of order 1 for inner points.
    It computes a free flow without any reflection or absorption.
    The results are saved in data.results"""
    if data.p_dim != 1:
        message = 'Transport is currently only implemented ' \
                  'for 1D Problems'
        raise NotImplementedError(message)
    # Simulate Outflowing
    data.result[affected_points, :] = transport_outflow_remains(data,
                                                                affected_points)
    # Simulate Inflow
    data.result[affected_points, :] += transport_inflow_innerPoint(
        data,
        affected_points
    )

    return


def no_transport(data, affected_points):
    """No Transport occurs here"""
    return


##################################
#           Collisions           #
##################################
def collision_operator(state,
                       collisions_relations,
                       collision_matrix):
    """Computes dt * J[f,f],
    with J[f,f] being the collision operator at all given points.
    These points are the ones specified in state.

    Note that this is the collision of all species.
    Collisions of species i with species j are not implemented.

    It is faster to include the dt in the collision matrix, thus the result is
    actually the collision operator times dt."""
    result = np.empty(state.shape, dtype=float)
    for p in range(state.shape[0]):
        u_c0 = state[p, collisions_relations[:, 0]]
        u_c1 = state[p, collisions_relations[:, 1]]
        u_c2 = state[p, collisions_relations[:, 2]]
        u_c3 = state[p, collisions_relations[:, 3]]
        col_factor = (np.multiply(u_c0, u_c2) - np.multiply(u_c1, u_c3))
        result[p] = collision_matrix.dot(col_factor)
    return result


def euler_scheme(data, affected_points):
    """Executes a collision step, by using the 1st order Euler scheme"""
    data.state[affected_points] += collision_operator(data.state[affected_points],
                                                      data.col,
                                                      data.col_mat)
    return


def collision_rkv4(data, affected_points):
    """Executes a collision step, by using the 4th order Runge Kutta scheme"""
    # Todo remove data, use sim.sv.size instead of state.shape[1]
    state = data.state[affected_points]
    result = np.zeros(state.shape, float)
    rkv_component = np.zeros(state.shape, float)
    rkv_offset = np.array([0, 0.5, 0.5, 1])
    rkv_weight = np.array([1/6, 2/6, 2/6, 1/6])
    for i in np.arange(4):
        offset = rkv_offset[i]
        weight = rkv_weight[i]
        rkv_component = collision_operator(state + offset * rkv_component,
                                           data.col,
                                           data.col_mat)
        result += weight * rkv_component
    data.state[affected_points] += result
    return


def no_collisions(data, affected_points):
    """No Collisions are done here"""
    return
