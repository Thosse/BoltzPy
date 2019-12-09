
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
            # TODO THESE ARE HACKS remove , after updating the tests
            if p == 0:
                continue
            if p == data.p_size-1:
                continue
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
