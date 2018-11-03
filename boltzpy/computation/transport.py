import boltzpy.data as b_dat


def transport_function(scheme):
    if scheme["Transport_Scheme"] == "FiniteDifferences":
        if scheme["Transport_Order"] == 1:
            return _calculate_transport_step
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def _calculate_transport_step(data):
    """Executes single collision step on complete P-Grid"""
    assert isinstance(data, b_dat.Data)
    if data.p_dim != 1:
        message = 'Transport is currently only implemented ' \
                  'for 1D Problems'
        raise NotImplementedError(message)

    dt = data.dt
    dp = data.dp
    offset = data.v_offset
    for (spc, [beg, end]) in enumerate(data.v_range):
        # Todo removal of boundaries (p in range(1, ... -1))
        # Todo is only temporary,
        # Todo until rules for input/output points
        # Todo or boundary points are set
        for p in range(1, data.p_size - 1):
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
    data.state[...] = data.result[...]
    return
