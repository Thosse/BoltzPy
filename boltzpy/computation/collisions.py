import numpy as np


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
