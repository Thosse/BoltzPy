import numpy as np


def collision_function(scheme):
    if scheme["Collisions_ComputationScheme"] == "EulerScheme":
        return _calculate_collision_step
    else:
        raise NotImplementedError


def _calculate_collision_step(data):
    """Executes a single collision step on complete P-Grid"""
    for p in range(data.p_size):
        u_c0 = data.state[p, data.col[:, 0]]
        u_c1 = data.state[p, data.col[:, 1]]
        u_c2 = data.state[p, data.col[:, 2]]
        u_c3 = data.state[p, data.col[:, 3]]
        col_factor = (np.multiply(u_c0, u_c2) - np.multiply(u_c1, u_c3))
        data.state[p] += data.col_mat.dot(col_factor)
    return
