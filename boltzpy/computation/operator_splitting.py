
import numpy as np


def operator_splitting(data, func_transport, func_collision):
    """Executes a single time step"""
    # executing time step
    func_transport(data)
    func_collision(data)
    assert np.all(data.state > 0)
    data.t += 1
    return

