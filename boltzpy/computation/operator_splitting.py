
import numpy as np

from boltzpy.computation import transport as cp_tr
from boltzpy.computation import collisions as cp_col


def operator_splitting_function(scheme):
    calc_transport_step = cp_tr.transport_function(scheme)
    calc_collision_step = cp_col.collision_function(scheme)
    if scheme["OperatorSplitting_Order"] == 1:
        def _calculate_time_step(data):
            """Executes a single time step"""
            # executing time step
            calc_transport_step(data)
            calc_collision_step(data)
            assert np.all(data.state > 0)
            data.t += 1
            return
        return _calculate_time_step
    else:
        raise NotImplementedError
