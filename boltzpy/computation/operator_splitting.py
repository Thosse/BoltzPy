import boltzpy.data as b_dat
from . import transport as b_transp
from . import collisions as b_collide

import numpy as np


def operator_splitting_function(scheme):
    calc_transport_step = b_transp.transport_function(scheme)
    calc_collision_step = b_collide.collision_function(scheme)
    if scheme["OperatorSplitting_Order"] == 1:
        def _calculate_time_step(data):
            """Executes a single time step"""
            assert isinstance(data, b_dat.Data)
            # executing time step
            calc_transport_step(data)
            calc_collision_step(data)
            assert np.all(data.state > 0)
            data.t += 1
            return
        return _calculate_time_step
    else:
        raise NotImplementedError