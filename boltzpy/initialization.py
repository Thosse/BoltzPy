
import numpy as np
import math


# This file will contain functions related to initialization of PSV Grids
# Todo Explain PSV Grid in simulation or computation
def create_psv_grid(self):
    """Set up and return an initialized PSV-Grid.

    Sets up a PSV-Grid and initializes the velocity distribution
    in each :class:`P-Grid <boltzpy.Grid>` point
    by applying its :meth:`chosen <boltzpy.Simulation>`
    :class:`initialization rule <Rule>`.

    Returns
    -------
    psv : :class:`~numpy.array` [:obj:`float`]
        The initialized PSV-Grid.
        Array of shape
        (:attr:`Simulation.p.size
        <boltzpy.Grid.size>`,
        :attr:`Simulation.sv.size
        <boltzpy.SVGrid.size>`).
    """
    self.check_integrity()
    assert self.init_arr.size == self.p.size
    psv = np.zeros(shape=(self.p.size,
                          self.sv.size),
                   dtype=float)
    # set Velocity Grids for all specimen
    for (i_p, i_rule) in enumerate(self.init_arr):
        # get active rule
        r = self.rule_arr[i_rule]
        # Todo - simply call r_apply method?
        for i_s in range(self.s.n):
            rho = r.rho[i_s]
            temp = r.temp[i_s]
            [begin, end] = self.sv.range_of_indices(i_s)
            v_grid = self.sv.iMG[begin:end]
            dv = self.sv.vGrids[i_s].d
            for (i_v, v) in enumerate(v_grid):
                # Physical Velocity
                pv = dv * v
                # Todo np.array(v) only for PyCharm Warning - Check out
                diff_v = np.sum((np.array(pv) - r.drift[i_s])**2)
                psv[i_p, begin + i_v] = rho * math.exp(-0.5*(diff_v/temp))
            # Todo read into Rjasanov's script and do this correctly
            # Todo THIS IS CURRENTLY WRONG! ONLY TEMPORARY FIX
            # Adjust initialized values, to match configurations
            adj = psv[i_p, begin:end].sum()
            psv[i_p, begin:end] *= rho/adj
    return psv
