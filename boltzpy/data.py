
import numpy as np

import boltzpy as bp


# Todo Add vG_squared and vG_norm attributes? faster output?
class Data:
    """Contains all parameters used by he computation.

    Parameters
    ----------
    file_address : :obj:`str`
        Address of the simulation file.
        Can be either a full path, a base file name or a file root.
        If no full path is given, then the file is placed in the
        :attr:`~boltzpy.constants.DEFAULT_DIRECTORY`.

    Attributes
    ----------
    state : :obj:`~numpy.array` [:obj:`float`]
        The distribution grid.
        Denotes the current state of the simulation.
    result : :obj:`~numpy.array` [:obj:`float`]
        Interim results of the computation are stored here.
    v_range : :obj:`~numpy.array` [:obj:`int`]
        Denotes begin and end of each
        :class:`Specimens <boltzpy.Specimen>` velocity grid.
    vG : :obj:`~numpy.array` [:obj:`float`]
        The physical velocity Grid (See :attr:`Model.pMG`)
    velocity_offset : :obj:`~numpy.array` [:obj:`float`]
        Offsets the velocities in and ONLY in the transport step
        (non-boundary points only!).
        This is best viewed as the velocity of a moving observer
        for the animation.
        It is useful to determine shock speeds.
    t : :obj:`int`
        Current time step.
    dt : :obj:`float`
        Current size of a time step.
        This may change during the simulation,
        if time-adaptive algorithms are used.
    tG : :obj:`~numpy.array` [:obj:`int`]
        Contains the time steps at which the output is written to file.
    dp : :obj:`float`
        Step size of the position space :class:`boltzpy.Grid`.
    p_dim : :obj:`int`
        Dimension of the position space :class:`boltzpy.Grid`.
    p_size : :obj:`int`
        Size of the position space :class:`boltzpy.Grid`.
    m : :obj:`~numpy.array` [:obj:`int`]
        Denotes the mass of every :class:`~boltzpy.Specimen`
    """
    def __init__(self, file_address):
        # create temporary Simulation instance
        sim = bp.Simulation.load(file_address)
        # data arrays, this contains all grids
        self.state = sim.geometry.initial_state
        self.result = np.copy(self.state)

        # Velocity Grid parameters
        self.v_range = np.zeros((sim.model.nspc, 2), dtype=int)
        self.v_range[:, 0] = sim.model._idx_offset[0: sim.model.nspc]
        self.v_range[:, 1] = sim.model._idx_offset[1:]

        self.vG = sim.model.vels
        # Todo reimplement offset -> geometry or simulation?
        self.velocity_offset = np.zeros(sim.model.ndim)
        #np.array(sim.scheme.Transport_VelocityOffset)
        # Todo Add this as property to SVGRID
        # Todo test if it faster to compute velocity (pv) on the fly
        self.dv = np.array([sim.model.vGrids[s].physical_spacing
                            for s in sim.model.species])

        self.n_spc = sim.model.nspc
        self.m = sim.model.masses

        self.t = 0
        self.tG = sim.timing.iG  # keep it, for adaptive time grids
        self.dt = sim.timing.delta

        self.dp = sim.geometry.delta
        self.p_dim = sim.geometry.ndim
        self.p_size = sim.geometry.size

        # Collision arrays
        self.col = sim.model.collision_relations
        self.weight = sim.model.collision_weights

        # Todo add rule arr (only boundary points necessary)
        #   with initialization scheme (standardized for initial_rho = 1)
        # Todo This involves a scheme to determine "sub-rules"
        # Todo as the position of the boundary is important for its
        # todo behaviour / reinitialization
        self.model = sim.model
        return

    def __getattr__(self, item):
        return self._params[item]

    # Todo Add more check_integrity / stability conditions?
    # Todo raise Warnings for weird configurations?
    def check_stability_conditions(self):
        """Checks Courant-Friedrichs-Levy Condition

        Raises
        ------
        AssertionError
            If any necessary condition is not satisfied."""
        # check Courant-Friedrichs-Levy-Condition
        max_v = np.max(np.linalg.norm(self.vG, axis=1))
        # Courant–Friedrichs–Lewy (CFL) condition
        assert max_v * (self.dt/self.dp) < 1/2
        return
