from time import time

import numpy as np

import boltzpy as bp
import boltzpy.initialization as bp_ini


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
        The physical velocity Grid (See :attr:`SVGrid.pMG`)
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
    category : :obj:`~numpy.array` [:obj:`int`]
        Defines the behaviour of each point in P-Space
        in the computation.
    dur_total : :obj:`float`
        Stores total computation time.
    dur_col : :obj:`float`
        Stores computation time spent on collision step.
    dur_transp : :obj:`float`
        Stores computation time spent on transport step.
    """
    def __init__(self, file_address):
        # create temporary Simulation instance
        sim = bp.Simulation(file_address)
        # data arrays, this contains all grids
        # Todo Rework initialization (move into rules?)
        # Todo Class for single Space points (V-Grid + 0.Moment)?
        self.state = bp_ini.create_psv_grid(sim)
        self.result = np.copy(self.state)

        # Velocity Grid parameters
        self.v_range = sim.sv.index_range
        # Todo rename pMG -> pG and implement efficiently
        self.vG = sim.sv.delta * sim.sv.iMG
        # Todo rename offset, remove from svgrid -> movingObserver?
        self.velocity_offset = np.array(sim.scheme.Transport_VelocityOffset)
        # Todo Add this as property to SVGRID
        # Todo test if it faster to compute velocity (pv) on the fly
        # self.dv = np.array([sim.sv.vGrids[s].d for s in range(sim.s.size)])

        self.n_spc = sim.s.size
        self.m = sim.s.mass

        self.t = 0
        self.tG = sim.t.iG  # keep it, for adaptive time grids
        self.dt = sim.t.delta

        self.dp = sim.p.delta
        self.p_dim = sim.p.ndim
        self.p_size = sim.p.size

        # Todo maybe a bad name -> denotes start time, not duration!
        self.dur_total = time()
        self.dur_col = 0.0
        self.dur_transp = 0.0

        # Collision arrays
        # Todo create struct -> 4 ints and 1 float together -> possible?
        if not sim.coll.is_set_up:
            sim.coll.setup(sim.scheme, sim.sv, sim.s)
        self.col = sim.coll.relations
        self.weight = sim.coll.weights

        # Array, denotes the behaviour_type of a space point
        # and thus its behaviour
        self.category = sim.init_arr

        # Todo add rule arr (only boundary points necessary)
        #   with initialization scheme (standardized for initial_rho = 1)
        # Todo This involves a scheme to determine "sub-rules"
        # Todo as the position of the boundary is important for its
        # todo behaviour / reinitialization

        self._params = dict()
        # Keep as a "conditional" attribute?
        self._params["col_mat"] = sim.coll.generate_collision_matrix(sim.t.delta)
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
        max_v = np.linalg.norm(self.vG, axis=1).max()
        # Courant–Friedrichs–Lewy (CFL) condition
        assert max_v * (self.dt/self.dp) < 1/2
        return
