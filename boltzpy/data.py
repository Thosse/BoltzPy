"""
General Information
    different time_taken variables
        to compere load of transport, collision, writeOuts and so on
    necessary? use profiling tool instead!
Output needs
    data,
    sv_idx_range_arr,
    velocities (sv.pMG),
    mass_arr
"""
import boltzpy as b_sim
import boltzpy.initialization as b_ini
import boltzpy.collision_relations as b_col

import numpy as np
from time import time


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
    v_offset : :obj:`~numpy.array` [:obj:`float`]
        Offsets the velocities in the transport step
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
        sim = b_sim.Simulation(file_address)
        # data arrays, this contains all grids
        # Todo Rework initialization (move into rules?)
        # Todo Class for single Space points (V-Grid + 0.Moment)?
        self.state = b_ini.create_psv_grid(sim)
        self.result = np.copy(self.state)

        # Velocity Grid parameters
        # Todo rework SVGRID._index -> idx_range, shape = (s.size, 2)
        # Todo noting begin and end of a specimens velocity grid
        self.v_range = np.array([[sim.sv._index[idx], sim.sv._index[idx+1]]
                                 for idx in range(sim.s.size)])
        # Todo rename pMG -> pG and implement efficiently
        self.vG = sim.sv.pMG
        # Todo rename offset, remove from svgrid -> movingObserver?
        self.v_offset = sim.sv.offset
        # Todo Add this as property to SVGRID
        # Todo test if it faster to compute velocity (pv) on the fly
        # self.dv = np.array([sim.sv.vGrids[s].d for s in range(sim.s.size)])

        self.m = sim.s.mass

        self.t = 0
        self.tG = sim.t.iG  # keep it, for adaptive time grids
        self.dt = sim.t.d

        self.dp = sim.p.d
        # Todo move into init parameters?
        self.p_dim = sim.p.dim
        self.p_size = sim.p.size

        # Todo maybe a bad name -> denotes start time, not duration!
        self.dur_total = time()
        self.dur_col = 0.0
        self.dur_transp = 0.0

        # Collision arrays
        # # Todo implement proper collision setup()
        col = b_col.CollisionRelations(sim)
        # Generate collision relations
        # Todo add coll_select_scheme = "free_flow"
        # Todo move this if into col.setup()
        if sim.coll_substeps != 0:
            col.setup()

        # Todo Move into Scheme
        self.col_steps = sim.coll_substeps
        # Todo create struct -> 4 ints and 1 float together -> possible?
        self.col = col.collision_arr
        self.weight = col.weight_arr

        # Array, denotes the category of a space point
        # and thus its behaviour
        self.category = sim.init_arr

        # Todo add rule arr (only boundary points necessary)
        #   with initialization scheme (standardized for rho = 1)
        # Todo This involves a scheme to determine "sub-rules"

        self._params = dict()
        # Keep as a "conditional" attribute?
        self._params["col_mat"] = col.mat

        return

    def __getattr__(self, item):
        return self._params[item]
