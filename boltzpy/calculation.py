
import boltzpy as b_sim
import boltzpy.data as b_dat
import boltzpy.output as b_out
import boltzpy.computation.operator_splitting as b_split

import numpy as np
from time import time

# Todo Lots of stuff to do

# Reduce data complexity (see todos) -> clean it up as much as possible
# rename computation -> calculation?
# move output into com/calc submodule
# Replace orders by a dict scheme -> create dict of dicts for asserts


def computation_function(scheme):
    if scheme["Approach"] == "DiscreteVelocityModels":
        return b_split.operator_splitting_function(scheme)
    else:
        msg = "Unsupported Approach: {}".format(scheme["Approach"])
        raise NotImplementedError(msg)


class Calculation:
    r"""Manages computation process

    For equivalent, straightforward, but less performant Code
    see the CalculationTest Class!

    ..todo::
        - enable writing of complete results (not moments) to sim file,
          for unittests
        - decide on tG:
            * is it an integer, or array(int)
            * for higher order transport -> multiple entries?
            * replace tG, and t_w by sparse matrix,
              such that transport is simple multiplication?
        - Properly implement computation( switches for Orders, vectorized))
        - Implement Operator Splitting of Order 2 (easy)
        - directly use p_flag? shrink it down
          (only 1 flag(==0) for inner points necessary)?
          Use p_flag as an array of pointers, that point to their
          computation function (depending on their type)
        - Implement complex Geometries for P-Grid:

            * Each P-Grid Point has a list of Pointers to its Neighbours
              (8 in 2D, 26 in 3D)
            * For each Velocity there is a list of Pointers
              to these Pointers
              (Which Neighbours to use for Transport-Calculation)

        - Implement adaptive P-Grid:
            * The P-Grid is given as an array of the P-Gris Points
              -> Array_0
            * Each P-Grid Point has several separate lists of Pointers:

              * List_0 contains Pointers to the Neighbours in the P-Grid.
              * List_1,... are empty in the beginning

            * If a finer Grid is necessary (locally), then:

              * Add a Array with the additional points - Array_1
              * The Entries of the additional Points are interpolated
                from the neighbours in the coarse grid
              * Fill List_1 with Pointers to the new,
                finer neighbouring points in Array_1
              * Note that it's necessary to check for some points, if they've
                been added already, by neighbouring points
                (for each Neighbour in List_0, check List_1).
                In that case only add the remaining points to Array_1 and
                let the Pointer in List_1 point to the existing point in
                Array_1
              * In Order to fulfill the boundary conditions it is necessary
                to add "Ghost-Boundary-Points" to the neighbouring points
                of the finer Grid.
                There is a PHD-Thesis about this. -> Find it
              * Those Ghost points are only interpolated
                from their Neighbours
                there is no extra transport & collision step necessary
                for them
              * When moving to a finer Grid, the time-step is halved
                for both transport and collision on all Grids.
                This is not that bad, as for Collisions only the number of
                sub-collision-steps is halved, by equal weight/time_step
                (as long as this is possible).
                => Only the transport step gets more intense
              * there should be additional p_flags for each point, denoting,
                if its a point with 'sub'points or a neighbouring point
                of a finer area,with 'Ghost-Boundary-Points'
              * The total number of Grid-Refinement-Levels should be bound

    Parameters
    ----------
    simulation : :class:`~boltzpy.Simulation`

    Attributes
    ----------
    data : :obj:`~np.ndarray` [:obj:`float`]
        Current state of the simulation, such that
        :math:`f(t_{cur}, p, v) = data[i_p, i_v]`.
        Array of shape
        (:attr:`cnf.p.size <boltzpy.Grid.size>`,
        :attr:`cnf.sv.size <boltzpy.SVGrid.size>`).
    f_out : :class:`~boltzpy.output.OutputFunction`
        Handles generation and saving of interim results
    """
    def __init__(self, simulation):
        assert isinstance(simulation, b_sim.Simulation)
        self.sim = simulation
        self.data = None
        self.f_out = None
        return

    #####################################
    #            Calculation            #
    #####################################
    def run(self, hdf5_group):
        """Starts the Calculation and writes the interim results
        to the disk
        """
        # Initialize PSV-Grids
        self.data = b_dat.Data(self.sim.file_address)
        # Todo Add check_integrity / stability conditions?
        assert self.check_stability_conditions()
        # configure calculation_function
        _calculate_time_step = computation_function(self.sim.scheme)
        # Prepare Output functions
        self.f_out = b_out.output_function(self.sim,
                                           hdf5_group=hdf5_group)

        print('Calculating...          ', end='\r')

        for (tw_idx, tw) in enumerate(self.data.tG[:, 0]):
            while self.data.t != tw:
                _calculate_time_step(self.data)
                self._print_time_estimate()
            # generate Output and write it to disk
            # Todo this needs a data (cpu/GPU) parameter, containing all
            # Todo replace this by a sv_grid attribute idx_range
            # Todo replace sv._index and self.index_range() by index_range
            idx_range = self.data.v_range
            self.f_out(self.data.state,
                       np.array(idx_range),
                       self.data.m,
                       self.data.vG,
                       tw_idx)

        time_taken_in_seconds = int(time() - self.data.dur_total)
        # large number of spaces necessary to overwrite the old terminal lines
        print('Calculating... Done' + 40*' ' + '\n'
              'Time taken = ' + self._format_time(time_taken_in_seconds))
        return

    def _print_time_estimate(self):
        """Prints an estimate of the remaining time to the terminal"""
        rem_steps = self.data.tG[-1, 0] - self.data.t
        est_step_duration = (time() - self.data.dur_total) / self.data.t
        est_time_in_seconds = int(rem_steps * est_step_duration)
        print('Calculating... '
              + self._format_time(est_time_in_seconds)
              + ' remaining',
              end='\r')
        return

    @staticmethod
    def _format_time(t_in_seconds):
        (days, t_in_seconds) = divmod(t_in_seconds, 86400)
        (hours, t_in_seconds) = divmod(t_in_seconds, 3600)
        (minutes, t_in_seconds) = divmod(t_in_seconds, 60)
        t_string = '{:2d}d {:2d}h {:2d}m {:2d}s'.format(days,
                                                        hours,
                                                        minutes,
                                                        t_in_seconds)
        return t_string

    def check_stability_conditions(self):
        """Checks Courant-Friedrichs-Levy Condition

        Returns
        -------
        bool
            True, if all conditions are satisfied.
            False, otherwise."""
        # check Courant-Friedrichs-Levy-Condition
        max_v = np.linalg.norm(self.data.vG,
                               axis=1).max()
        dt = self.data.dt
        dp = self.data.dp
        cfl_condition = max_v * (dt/dp) < 1/2
        return cfl_condition
