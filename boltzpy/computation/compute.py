r""""..todo::

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
    - implement p-local time-adaptive scheme?
        * split collision in multiple collision steps, to keep stability
"""

from time import time
import h5py

import boltzpy as bp
import boltzpy.computation as cp
from boltzpy.computation import operator_splitting as cp_os
from boltzpy.computation import output as cp_out


# Todo Check if file address is a proper hdf5 file
# Todo check if data can properly generated from the file
def compute(file_address,
            hdf5_group_name="Computation"):
    assert isinstance(file_address, str)
    # Generate Computation data
    data = cp.Data(file_address)
    data.check_stability_conditions()
    # Generate computation functions
    f_compute = generate_computation_functions(file_address)
    f_output = generate_output_functions(file_address,
                                         hdf5_group_name)

    # Start computation
    print('Calculating...          ', end='\r')
    # Todo this might be buggy, if data.tG changes
    # Todo e.g. in adaptive time schemes
    # Todo proposition: iterate over length?
    for (tw_idx, tw) in enumerate(data.tG[:, 0]):
        while data.t != tw:
            f_compute(data)
            _print_time_estimate(data)
        # generate Output and write it to disk
        f_output(data, tw_idx)

    time_taken_in_seconds = int(time() - data.dur_total)
    # large number of spaces necessary to overwrite the old terminal lines
    print('Calculating... Done'
          + 40 * ' ' + '\n'
          + 'Time taken = ' + _format_time(time_taken_in_seconds))
    return


def generate_computation_functions(file_address):
    hdf5_group = h5py.File(file_address + ".hdf5")["Computation"]
    scheme = bp.Scheme.load(hdf5_group)
    if scheme["Approach"] == "DiscreteVelocityModels":
        return cp_os.operator_splitting_function(scheme)
    else:
        msg = "Unsupported Approach: {}".format(scheme["Approach"])
        raise NotImplementedError(msg)


# Todo Move this into output.py?
def generate_output_functions(file_address,
                              hdf5_group_name):
    simulation = bp.Simulation(file_address)
    assert hdf5_group_name not in {'Collisions',
                                   'Initialization',
                                   'Position_Grid',
                                   'Species',
                                   'Time_grid',
                                   'Velocity_Grids'}
    hdf5_file = h5py.File(file_address + '.hdf5')
    if hdf5_group_name not in hdf5_file.keys():
        hdf5_file.create_group(hdf5_group_name)
    hdf5_group = h5py.File(file_address + '.hdf5')[hdf5_group_name]
    return cp_out.output_function(simulation,
                                  hdf5_group)


# Todo profile this!
def _print_time_estimate(data):
    """Prints an estimate of the remaining time to the terminal"""
    rem_steps = data.tG[-1, 0] - data.t
    est_step_duration = (time() - data.dur_total) / data.t
    est_time_in_seconds = int(rem_steps * est_step_duration)
    print('Calculating... '
          + _format_time(est_time_in_seconds)
          + ' remaining',
          end='\r')
    return


def _format_time(t_in_seconds):
    (days, t_in_seconds) = divmod(t_in_seconds, 86400)
    (hours, t_in_seconds) = divmod(t_in_seconds, 3600)
    (minutes, t_in_seconds) = divmod(t_in_seconds, 60)
    t_string = '{:2d}d {:2d}h {:2d}m {:2d}s'.format(days,
                                                    hours,
                                                    minutes,
                                                    t_in_seconds)
    return t_string
