import boltzpy as b_sim

import numpy as np
import h5py


# Todo The whole module needs a proper naming scheme
# Todo especially the output_functions / get_output_functions
# Todo as they are easily mistaken for the important output_function

# Todo Can momentum_xyz be unified somehow? extra parameter necessary?
def output_function(simulation,
                    hdf5_group,
                    ):
    """Returns a single callable function
    which receives the current :attr:`Calculation.data`,
    generates the desired output,
    and writes them to the given *hdf5_group* on the disk.

    The generated function applies the respective function
    for each output in the *output_parameters*
    onto the the  :attr:`Calculation.data`
    and writes the results to the
    :class:`boltzpy.Simulation` file.

    Parameters
    ----------
    simulation : :class:`~boltzpy.Simulation`
    hdf5_group : :obj:`h5py.Group`
    """
    assert isinstance(simulation, b_sim.Simulation)
    assert isinstance(hdf5_group, h5py.Group)
    # set up hdf5 datasets to store results in
    dataset_list = get_hdf5_datasets(simulation,
                                     hdf5_group)
    # setup output functions
    f_out_list = get_output_functions(simulation)
    # combine both lists to iterate over the tuples
    # Todo why doesn't output_list = zip(f_out_list, dataset_list) work?
    # Todo it only saves the value for t = 0, all the other values are 0
    output_list = [(f_out_list[i], dataset_list[i])
                   for i in range(len(dataset_list))]

    # setup output function, iteratively calls each output function
    def func(data, sv_idx_range_arr, mass_arr, velocities, time_idx):
        for (f_out, hdf5_dataset) in output_list:
            result = f_out(data, sv_idx_range_arr, mass_arr, velocities)
            hdf5_dataset[time_idx] = result
        return

    return func


def get_hdf5_datasets(simulation,
                      hdf5_group):
    output_list = simulation.output_parameters.flatten()
    # setup a dataset for each output
    for output in output_list:
        # clear previous results, if any
        if output in hdf5_group.keys():
            del hdf5_group[output]
        shape = get_output_shape(output,
                                 simulation.t.size,
                                 simulation.p.size,
                                 simulation.s.size,
                                 simulation.sv.size)
        hdf5_group.create_dataset(output,
                                  shape=shape,
                                  dtype=float)
    return [hdf5_group[output] for output in output_list]


def get_output_shape(output,
                     t_size,
                     p_size,
                     s_size,
                     sv_size):
    """Returns the shape of the hdf5 dataset of the given output
    based on the grid sizes.

    Parameters
    ----------
    output : :obj:`str`
         A single Output of the simulation.
         Must be in :const:`~boltzpy.constants.SUPP_OUTPUT`
    t_size : :obj:'int'
        Size of the time :class:`boltzpy.Grid`
    p_size : :obj:'int'
        Size of the position :class:`boltzpy.Grid`
    s_size : :obj:'int'
        Number of different :class:`boltzpy.Species`
    sv_size : :obj:'int'
        Size of the Velocity :class:`boltzpy.SVGrid`

    Returns
    -------
    :obj:`tuple`
    """
    if output in {'Mass',
                  'Momentum_X',
                  'Momentum_Y',
                  'Momentum_Z',
                  'Momentum_Flow_X',
                  'Momentum_Flow_Y',
                  'Momentum_Flow_Z',
                  'Energy',
                  'Energy_Flow_X',
                  'Energy_Flow_Y',
                  'Energy_Flow_Z'}:
        return (t_size,
                p_size,
                s_size,
                )
    elif output == 'Complete_Distribution':
        return (t_size,
                p_size,
                sv_size)
    else:
        message = 'Unsupported Output: {}'.format(output)
        raise NotImplementedError(message)


# Todo replace the different types of flows by a direction parameter
# Todo this is a vector and allows more flexibility
def get_output_functions(simulation):
    output_functions = []
    for output in simulation.output_parameters.flatten():
        # ignore time dimension
        if output == 'Mass':
            f = mass_function
        elif output == 'Momentum_X':
            f = get_momentum_function(0)
        elif output == 'Momentum_Flow_X':
            f = get_momentum_flow_function(0)
        elif output == 'Energy':
            f = energy_function
        elif output == 'Energy_Flow_X':
            f = get_energy_flow_function(0)
        elif output == 'Complete_Distribution':
            f = complete_distribution_function
        else:
            message = 'Unsupported Output: {}'.format(output)
            raise NotImplementedError(message)
        output_functions.append(f)
    return output_functions


# Todo multiply with mass
def mass_function(data, sv_idx_range_arr, mass_arr, velocities):
    """Calculates and returns the mass"""
    # shape = (position_grid.size, species.size)
    shape = (data.shape[0], mass_arr.size)
    mass = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(sv_idx_range_arr):
        # mass is the sum over velocity grid of specimen
        mass[..., s_idx] = np.sum(data[..., beg:end],
                                  axis=-1)
        # mass *= mass[s_idx]
    return mass


# Todo This is currently wrong!
# Todo It uses the indices, not the physical velocities
def get_momentum_function(direction):
    """Generates and returns generating function for Momentum"""
    assert direction in [0, 1, 2]

    def f_momentum(data, sv_idx_range_arr, mass_arr, velocities):
        # shape = (position_grid.size, species.size)
        shape = (data.shape[0], mass_arr.size)
        momentum = np.zeros(shape, dtype=float)
        for (s_idx, [beg, end]) in enumerate(sv_idx_range_arr):
            V_dir = velocities[beg:end, direction]
            momentum[..., s_idx] = np.sum(V_dir * data[..., beg:end],
                                          axis=1)
            momentum[..., s_idx] *= mass_arr[s_idx]
        return momentum

    return f_momentum


def get_momentum_flow_function(direction):
    """Generates and returns generating function for Momentum Flow"""
    assert direction in [0, 1, 2]

    def f_momentum_flow(data, sv_idx_range_arr, mass_arr, velocities):
        # shape = (position_grid.size, species.size)
        shape = (data.shape[0], mass_arr.size)
        momentum_flow = np.zeros(shape, dtype=float)
        for (s_idx, [beg, end]) in enumerate(sv_idx_range_arr):
            V_dir = velocities[beg:end, direction]
            momentum_flow[..., s_idx] = np.sum(V_dir ** 2
                                               * data[..., beg:end],
                                               axis=1)
            momentum_flow[..., s_idx] *= mass_arr[s_idx]
        return momentum_flow

    return f_momentum_flow


def energy_function(data, sv_idx_range_arr, mass_arr, velocities):
    """Calculates and returns the energy"""
    # shape = (position_grid.size, species.size)
    shape = (data.shape[0], mass_arr.size)
    energy = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(sv_idx_range_arr):
        V = velocities[beg:end, :]
        V_2 = np.sqrt(np.sum(V ** 2, axis=1))
        energy[..., s_idx] = np.sum(V_2 * data[..., beg:end],
                                    axis=1)
        energy[..., s_idx] *= 0.5 * mass_arr[s_idx]
    return energy


def get_energy_flow_function(direction):
    """Generates and returns generating function for Energy Flow"""
    assert direction in [0, 1, 2]

    def f_energy_flow(data, sv_idx_range_arr, mass_arr, velocities):
        # shape = (position_grid.size, species.size)
        shape = (data.shape[0], mass_arr.size)
        energy_flow = np.zeros(shape, dtype=float)
        for (s_idx, [beg, end]) in enumerate(sv_idx_range_arr):
            V = velocities[beg:end, :]
            V_norm = np.sqrt(np.sum(V ** 2, axis=1))
            V_dir = velocities[beg:end, direction]
            energy_flow[..., s_idx] = np.sum(V_norm
                                             * V_dir
                                             * data[..., beg:end],
                                             axis=1)
            energy_flow[..., s_idx] *= 0.5 * mass_arr[s_idx]
        return energy_flow

    return f_energy_flow


def complete_distribution_function(data):
    """Returns complete distribution of given data
    """
    return data

