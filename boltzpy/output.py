
import numpy as np
import h5py

import boltzpy as bp


# Todo move this into Scheme, getting the sizes as attributes?
# Todo separate the moment functions to work on a single (spc) grid and return that result, move for loop into call
def generate_output_function(simulation,
                             hdf5_group_name="Computation"):
    """Returns a single callable function that handles the output.

    The returned function receives the current :attr:`Calculation.data`,
    generates the desired output,
    and writes it to :obj:`datasets <h5py.Dataset>`
    located in the given *hdf5_group*.

    Parameters
    ----------
    simulation : :class:`~boltzpy.Simulation`
    hdf5_group_name : :obj:`str`

    Returns
    -------
    :obj:`function`
    """
    assert isinstance(simulation, bp.Simulation)
    assert isinstance(hdf5_group_name, str)
    # Todo 1. move output_parameters into Scheme
    # Todo 2. test scheme here

    # Set up hdf group in storage file
    assert hdf5_group_name not in {'Collisions',
                                   'Geometry',
                                   'Position_Grid',
                                   'Scheme',
                                   'Species',
                                   'Time_grid',
                                   'Velocity_Grids'}
    hdf5_file = h5py.File(simulation.file_address + '.hdf5')
    if hdf5_group_name not in hdf5_file.keys():
        hdf5_file.create_group(hdf5_group_name)
    hdf5_group = hdf5_file[hdf5_group_name]
    # Todo clear everything else from this group (move output parameters to Scheme)

    # Todo describe output list
    output_list = np.empty(shape=(simulation.output_parameters.size, 2),
                           dtype=object)

    # set up and store hdf5 datasets
    for [idx_out, output] in enumerate(simulation.output_parameters.flatten()):
        # clear previous results, if any
        if output in hdf5_group.keys():
            del hdf5_group[output]
        # Define shape of dataset
        if output == "Complete_Distribution":
            shape = (simulation.t.size,
                     simulation.p.size,
                     simulation.sv.size)
        else:
            shape = (simulation.t.size,
                     simulation.p.size,
                     simulation.s.size)
        hdf5_group.create_dataset(output,
                                  shape=shape,
                                  dtype=float)
        output_list[idx_out, 1] = hdf5_group[output]

    # set up and store callable output functions
    for [idx_out, output] in enumerate(simulation.output_parameters.flatten()):
        if output == "Density":
            output_list[idx_out, 0] = density
        if output == "Mass":
            output_list[idx_out, 0] = mass
        if output == "Momentum_X":
            output_list[idx_out, 0] = momentum_x
        if output == "Momentum_Flow_X":
            output_list[idx_out, 0] = momentum_flow_x
        if output == "Energy":
            output_list[idx_out, 0] = energy
        if output == "Energy_Flow_X":
            output_list[idx_out, 0] = energy_flow_x
        if output == "Complete_Distribution":
            output_list[idx_out, 0] = complete_distribution

    # finally, setup output function,
    # that calls sub functions and writes to the proper groups
    def func(data, write_idx):
        for (f_out, hdf5_dataset) in output_list:
            hdf5_dataset[write_idx] = f_out(data)
        return

    return func


# Todo Multiply by some constant, for realistic values
def density(data):
    """Calculates and returns the particle density"""
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    result = np.empty(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        # mass is the sum over velocity grid of specimen
        result[..., s_idx] = np.sum(data.state[..., beg:end], axis=-1)
    return result


def mass(data):
    """Calculates and returns the mass"""
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    result = np.empty(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        # mass is the sum over velocity grid of specimen
        result[..., s_idx] = np.sum(data.state[..., beg:end],
                                    axis=-1)
        # TODO result[..., s_idx]  *= mass[s_idx]
    return result


def momentum_x(data):
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    result = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        V_dir = data.vG[beg:end, 0]
        result[..., s_idx] = np.sum(V_dir * data.state[..., beg:end], axis=1)
        result[..., s_idx] *= data.m[s_idx]
    return result


def momentum_flow_x(data):
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    result = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        V_dir = data.vG[beg:end, 0]
        result[..., s_idx] = np.sum(V_dir ** 2 * data.state[..., beg:end],
                                    axis=1)
        result[..., s_idx] *= data.m[s_idx]
    return result


def energy(data):
    """Calculates and returns the energy"""
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    result = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        V = data.vG[beg:end, :]
        V_2 = np.sqrt(np.sum(V ** 2, axis=1))
        result[..., s_idx] = np.sum(V_2 * data.state[..., beg:end],
                                    axis=1)
        result[..., s_idx] *= 0.5 * data.m[s_idx]
    return result


def energy_flow_x(data):
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    result = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        V = data.vG[beg:end, :]
        V_norm = np.sqrt(np.sum(V ** 2, axis=1))
        V_dir = data.vG[beg:end, 0]
        result[..., s_idx] = np.sum(V_norm * V_dir * data.state[..., beg:end],
                                    axis=1)
        result[..., s_idx] *= 0.5 * data.m[s_idx]
    return result


def complete_distribution(data):
    """Returns complete distribution of given data
    """
    return data.state
