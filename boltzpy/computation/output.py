
import numpy as np
import h5py

import boltzpy as bp
import boltzpy.constants as bp_c

# Todo momentum and the flows could receive an additional parameter: direction
# Todo      The direction would be a (normalized) vector (2D/3D)
# Todo      This would allow to view the in any possible direction
def output_function(simulation,
                    hdf5_group,
                    ):
    """Returns a single callable function that handles the output.

    The returned function receives the current :attr:`Calculation.data`,
    generates the desired output,
    and writes it to :obj:`datasets <h5py.Dataset>`
    located in the given *hdf5_group*.

    Parameters
    ----------
    simulation : :class:`~boltzpy.Simulation`
    hdf5_group : :obj:`h5py.Group <h5py:Group>`

    Returns
    -------
    :obj:`function`
    """
    assert isinstance(simulation, bp.Simulation)
    assert isinstance(hdf5_group, h5py.Group)
    # set up hdf5 datasets for every output
    datasets = _setup_datasets(simulation, hdf5_group)
    # setup output functions for every output
    subfuncs = _setup_subfuncs(simulation)
    # combine both lists to iterate over the tuples
    output_list = [(subfuncs[output], datasets[output])
                   for output in simulation.output_parameters.flatten()]

    # setup output function, iteratively calls each output function
    def func(data, write_idx):
        for (f_out, hdf5_dataset) in output_list:
            result = f_out(data)
            hdf5_dataset[write_idx] = result
        return

    return func


def _setup_datasets(simulation, hdf5_group):
    """Create a :obj:`h5py.Dataset <h5py:Dataset>` for every output
    in the hdf5_group and return an ordered list of them.

    Any already existing datasets will be replaced.

    Parameters
    ----------
    simulation : :class:`~boltzpy.Simulation`
    hdf5_group : :obj:`h5py.Group <h5py:Group>`

    Returns
    -------
    :obj:`dict` [:obj:`str`:  :obj:`h5py.Group <h5py:Group>`]
    """
    datasets = dict()
    # setup a dataset for each output
    for output in simulation.output_parameters.flatten():
        # clear previous results, if any
        if output in hdf5_group.keys():
            del hdf5_group[output]
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
        datasets[output] = hdf5_group[output]
    return datasets


def _setup_subfuncs(simulation):
    """Create a generating function, for each specified output.
    Return an ordered list of these functions.


        Parameters
        ----------
        simulation : :class:`~boltzpy.Simulation`

        Returns
        -------
        :obj:`dict` [:obj:`str`:  :obj:`function`]
        """
    subfuncs = dict()
    for output in simulation.output_parameters.flatten():
        assert output in bp_c.SUPP_OUTPUT
        # Read sub functions from globally defined functions
        # store sub functions in dictionary
        subfuncs[output] = globals()["_subfunc_" + output.lower()]
    return subfuncs


# Todo multiply with mass?
def _subfunc_mass(data):
    """Calculates and returns the mass"""
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    mass = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        # mass is the sum over velocity grid of specimen
        mass[..., s_idx] = np.sum(data.state[..., beg:end],
                                  axis=-1)
        # mass *= mass[s_idx]
    return mass


def _get_subfunc_momentum(direction):
    """Generates and returns generating function for momentum"""
    assert direction in [0, 1, 2]

    def f_momentum(data):
        # shape = (position_grid.size, species.size)
        shape = (data.p_size, data.n_spc)
        momentum = np.zeros(shape, dtype=float)
        for (s_idx, [beg, end]) in enumerate(data.v_range):
            V_dir = data.vG[beg:end, direction]
            momentum[..., s_idx] = np.sum(V_dir * data.state[..., beg:end],
                                          axis=1)
            momentum[..., s_idx] *= data.m[s_idx]
        return momentum

    return f_momentum

_subfunc_momentum_x = _get_subfunc_momentum(0)
_subfunc_momentum_y = _get_subfunc_momentum(1)
_subfunc_momentum_z = _get_subfunc_momentum(2)


def _get_subfunc_momentum_flow(direction):
    """Generates and returns generating function for momentum flow"""
    assert direction in [0, 1, 2]

    def f_momentum_flow(data):
        # shape = (position_grid.size, species.size)
        shape = (data.p_size, data.n_spc)
        momentum_flow = np.zeros(shape, dtype=float)
        for (s_idx, [beg, end]) in enumerate(data.v_range):
            V_dir = data.vG[beg:end, direction]
            momentum_flow[..., s_idx] = np.sum(V_dir ** 2
                                               * data.state[..., beg:end],
                                               axis=1)
            momentum_flow[..., s_idx] *= data.m[s_idx]
        return momentum_flow

    return f_momentum_flow

_subfunc_momentum_flow_x = _get_subfunc_momentum_flow(0)
_subfunc_momentum_flow_y = _get_subfunc_momentum_flow(1)
_subfunc_momentum_flow_z = _get_subfunc_momentum_flow(2)


def _subfunc_energy(data):
    """Calculates and returns the energy"""
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    energy = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        V = data.vG[beg:end, :]
        V_2 = np.sqrt(np.sum(V ** 2, axis=1))
        energy[..., s_idx] = np.sum(V_2 * data.state[..., beg:end],
                                    axis=1)
        energy[..., s_idx] *= 0.5 * data.m[s_idx]
    return energy


def _get_subfunc_energy_flow(direction):
    """Generates and returns generating function for energy flow"""
    assert direction in [0, 1, 2]

    def f_energy_flow(data):
        # shape = (position_grid.size, species.size)
        shape = (data.p_size, data.n_spc)
        energy_flow = np.zeros(shape, dtype=float)
        for (s_idx, [beg, end]) in enumerate(data.v_range):
            V = data.vG[beg:end, :]
            V_norm = np.sqrt(np.sum(V ** 2, axis=1))
            V_dir = data.vG[beg:end, direction]
            energy_flow[..., s_idx] = np.sum(V_norm
                                             * V_dir
                                             * data.state[..., beg:end],
                                             axis=1)
            energy_flow[..., s_idx] *= 0.5 * data.m[s_idx]
        return energy_flow

    return f_energy_flow

_subfunc_energy_flow_x = _get_subfunc_energy_flow(0)
_subfunc_energy_flow_y = _get_subfunc_energy_flow(1)
_subfunc_energy_flow_z = _get_subfunc_energy_flow(2)


def _subfunc_complete_distribution(data):
    """Returns complete distribution of given data
    """
    return data.state
