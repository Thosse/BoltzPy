
import numpy as np



def pressure(state,
             delta_v,
             velocities,
             mass,
             mean_velocity):
    r"""

    Parameters
    ----------
    state : :obj:`~numpy.ndarray` [:obj:`float`]
    delta_v : :obj:`float`
        The physical spacing of the respective velocity grid.
    velocities: :obj:`~numpy.ndarray` [:obj:`float`]
        An array of all velocities.
        Each velocity is either 2 or 3 dimensional.
        Must be a 2D array.
    mass : :obj:`int`
    mean_velocity : :obj:`~numpy.ndarray` [:obj:`float`]
    """
    assert velocities.ndim == 2
    assert state.shape[-1] == velocities.shape[0]
    # Reshape arrays to use np.dot
    new_shape = state.shape[:-1]
    size = np.prod(new_shape, dtype=int)
    flat_shape = (size, state.shape[-1])
    dim = velocities.shape[1]
    state = state.reshape(flat_shape)
    velocities = velocities[np.newaxis, ...]
    mean_velocity = mean_velocity.reshape((size, 1, dim))
    deviation = mass / dim * np.sum((velocities - mean_velocity) ** 2, axis=2)
    result = delta_v**2 * np.sum(deviation * state, axis=1)
    return result.reshape(new_shape)


def temperature(pressure,
                number_density):
    assert pressure.shape == number_density.shape
    return pressure / number_density


def momentum_flow(state,
                  delta_v,
                  velocities,
                  mass):
    r"""Compute the momentum flow of the current distribution.

    Note
    ----
    Be aware that this must be computed separately for each single specimen.

    Parameters
    ----------
    state : :obj:`~numpy.ndarray` [:obj:`float`]
        A discretized velocity distribution function.
        Must be 2D array.
        For homogeneous case, add a np.newaxis.
    delta_v : :obj:`float`
        The physical spacing of the respective velocity grid.
    velocities: :obj:`~numpy.ndarray` [:obj:`float`]
        An array of all velocities.
        Each velocity is either 2 or 3 dimensional.
        Must be a 2D array.
    mass : :obj:`int`
        The particle mass of the species.
    """
    assert velocities.ndim == 2
    assert state.shape[-1] == velocities.shape[0]
    # Reshape arrays to use np.dot
    shape = state.shape[:-1]
    size = np.prod(shape, dtype=int)
    flat_shape = (size, state.shape[-1])
    dim = velocities.shape[1]
    new_shape = shape +(dim,)
    state = state.reshape(flat_shape)
    result = delta_v**2 * mass * np.dot(state, velocities**2)
    return result.reshape(new_shape)


# Todo Not sure if this is correct! Check this!
def energy_flow(state,
                delta_v,
                velocities,
                mass):
    r"""Compute the energy flow of the current distribution.

    Note
    ----
    Be aware that this must be computed separately for each single specimen.

    Parameters
    ----------
    state : :obj:`~numpy.ndarray` [:obj:`float`]
        A discretized velocity distribution function.
        Must be 2D array.
        For homogeneous case, add a np.newaxis.
    delta_v : :obj:`float`
        The physical spacing of the respective velocity grid.
    velocities: :obj:`~numpy.ndarray` [:obj:`float`]
        An array of all velocities.
        Each velocity is either 2 or 3 dimensional.
        Must be a 2D array.
    mass : :obj:`int`
        The particle mass of the species.
    """
    assert state.ndim == 2
    # Reshape arrays to use np.dot
    shape = state.shape[:-1]
    size = np.prod(shape, dtype=int)
    flat_shape = (size, state.shape[-1])
    dim = velocities.shape[1]
    new_shape = shape + (dim,)
    state = state.reshape(flat_shape)
    energies = 0.5 * mass * np.sum(velocities**2, axis=1)[:, np.newaxis]
    result = delta_v**dim * np.dot(state, energies * velocities)
    return result.reshape(new_shape)
