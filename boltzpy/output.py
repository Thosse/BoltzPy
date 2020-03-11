
import numpy as np


def momentum(state,
             delta_v,
             velocities,
             mass):
    r"""Compute the momentum of the current distribution.

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
    weighted_state = state * mass * delta_v**2
    return np.dot(weighted_state, velocities)


def energy(state,
           delta_v,
           velocities,
           mass):
    r"""Compute the energy of the current distribution.

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
    dim = velocities.shape[1]
    energies = 0.5 * mass * np.sum(velocities**2, axis=1)
    weighted_state = state * delta_v**dim
    return np.dot(weighted_state, energies)


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
    assert state.ndim == 2
    weighted_state = state * mass * delta_v**2
    return np.dot(weighted_state, velocities**2)


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
    dim = velocities.shape[1]
    energies = 0.5 * mass * np.sum(velocities**2, axis=1)[:, np.newaxis]
    weighted_state = state * delta_v**dim
    return np.dot(weighted_state, energies * velocities)
