
import numpy as np


def project_velocities(vectors, direction):
    assert direction.shape == (vectors.shape[-1],)
    assert np.linalg.norm(direction) > 1e-8
    shape = vectors.shape[:-1]
    size = np.prod(shape, dtype=int)
    dim = vectors.shape[-1]
    vector = vectors.reshape((size, dim))
    angle = direction / np.linalg.norm(direction)
    result = np.dot(vector, angle)
    return result.reshape(shape)


def mf_stress(mass_array,
              centered_velocities,
              direction_1,
              direction_2):
    assert mass_array.shape == (centered_velocities.shape[0],)
    assert centered_velocities.shape == mass_array.shape + direction_1.shape
    assert direction_1.shape == direction_2.shape
    assert direction_2.shape == (centered_velocities.shape[-1],)
    dirs_are_equal = np.all(direction_1 - direction_2 < 1e-16)
    dirs_are_orthogonal = np.all(np.dot(direction_1, direction_2) < 1e-16)
    assert dirs_are_equal or dirs_are_orthogonal
    proj_vels_1 = project_velocities(centered_velocities, direction_1)
    proj_vels_2 = project_velocities(centered_velocities, direction_2)
    return mass_array * proj_vels_1 * proj_vels_2


# todo Add test for models (with default params) that its orthogonal to the moments
def mf_orthogonal_stress(mass_array,
                         centered_velocities,
                         direction_1,
                         direction_2):
    dirs_are_equal = np.all(direction_1 - direction_2 < 1e-16)
    dirs_are_orthogonal = np.all(np.dot(direction_1, direction_2) < 1e-16)
    result = mf_stress(mass_array, centered_velocities, direction_1, direction_2)
    if dirs_are_orthogonal:
        return result
    elif dirs_are_equal:
        dim = centered_velocities.shape[-1]
        mf_pressure = 1/dim * mass_array * np.sum(centered_velocities**2, axis=-1)
        return result - mf_pressure
    else:
        raise ValueError


def mf_heat_flow(mass_array, centered_velocities, direction):
    assert mass_array.shape == (centered_velocities.shape[0],)
    assert centered_velocities.shape == mass_array.shape + direction.shape
    assert direction.shape == (centered_velocities.shape[-1],)
    proj_vels = project_velocities(centered_velocities, direction)
    squared_sum = np.sum(centered_velocities**2, axis=-1)
    return mass_array * proj_vels * squared_sum


# todo Add test for models (with default params) that its orthogonal to the moments
def mf_orthogonal_heat_flow(mass_array, centered_velocities, direction):
    raise NotImplementedError


def number_density(state, delta_v):
    r"""

    Parameters
    ----------
    state : :obj:`~numpy.ndarray` [:obj:`float`]
    delta_v : :obj:`float`
        The physical spacing of the current velocity grid.
    """
    return np.sum(state, axis=-1) * delta_v ** 2


def mass_density(number_density, mass):
    r"""Compute the mass density

    Parameters
    ----------
    number_density : :obj:`~numpy.ndarray` [:obj:`float`]
    mass : :obj:`int`
            The mass factor of the species.
    """
    return mass * number_density


def momentum(state, delta_v, velocities, mass):
    r"""

    Parameters
    ----------
    state : :obj:`~numpy.ndarray` [:obj:`float`]
    delta_v : :obj:`float`
        The physical spacing of the current velocity grid.
    velocities: :obj:`~numpy.ndarray` [:obj:`float`]
        Each velocity is either 2 or 3 dimensional.
        Must be a 2D array.
    mass : :obj:`int`
    """
    assert velocities.ndim == 2
    assert state.shape[-1] == velocities.shape[0]
    # Reshape arrays to use np.dot
    shape = state.shape[:-1]
    size = np.prod(shape, dtype=int)
    flat_shape = (size, state.shape[-1])
    dim = velocities.shape[1]
    new_shape = shape + (dim,)
    state = state.reshape(flat_shape)
    result = delta_v**2 * mass * np.dot(state, velocities)
    return result.reshape(new_shape)


def mean_velocity(momentum, mass_density):
    r"""Compute the mean velocites in the current distribution.

    Note    ----
        Be aware that this must be computed separately for each single specimen.

    Parameters
    ----------
    momentum : :obj:`~numpy.ndarray` [:obj:`float`]
    mass_density : :obj:`~numpy.ndarray` [:obj:`float`]
    """
    shape = momentum.shape[:-1] + (1,)
    mass_density = mass_density.reshape(shape)
    return momentum / mass_density


def energy_density(state, delta_v, velocities, mass):
    r"""

    Parameters
    ----------
    state : :obj:`~numpy.ndarray` [:obj:`float`]
        Must be 2D array.
    delta_v : :obj:`float`
        The physical spacing of the current velocity grid.
    velocities: :obj:`~numpy.ndarray` [:obj:`float`]
        Each velocity is either 2 or 3 dimensional.
        Must be a 2D array.
    mass : :obj:`int`
        The particle mass of the species.
    """
    assert velocities.ndim == 2
    assert state.shape[-1] == velocities.shape[0]
    # Reshape arrays to use np.dot
    new_state = state.shape[:-1]
    size = np.prod(new_state, dtype=int)
    flat_shape = (size, state.shape[-1])
    dim = velocities.shape[1]
    state = state.reshape(flat_shape)
    energy = 0.5 * mass * np.sum(velocities**2, axis=-1)
    result = delta_v**dim * np.dot(state, energy)
    return result.reshape(new_state)


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
