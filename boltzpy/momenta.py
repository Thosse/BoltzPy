import numpy as np


def particle_number(distribution,
                    delta_v):
    r"""Compute the number of particles in the current distribution.

    Note
    ----
    Be aware that this must be computed separately for each single specimen.

    Parameters
    ----------
    distribution : :obj:`~numpy.ndarray` [:obj:`float`]
        A discretized velocity distribution function.
        Must be 2D array.
        For homogeneous case, add a np.newaxis.
    delta_v : :obj:`float`
        The physical spacing of the respective velocity grid.
    """
    assert distribution.ndim == 2
    return np.sum(distribution, axis=1) * delta_v ** 2


def mean_velocity(distribution,
                  delta_v,
                  velocities,
                  particle_numbers):
    r"""Compute the mean velocites in the current distribution.

    Note
    ----
        Be aware that this must be computed separately for each single specimen.

    Parameters
    ----------
    distribution : :obj:`~numpy.ndarray` [:obj:`float`]
        A discretized velocity distribution function.
        Must be 2D array.
        For homogeneous case, add a np.newaxis.
    delta_v : :obj:`float`
        The physical spacing of the respective velocity grid.
    velocities: :obj:`~numpy.ndarray` [:obj:`float`]
        An array of all velocities.
        Each velocity is either 2 or 3 dimensional.
        Must be a 2D array.
    particle_numbers : :obj:`~numpy.ndarray` [:obj:`float`]
        Denotes the particle number for each space point.
        In the homogeneous case this as a single number.
        Must be a 1D Array.
    """
    assert distribution.ndim == 2
    means = (delta_v**2 *
             np.dot(distribution, velocities)
             / particle_numbers[:, np.newaxis])
    return means


def temperature(distribution,
                delta_v,
                velocities,
                mass,
                particle_numbers,
                mean_velocities):
    r"""Compute the mean velocites in the current distribution.

        Note
        ----
            Be aware that this must be computed separately for each single specimen.

        Parameters
        ----------
        distribution : :obj:`~numpy.ndarray` [:obj:`float`]
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
        particle_numbers : :obj:`~numpy.ndarray` [:obj:`float`]
            Denotes the particle number for each space point.
            In the homogeneous case this has a single entry.
            Must be a 1D Array.
        mean_velocities : :obj:`~numpy.ndarray` [:obj:`float`]
            Denotes the mean velocitiy for each space point.
            In the homogeneous case this has a single (2D or 3D) entry.
            Each velocity is either 2 or 3 dimensional.
            Must be a 2D Array.
        """
    assert distribution.ndim == 2
    dimension = velocities.shape[1]
    velocities = velocities[np.newaxis, ...]
    mean_velocities = mean_velocities[:, np.newaxis, :]
    factor = mass * delta_v**2 / (dimension * particle_numbers)
    deviation = np.sum((velocities - mean_velocities) ** 2,
                       axis=2)
    return factor * np.sum(deviation * distribution, axis=1)
