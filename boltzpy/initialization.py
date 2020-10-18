import numpy as np
import boltzpy.output as bp_o
from scipy.optimize import newton as sp_newton


# Maxwellian (continous function)
def maxwellian(velocities,
               mass,
               input_momenta):
    # unpack momenta
    dimension = input_momenta.size - 2
    particle_number = input_momenta[0]
    mean_velocity = input_momenta[1: dimension + 1]
    temperature = input_momenta[dimension + 1]

    # compute discrete maxwellian, with continuous definition
    factor = particle_number / np.sqrt(2*np.pi * temperature / mass) ** (dimension/2)
    exponential = np.exp(- np.sum((velocities - mean_velocity) ** 2, axis=1)
                         / (2 * temperature / mass))
    return factor * exponential


def compute_initial_distribution(velocities,
                                 delta_v,
                                 mass,
                                 particle_number,
                                 mean_velocity,
                                 temperature):
    # write parameters into array
    desired_momenta = np.array([particle_number, *mean_velocity, temperature],
                               dtype=float)
    # Compute discrete Momenta with newton scheme
    discrete_momenta = sp_newton(_maxwellian_iteration,
                                 desired_momenta,
                                 args=(delta_v,
                                       velocities,
                                       mass,
                                       particle_number,
                                       mean_velocity,
                                       temperature))
    assert isinstance(discrete_momenta, np.ndarray)
    return maxwellian(velocities, mass, discrete_momenta)


def _maxwellian_iteration(input_momenta,
                          delta_v,
                          velocities,
                          mass,
                          desired_particle_number,
                          desired_mean_velocity,
                          desired_temperature):
    # create maxwellian, based on continous function
    # add axis to maxwellian, since moment functions need a 2D array
    discrete_maxwellian = maxwellian(velocities,
                                     mass,
                                     input_momenta)[np.newaxis, :]
    # compute momenta
    cmp_particle_number = bp_o.particle_number(discrete_maxwellian,
                                               delta_v)
    cmp_mean_velocity = bp_o.mean_velocity(discrete_maxwellian,
                                           delta_v,
                                           velocities,
                                           cmp_particle_number)
    cmp_temperature = bp_o.temperature(discrete_maxwellian,
                                       delta_v,
                                       velocities,
                                       mass,
                                       cmp_particle_number,
                                       cmp_mean_velocity)
    # write momenta into array, to compute difference
    # output_momenta = np.array([cmp_particle_number[0],
    #                            *cmp_mean_velocity[0],
    #                            cmp_temperature][0],
    #                           dtype=float)

    # Todo remove the ugly if case!
    # return difference from desired momenta
    if desired_mean_velocity.size == 2:
        result = np.array([cmp_particle_number[0] - desired_particle_number,
                           cmp_mean_velocity[0, 0] - desired_mean_velocity[0],
                           cmp_mean_velocity[0, 1] - desired_mean_velocity[1],
                           cmp_temperature[0] - desired_temperature])
    elif desired_mean_velocity.size == 3:
        result = np.array([cmp_particle_number[0] - desired_particle_number,
                           cmp_mean_velocity[0, 0] - desired_mean_velocity[0],
                           cmp_mean_velocity[0, 1] - desired_mean_velocity[1],
                           cmp_mean_velocity[0, 2] - desired_mean_velocity[2],
                           cmp_temperature[0] - desired_temperature])
    else:
        raise NotImplementedError
    return result   # desired_momenta - output_momenta
