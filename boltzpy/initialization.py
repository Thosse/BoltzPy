import numpy as np
import boltzpy.output as bp_o
import boltzpy as bp
from scipy.optimize import newton as sp_newton


def compute_initial_distribution(velocities,
                                 delta_v,
                                 mass,
                                 particle_number,
                                 mean_velocity,
                                 temperature):
    dim = velocities.shape[-1]
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
    state = bp.Model.maxwellian(velocities,
                                mass,
                                discrete_momenta[0],
                                discrete_momenta[1: dim + 1],
                                discrete_momenta[dim + 1],
                                delta_v)
    return state


def _maxwellian_iteration(input_momenta,
                          delta_v,
                          velocities,
                          mass,
                          desired_particle_number,
                          desired_mean_velocity,
                          desired_temperature):
    dim = velocities.shape[-1]
    # add axis to maxwellian, since moment functions need a 2D array
    state = bp.Model.maxwellian(velocities,
                                mass,
                                input_momenta[0],
                                input_momenta[1: dim + 1],
                                input_momenta[dim + 1],
                                delta_v)
    state = state[np.newaxis, :]
    # compute momenta
    c_number_density = bp_o.number_density(state, delta_v)
    c_momentum = bp_o.momentum(state, delta_v, velocities, mass)
    c_mass_density = bp_o.mass_density(c_number_density, mass)
    c_mean_velocity = bp_o.mean_velocity(c_momentum, c_mass_density)
    c_pressure = bp_o.pressure(state, delta_v, velocities, mass, c_mean_velocity)
    c_temperature = bp_o.temperature(c_pressure, c_number_density)
    # write momenta into array, to compute difference
    # output_momenta = np.array([cmp_particle_number[0],
    #                            *cmp_mean_velocity[0],
    #                            cmp_temperature][0],
    #                           dtype=float)

    # Todo remove the ugly if case!
    # return difference from desired momenta
    if desired_mean_velocity.size == 2:
        result = np.array([c_number_density[0] - desired_particle_number,
                           c_mean_velocity[0, 0] - desired_mean_velocity[0],
                           c_mean_velocity[0, 1] - desired_mean_velocity[1],
                           c_temperature[0] - desired_temperature])
    elif desired_mean_velocity.size == 3:
        result = np.array([c_number_density[0] - desired_particle_number,
                           c_mean_velocity[0, 0] - desired_mean_velocity[0],
                           c_mean_velocity[0, 1] - desired_mean_velocity[1],
                           c_mean_velocity[0, 2] - desired_mean_velocity[2],
                           c_temperature[0] - desired_temperature])
    else:
        raise NotImplementedError
    return result   # desired_momenta - output_momenta
