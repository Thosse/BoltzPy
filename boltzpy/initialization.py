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
    discrete_momenta = sp_newton(bp.Model._maxwellian_moments_error,
                                 desired_momenta,
                                 args=(desired_momenta,
                                       velocities,
                                       mass,
                                       delta_v))
    assert isinstance(discrete_momenta, np.ndarray)
    state = bp.Model.maxwellian(velocities,
                                mass,
                                discrete_momenta[0],
                                discrete_momenta[1: dim + 1],
                                discrete_momenta[dim + 1],
                                delta_v)
    return state
