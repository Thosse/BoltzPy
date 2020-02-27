
import numpy as np
import boltzpy.momenta as bp_m

# Todo separate the moment functions to work on a single (spc) grid
#  and return that result, move for loop into call


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
        dv = data.dv[s_idx]
        particles = bp_m.particle_number(data.state[..., beg:end],
                                         dv)
        result[..., s_idx] = particles
    return result


def momentum_x(data):
    # shape = (position_grid.size, species.size)
    shape = (data.p_size, data.n_spc)
    result = np.zeros(shape, dtype=float)
    for (s_idx, [beg, end]) in enumerate(data.v_range):
        velocities = data.vG[beg:end, :]
        dv = data.dv[s_idx]
        particles = bp_m.particle_number(data.state[..., beg:end],
                                         dv)
        mean_velocity = bp_m.mean_velocity(data.state[..., beg:end],
                                           dv,
                                           velocities,
                                           particles)
        result[..., s_idx] = mean_velocity[..., 0]
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
        velocities = data.vG[beg:end, :]
        mass = data.m[s_idx]
        dv = data.dv[s_idx]
        particles = bp_m.particle_number(data.state[..., beg:end],
                                         dv)
        mean_velocity = bp_m.mean_velocity(data.state[..., beg:end],
                                           dv,
                                           velocities,
                                           particles)
        result[..., s_idx] = bp_m.temperature(data.state[..., beg:end],
                                              dv,
                                              velocities,
                                              mass,
                                              particles,
                                              mean_velocity)
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
