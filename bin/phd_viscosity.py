
# Desired Command / Road Map
import boltzpy as bp
import numpy as np
import h5py
from scipy.optimize import newton as sp_newton
from time import time

file = h5py.File("exp_viscosity.hdf5", mode="a")
# mixture
NDIM = 2
masses = [2, 3]
shapes = [[5] * NDIM, [7] * NDIM]
DEFAULT_WEIGHT = 5e3
key_1 = masses.__str__()
key_2 = shapes.__str__()
if key_1 in file.keys() and key_2 in file[key_1].keys():
    model = bp.CollisionModel.load(file[key_1][key_2])
else:
    model = bp.CollisionModel(masses,
                              shapes,
                              0.25,
                              [6, 4],
                              np.full((len(masses), len(masses)), DEFAULT_WEIGHT),
                              )
    file.create_group(key_1 + "/" + key_2)
    model.save(file[key_1 + "/" + key_2])

number_densities = np.ones((model.nspc,))
mean_v = np.zeros((model.nspc, model.ndim))
temperature = np.array([2.75] * model.nspc)
dt = 1e-6

#########################################
#       angular weight adjustment       #
#########################################

# if model.ndim == 2:
#     # first weight affects 11, second 01
#     weights = np.array([1.00, 1.50])
# else:
#     # first weight affects 111, second 011, third 001
#     weights = np.array([1.2, 1.7, .475])
# grp = model.group(model.key_angle(model.collision_relations))
# model.collision_weights[grp[(0, 1)]] = weights[1] * DEFAULT_WEIGHT
# model.collision_weights[grp[(1, 1)]] = weights[0] * DEFAULT_WEIGHT
# model.update_collisions(model.collision_relations,
#                         model.collision_weights)

def prandtl_weight_adjust(weights, model):
    if model.ndim == 2:
        W_MAT = np.array([[1, 0], [-1, 1]])
    else:
        W_MAT = np.array([[1, 0, 0], [-1, 1, 0], [-1, -1, 1]])
    weight_factor = np.dot(weights, W_MAT)

    key_spc = model.key_species(model.collision_relations)[:, 1:3]
    key_angles = model.key_angle(model.collision_relations)
    angles = np.array(key_angles[:, 0:model.ndim], dtype=float)

    # for intraspecies collisions in 3D use length and height from key_angle
    if model.ndim == 3:
        is_intra = key_spc[:, 0] == key_spc[:, 1]
        angles[is_intra] = (key_angles[is_intra, 0:model.ndim]
                            + key_angles[is_intra, model.ndim:])
    angles = angles / angles[:, -1, None]

    grp = model.group(angles)
    for key, pos in grp.items():
        p = np.array(key, dtype=float)
        w = np.sum(p * weight_factor) * DEFAULT_WEIGHT
        model.collision_weights[pos] = w
    model.update_collisions(model.collision_relations,
                            model.collision_weights)
    return


def cmp_visc(weights, model):
    print("Weights = ", weights)
    prandtl_weight_adjust(weights, model)
    result = np.empty(model.ndim, dtype=float)
    if model.ndim == 3:
        result[2] = model.cmp_viscosity(
            number_densities=number_densities,
            temperature=temperature,
            directions=[[0, 0, 1], [1, 0, 0]],
            dt=dt)

        result[1] = model.cmp_viscosity(
            number_densities=number_densities,
            temperature=temperature,
            directions=[[0, 1, 1], [0, 1, -1]],
            dt=dt)

        result[0] = model.cmp_viscosity(
            number_densities=number_densities,
            temperature=temperature,
            directions=[[1, 1, 1], [0, 1, -1]],
            dt=dt)

    elif model.ndim == 2:
        result[1] = model.cmp_viscosity(
            number_densities=number_densities,
            temperature=temperature,
            directions=[[0, 1], [1, 0]],
            dt=dt)

        result[0] = model.cmp_viscosity(
            number_densities=number_densities,
            temperature=temperature,
            directions=[[1, 1], [1, -1]],
            dt=dt)
    print("Results = ", result)
    return result

result = np.ones(model.ndim)
weights = np.ones(model.ndim)
diff = 1
while diff > 1e-8:
    p = np.argsort(result)
    w = np.ones(model.ndim, dtype=float)
    mean = np.sum(result) / result.size
    w[p[0]] = result[p[0]] / mean
    w[p[-1]] = result[p[-1]] / mean
    weights = weights * w
    result = cmp_visc(weights, model)
    diff = np.max(result) - np.min(result)
    print("Difference = ", diff, "\n")

print("Used weights = ", weights)
tic = time()
if model.ndim == 3:
    viscosity_001 = model.cmp_viscosity(
        number_densities=number_densities,
        temperature=temperature,
        directions=[[0, 0, 1], [1, 0, 0]],
        dt=dt)
    print("viscose_001 = ", "{:e}".format(viscosity_001))

    viscosity_011 = model.cmp_viscosity(
        number_densities=number_densities,
        temperature=temperature,
        directions=[[0, 1, 1], [0, 1, -1]],
        dt=dt)
    print("viscose_011 = ", "{:e}".format(viscosity_011))

    # viscosity_011b = model.cmp_viscosity(
    #     number_densities=number_densities,
    #     temperature=temperature,
    #     directions=[[0, 1, 1], [1, 0, 0]],
    #     dt=dt)
    # print("viscose_011x100 = ", "{:e}".format(viscosity_011b))


    viscosity_111 = model.cmp_viscosity(
        number_densities=number_densities,
        temperature=temperature,
        directions=[[1, 1, 1], [0, 1, -1]],
        dt=dt)
    print("viscose_111 = ", "{:e}".format(viscosity_111))

    heat_transter_001 = model.cmp_heat_transfer(
        number_densities=number_densities,
        temperature=temperature,
        direction=[0, 0, 1],
        dt=dt)
    print("thermal_001 = ", "{:e}".format(heat_transter_001))

    heat_transter_011 = model.cmp_heat_transfer(
        number_densities=number_densities,
        temperature=temperature,
        dt=dt,
        direction=[0, 1, 1])
    print("thermal_011 = ", "{:e}".format(heat_transter_011))

    heat_transter_111 = model.cmp_heat_transfer(
        number_densities=number_densities,
        temperature=temperature,
        dt=dt,
        direction=[1, 1, 1])
    print("thermal_111 = ", "{:e}".format(heat_transter_111))

    # print("Diff_ther  = ", "{:e}".format(heat_transter_001 - heat_transter_011))
    print()
    print("Prandtl_001 = ", "{:e}".format(viscosity_001 / heat_transter_001))
    print("Prandtl_011 = ", "{:e}".format(viscosity_011 / heat_transter_001))
    print("Prandtl_111 = ", "{:e}".format(viscosity_111 / heat_transter_001))
    print()
elif model.ndim == 2:
    viscosity_01 = model.cmp_viscosity(
        number_densities=number_densities,
        temperature=temperature,
        directions=[[0, 1], [1, 0]],
        dt=dt)
    print("viscose_01 = ", "{:e}".format(viscosity_01))

    viscosity_11 = model.cmp_viscosity(
        number_densities=number_densities,
        temperature=temperature,
        directions=[[1, 1], [1, -1]],
        dt=dt)
    print("viscose_11 = ", "{:e}".format(viscosity_11))

    heat_transter_01 = model.cmp_heat_transfer(
        number_densities=number_densities,
        temperature=temperature,
        direction=[0, 1],
        dt=dt)
    print("thermal_01 = ", "{:e}".format(heat_transter_01))

    heat_transter_11 = model.cmp_heat_transfer(
        number_densities=number_densities,
        temperature=temperature,
        dt=dt,
        direction=[1, 1])
    print("thermal_11 = ", "{:e}".format(heat_transter_11))

    # print("Diff_ther  = ", "{:e}".format(heat_transter_001 - heat_transter_011))
    print()
    print("Prandtl_01 = ", "{:e}".format(viscosity_01 / heat_transter_01))
    print("Prandtl_11 = ", "{:e}".format(viscosity_11 / heat_transter_11))
    print()

