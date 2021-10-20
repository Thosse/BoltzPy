import numpy as np
import matplotlib.pyplot as plt
import h5py
import boltzpy as bp

N_MEANS_VELS = 200
N_TEMPS = 200
FILENAME = "exp_initial_state_estimation.hdf5"
FILE = h5py.File(FILENAME, mode="a")
model = bp.BaseModel([1],
                     [(7, 7)],
                     0.25,
                     [4]
                     )
MODEL_NAME = "{}x{}".format(*model.shapes[0])
assert model.nspc == 1, "This script is intended for a single species."

if MODEL_NAME in FILE.keys():
    COMPUTE = (input("Precomputed Solution detected for {}x{} model "
                     "Type 'redo' if you want to recompute."
                     "".format(*model.shapes[0])) == "redo")
    if COMPUTE:
        del FILE[MODEL_NAME]
else:
    COMPUTE = True
if not COMPUTE:
    h5py_group = FILE[MODEL_NAME]
else:
    h5py_group = FILE.create_group(MODEL_NAME)
    model.save(h5py_group)
    # create and save velocities to test
    h5py_group["mean_velocities"] = np.linspace(0, model.max_vel, N_MEANS_VELS)
    # ignore temperature==0
    h5py_group["temperatures"] = np.linspace(0, 1.2 * model.temperature_range(0)[1],
                                             N_TEMPS + 1)[1:]
    success = []
    fail = []
    result = np.zeros((N_MEANS_VELS, N_TEMPS), dtype=int)
    estimation = np.zeros((N_MEANS_VELS, 2))
    number_dens = np.array([1.0])
    mean_vel = np.array([[0.0, 0.0]])
    temp = np.array([0.0])
    for i, mv in enumerate(h5py_group["mean_velocities"]):
        print(i / N_MEANS_VELS, end="\r")
        mean_vel[0, 0] = mv
        estimation[i] = model.temperature_range(mean_vel)
        for j, t in enumerate(h5py_group["temperatures"]):
            temp[0] = t
            try:
                state = model.cmp_initial_state(number_dens,
                                                mean_vel,
                                                temp)
                assert np.allclose(model.cmp_number_density(state), number_dens)
                assert np.allclose(model.cmp_mean_velocity(state), mean_vel)
                assert np.allclose(model.cmp_temperature(state), temp)
                success.append([mv, t])
                result[i, j] = 1
            except ValueError:
                fail.append([mv, t])
            except AssertionError:
                fail.append([mv, t])
            except RuntimeError:
                fail.append([mv, t])
    h5py_group["success"] = np.array(success)
    h5py_group["fail"] = np.array(fail)
    h5py_group["estimation"] = estimation
    h5py_group["result"] = result
    FILE.flush()

# plot area
success = h5py_group["success"]
fail = h5py_group["fail"]
plt.scatter(success[:, 0], success[:, 1], c="blue", alpha=0.05)
plt.scatter(fail[:, 0], fail[:, 1], c="white", alpha=0.01)
# plot estimation area, by drawing the boundary
est_min = h5py_group["estimation"][:, 0]
est_max = h5py_group["estimation"][:, 1]
border = np.zeros((3 * N_MEANS_VELS, 2))
# mean velocity values
mean_vels = h5py_group["mean_velocities"][()]
border[:, 0] = np.concatenate((mean_vels, mean_vels[::-1], mean_vels))
# temperature values
border[:, 1] = np.concatenate((est_min, est_max[::-1], est_min))
plt.plot(*(border.transpose()), c="black", linewidth=3)
plt.show()
