# Desired Command / Road Map
import boltzpy as bp
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import boltzpy.helpers.TimeTracker as h_tt

file_name = "Simulations/plot_spacing_vs_ncols"

# mixture
ndim = 2
masses = np.array([11, 17], dtype=int)
spacings = np.array([2*masses[::-1],    # optimal mass-ratio
                     np.full(2, 2),     # equidistant
                     [34, 44]],           # bad choice
                    dtype=int)
min_shape = 3
max_shape = 31

file = h5py.File(file_name + ".hdf5", mode="a")
compute_results = True
if masses.__str__() in file.keys():
    print("Computed Results found!")
    old_spacings = file[masses.__str__()]["spacings"][()]
    if spacings.shape != old_spacings.shape or np.any(spacings != old_spacings):
        print("Spacings have changed!")
    old_shape_range = file[masses.__str__()]["shape_range"][()]
    if old_shape_range.size != max_shape - min_shape or np.any(old_shape_range != np.arange(min_shape, max_shape)):
        print("Shapes have changed!")
    answer = input("Compute again (yes/no)?")
    compute_results = answer == "yes"
    if compute_results:
        del file[masses.__str__()]
# compute results, if necessary
if compute_results:
    hdf_group = file.create_group(masses.__str__())
    hdf_group["spacings"] = spacings
    shape_range = hdf_group.create_dataset(
        "shape_range",
        max_shape - min_shape,
        dtype=float)
    shape_range[:] = np.arange(min_shape, max_shape)
    cols_mix = hdf_group.create_dataset(
        "cols_mix",
        (max_shape - min_shape, spacings.shape[0]),
        dtype=int)
    cols_single = hdf_group.create_dataset(
        "cols_single",
        (max_shape - min_shape, spacings.shape[0]),
        dtype=int)
    for i1, shape_x in enumerate(shape_range):
        shapes = np.full((2, ndim), shape_x)
        for i2, spacing in enumerate(spacings):
            model = bp.CollisionModel(masses,
                                      shapes,
                                      0.25,
                                      spacing,
                                      np.full((len(masses), len(masses)), 1),
                                      )
            grp = model.group(model.collision_relations, model.key_species)
            cols_single[i1, i2] = grp[(0, 0, 0, 0)].shape[0]
            key = (0, 0, 1, 1)
            if key in grp.keys():
                cols_mix[i1, i2] = grp[key].shape[0]
            else:
                cols_mix[i1, i2] = 0
        print(min_shape + i1, "\t/\t", max_shape - 1)
        file.flush()

# plot results
spacings = file[masses.__str__()]["spacings"][()]
shape_range = file[masses.__str__()]["shape_range"][()]
cols_mix = file[masses.__str__()]["cols_mix"][()]
cols_single = file[masses.__str__()]["cols_single"][()]

x_axis = shape_range
y_axis = cols_mix

# plt.loglog(x_axis, y_axis[:, 0], c='b', label=spacings[0].__str__())
# plt.loglog(x_axis, y_axis[:, 1], c="r", label=spacings[1].__str__())
# plt.loglog(x_axis, y_axis[:, 2], c="g", label=spacings[2].__str__())
# print((np.log(cols_mix[-1]) - np.log(cols_mix[-17])) / (np.log(shape_range[-1]) - np.log(shape_range[-17])) )

plt.bar(x_axis - 0.15, y_axis[:, 0], 0.30, label=spacings[0].__str__())
plt.bar(x_axis + 0.15, y_axis[:, 1], 0.30, label=spacings[1].__str__())
plt.bar(x_axis , y_axis[:, 2], 0.30, label=spacings[2].__str__())
plt.xticks(x_axis[2::5], [(s, s) for s in x_axis[2::5]])
plt.xlabel("Shape of both Grids")
plt.ylabel("Number of Interspecies Collisions")
plt.legend()
plt.savefig(file_name + ".svg")
plt.show()
