
import boltzpy as bp
import numpy as np
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt

file = h5py.File(bp.SIMULATION_DIR + "/phd_viscosity.hdf5", mode="a")
for ndim in [2, 3]:
    if str(ndim) not in file.keys():
        file.create_group(str(ndim))

COMPUTE = {"angle_effect": False,
           "bisection_2D": False,
           "weight_adjustment_effects": False,
           "dependency_second_angle": False,
           "range_test": False
           }
m = dict()
r = dict()
nd = dict()
mean_v = dict()
temp = dict()
# mixture
masses = [2, 3]
DEFAULT_WEIGHT = 5e3
dt = 1e-6
print("Generate Collision Models")
for ndim in [2, 3]:
    m[ndim] = bp.CollisionModel(masses,
                                [[5] * ndim, [7] * ndim],
                                0.25,
                                [6, 4],
                                np.full((len(masses), len(masses)), DEFAULT_WEIGHT),
                                )
    print(ndim, ": ", m[ndim].ncols)
    nd[ndim] = np.ones(m[ndim].nspc)
    mean_v[ndim] = np.zeros((m[ndim].nspc, m[ndim].ndim))
    temp[ndim] = np.array([2.75] * m[ndim].nspc)
print("DONE!\n\n")

# delete results that are overwritten
for key, val in COMPUTE.items():
    if val and key in file.keys():
        del file[key]


def adjust_weight_by_angle(model, weights):
    assert weights.shape == (model.ndim,)
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
        w = np.sum(p * weight_factor)
        model.collision_weights[pos] *= w
    model.update_collisions(model.collision_relations,
                            model.collision_weights)
    return


def cmp_visc(i):
    result = np.empty(m[i].ndim)
    if m[i].ndim == 3:
        result[0] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[1, 1, 1], [0, 1, -1]],
            dt=dt)
        result[1] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 1, 1], [0, 1, -1]],
            dt=dt)
        result[2] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 0, 1], [1, 0, 0]],
            dt=dt)

    elif m[i].ndim == 2:
        result[0] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[1, 1], [1, -1]],
            dt=dt)
        result[1] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 1], [1, 0]],
            dt=dt)

    return result


def cmp_heat(i):
    result = np.empty(m[i].ndim, dtype=float)
    if m[i].ndim == 3:
        result[0] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            dt=dt,
            direction=[1, 1, 1])
        result[1] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            dt=dt,
            direction=[0, 1, 1])
        result[2] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            direction=[0, 0, 1],
            dt=dt)

    elif m[i].ndim == 2:
        result[0] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            direction=[1, 1],
            dt=dt)
        result[1] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            direction=[0, 1],
            dt=dt)
    return result


print("############################################################################\n"
      "Compute Viscosity and Heat Transfer for increased weights on specific angles\n"
      "############################################################################")
if "angle_effect" not in file.keys():
    hdf_group = file.create_group("angle_effect")
    angles = dict()
    visc = dict()
    heat = dict()
    for dim in [2, 3]:
        grp = m[dim].group(m[dim].key_angle(m[dim].collision_relations))
        if dim == 3:
            grp = {key: grp[key]
                   for key in [(0, 0, 1, 0, 0, 1),
                               (0, 0, 1, 0, 1, 1),
                               (0, 0, 1, 0, 1, 4),
                               (0, 1, 1, 0, 1, 1),
                               (0, 1, 1, 1, 1, 1),
                               (1, 1, 1, 1, 1, 2)
                               ]}
        n_keys = len(grp.keys())
        angles[dim] = np.empty((n_keys + 1, dim * (dim - 1)), dtype=int)
        visc[dim] = np.empty((n_keys + 1, dim))
        heat[dim] = np.empty((n_keys + 1, dim))
        ncols = np.empty(n_keys, dtype=int)
        for idx, (key, val) in enumerate(grp.items()):
            print("\rdim = %1d    -   angle = %3d / %3d"
                  % (dim, idx, n_keys),
                  end="")
            # first element is without any changes
            ncols[idx] = val.size
            idx = idx + 1
            w_factor = 10
            m[dim].collision_weights[val] *= w_factor
            m[dim].update_collisions(m[dim].collision_relations,
                                     m[dim].collision_weights)
            angles[dim][idx] = key
            visc[dim][idx] = cmp_visc(dim)
            heat[dim][idx] = cmp_heat(dim)
            m[dim].collision_weights[val] = DEFAULT_WEIGHT
        # reset collision weights
        m[dim].collision_weights[...] = DEFAULT_WEIGHT
        m[dim].update_collisions(m[dim].collision_relations,
                                 m[dim].collision_weights)
        angles[dim][0] = [0.0]
        visc[dim][0] = cmp_visc(dim)
        heat[dim][0] = cmp_heat(dim)
        # store results in file
        file["angle_effect/angles/" + str(dim)] = angles[dim]
        file["angle_effect/visc/" + str(dim)] = visc[dim]
        file["angle_effect/heat/" + str(dim)] = heat[dim]
        file.flush()
    print("DONE!\n")
else:
    print("SKIPPED!\n")

print("######################################################\n"
      "#       compute bisection Scheme for 2D models       #\n"
      "######################################################")
atol = 1e-8
rtol = 1e-3
if "bisection_2D" not in file.keys():
    hdf_group = file.create_group("bisection_2D")
    grp = m[2].group(m[2].key_angle(m[2].collision_relations))
    for key in [(0,1), (1,1), (1,2)]:
        subgrp = hdf_group.create_group(str(key))
        # prepare initial parameters for bisection
        initial_weights = [0.1, 10]
        # store all computed viscosities here
        visc = []
        for w in initial_weights:
            m[2].collision_weights[:] = DEFAULT_WEIGHT
            m[2].collision_weights[grp[key]] *= w
            m[2].update_collisions(m[2].collision_relations,
                                   m[2].collision_weights)
            visc.append(cmp_visc(2))
        # the current weights and diffs describe the best results so far
        order = np.argsort([visc[i][1] - visc[i][0] for i in [0, 1]])
        # store all weights and viscosities here, dvided into uuper and lower bounds
        visc_lo = [visc[order[0]], ] * 2
        visc_hi = [visc[order[1]], ] * 2
        w_lo = [initial_weights[order[0]]] * 2
        w_hi = [initial_weights[order[1]]] * 2
        assert visc_lo[-1][1] - visc_lo[-1][0] < 0
        assert (visc_hi[-1][1] - visc_hi[-1][0] > 0)
        assert len(w_hi) == len(w_lo)
        assert len(visc_hi) == len(visc_lo)
        assert len(w_hi) == len(visc_lo)
        assert len(visc) == len(visc_lo)

        while True:
            w = (w_lo[-1] + w_hi[-1]) / 2
            # compute new viscosity
            m[2].collision_weights[:] = DEFAULT_WEIGHT
            m[2].collision_weights[grp[key]] *= w
            m[2].update_collisions(m[2].collision_relations,
                                   m[2].collision_weights)
            visc.append(cmp_visc(2))
            diff = visc[-1][1] - visc[-1][0]
            # update hi/lo lists
            visc_lo.append(visc_lo[-1])
            w_lo.append(w_lo[-1])
            visc_hi.append(visc_hi[-1])
            w_hi.append(w_hi[-1])
            if diff < 0:
                visc_lo[-1] = visc[-1]
                w_lo[-1] = w
            else:
                visc_hi[-1] = visc[-1]
                w_hi[-1] = w
            assert len(w_hi) == len(w_lo)
            assert len(visc_hi) == len(visc_lo)
            assert len(w_hi) == len(visc_lo)
            assert len(visc) == len(visc_lo)
            assert (visc_lo[-1][1] - visc_lo[-1][0] < 0)
            assert (visc_hi[-1][1] - visc_hi[-1][0] > 0)

            adiff = np.max(visc[-1]) - np.min(visc[-1])
            rdiff = np.max(visc[-1]) / np.min(visc[-1]) - 1
            print("Weight = ", w, "\tAbsolute: ", adiff, "\tRealtive: ", rdiff)
            if adiff < atol and rdiff < rtol:
                m[2].collision_weights[:] = DEFAULT_WEIGHT
                break
        # store results in h5py
        subgrp["w_hi"] = np.array(w_hi)
        subgrp["w_lo"] = np.array(w_lo)
        subgrp["visc_hi"] = np.array(visc_hi)
        subgrp["visc_lo"] = np.array(visc_lo)
        subgrp["visc"] = np.array(visc)
        file.flush()
    print("DONE!\n")
else:
    print("SKIPPED!\n")

print("############################################################################\n"
      "Plot: Directional viscosities for specific angles in 2D and bisection Scheme\n"
      "############################################################################")
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25))
# fig.suptitle("Influence of Collision Angles on Viscosity, Heat Transfer, and Prandtl Number",
#              fontsize=16)
axes[0].set_title(r"Weight Adjustments by Angle in a 2D Model",
                  fontsize=14)

BAR_WIDTH = 0.2
OFFSET = {2: 0.125 * np.array([-1, 1], dtype=float),
          3: 0.25 * np.array([-1, 0, 1], dtype=float)}
colors = {2: ["tab:green", "tab:blue",
              "limegreen", "cornflowerblue",
              "tab:red", ],
          3: ["tab:orange", "tab:green", "tab:blue",
              "orange", "limegreen", "cornflowerblue",
              "tab:red", ]}
labels = {2: [r"$\Lambda_1 = \lambda\left(\begin{pmatrix}1 \\ 1 \end{pmatrix}, \begin{pmatrix}1 \\ -1 \end{pmatrix}\right)$",
              r"$\Lambda_2 = \lambda\left(\begin{pmatrix}0 \\ 1 \end{pmatrix}, \begin{pmatrix}1 \\  0 \end{pmatrix}\right)$",
              "Heat Transfers"],
          3: [r"$\lambda\left(\begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}, \begin{pmatrix}0 \\ 1 \\ -1 \end{pmatrix}\right)$",
              r"$\lambda\left(\begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}, \begin{pmatrix}0 \\ 1 \\ -1 \end{pmatrix}\right)$",
              r"$\lambda\left(\begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}, \begin{pmatrix}0 \\ 1 \\ 0 \end{pmatrix}\right)$",
              "Heat Transfers"]}
axes[0].set_ylabel(r"Viscosity and Heat Transfer", fontsize=14)

print("load stored results for left plot")
angles = {dim: file["angle_effect/angles/" + str(dim)][()]
          for dim in [2, 3]}
visc = {dim: file["angle_effect/visc/" + str(dim)][()]
        for dim in [2, 3]}
heat = {dim: file["angle_effect/heat/" + str(dim)][()]
        for dim in [2, 3]}

print("create left plot")
dim = 2
a = 0   # plt ax
axes[a].set_xlabel(r"Affected Angles", fontsize=14)
pos = np.arange(angles[dim].shape[0])
twax = axes[a].twinx()
if a == 1:
    twax.set_ylabel(r"Prandtl Number", fontsize=14)
for i in range(dim):
    twax.scatter(pos + OFFSET[dim][i],
                 visc[dim][:, i] / heat[dim][:, i],
                 color=colors[dim][dim + i],
                 edgecolor="black",
                 label=labels[dim][i],
                 lw=2,
                 s=75,)
twax.set_ylim(0, 1)

axes[a].scatter([],
                [],
                color="white",
                edgecolor="black",
                label="Prandtl Numbers",
                lw=2,
                s=75)

for i in range(dim):
    axes[a].bar(x=pos + OFFSET[dim][i],
                height=visc[dim][:, i],
                width=BAR_WIDTH,
                bottom=0.0,
                color=colors[dim][i],
                label=labels[dim][i])
    cur_label = labels[dim][dim] if i == 0 else "_nolegend_"
    axes[a].plot(pos,
                 heat[dim][:, i],
                 color=colors[dim][2*dim],
                 label=cur_label,
                 marker="o",
                 markersize=6)
if dim == 2:
    ticklabels = [str(tuple(a)) for a in angles[dim]]
else:
    ticklabels = [r"$\begin{pmatrix} %1d, %1d, %1d \\ %1d, %1d, %1d \end{pmatrix}$"
                  % tuple([int(n) for n in a])
                  for a in angles[dim]]
ticklabels[0] = "Original"
axes[a].set_xticks(pos)
axes[a].set_xticklabels(ticklabels)
axes[a].legend(loc="upper right")


print("create right plot: bisection results")
axes[1].set_title(r"Applying a Biscetion Scheme to Harmonize Viscosities in 2D",
                  fontsize=14)
axes[1].set_xlabel(r"Algorithmic Time Step", fontsize=14)
axes[0].set_ylabel(r"Directional Viscosities", fontsize=14)
colors = ["tab:green", "tab:blue", "tab:green", "tab:blue",
          "limegreen", "blue"]
styles = ["solid", "dotted", "solid"]

labels = [r"$\Lambda_1$", r"$\Lambda_2$"]
max_steps = 0
for k, key in enumerate([(0,1), (1,1), (1,2)]):
    print("load stored results for key = ", key)
    subgrp = file["bisection_2D/" + str(key)]
    visc = subgrp["visc"][()]
    print("final weight = ",
          0.5 * (subgrp["w_hi"][-1] + subgrp["w_lo"][-1]))
    pos = np.arange(visc.shape[0]) + 1
    max_steps = max(max_steps, visc.shape[0])
    for j in range(visc.shape[-1]):
        label = labels[j] + r", editing $\gamma_{" + str(key) + "}$"
        axes[1].plot(pos,
                     visc[:, j],
                     color=colors[2 * k + j],
                     lw=2,
                     ls=styles[k],
                     marker="o",
                     markersize=5,
                     label=label)
axes[1].legend(loc="upper right", ncol=3)
pos = np.arange(max_steps) + 1
axes[1].set_xticks(pos[1::2])
# axes[a].legend(loc="upper right")

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity2D.pdf")


print("###################################################\n"
      "compute dependency of second angle for 3D viscosity\n"
      "###################################################\n")
key = "dependency_second_angle"
N_ANGLES = 100
if key not in file.keys():
    hdf_group = file.create_group(key)
    # prepare first_angles and unitary rotation matrices
    # each matrix can be multiplied with (x,y,0) vectors
    # to get vectors orthogonal to the respecive angle
    rotations = []
    rotations.append([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    rotations.append([[1,  0, 0],
                      [0,  1, 1],
                      [0, -1, 1]])
    rotations.append([[-1,  1, 1],
                      [-1, -1, 1],
                      [ 2,  0, 1]])
    rotations.append([[-3, -2, 1],
                      [ 0, 10, 2],
                      [ 1, -6, 3]])

    rotations = np.array(rotations, dtype=int)
    # store integer first angles, to use as keys
    first_angles = np.copy(rotations[:, :, -1])
    # change data type to float and normalize each column
    rotations = np.array(rotations, dtype=float)
    for a in range(rotations.shape[0]):
        for col in range(rotations.shape[-1]):
            rotations[a, :, col] /= np.linalg.norm(rotations[a, :, col])
    # base angles in xy plane
    xy_angles = np.zeros((N_ANGLES, 3))
    ls = np.linspace(0, 2 * np.pi, N_ANGLES)
    xy_angles[:, 0] = np.cos(ls)
    xy_angles[:, 1] = np.sin(ls)
    # compute second angles from rotating base_angles
    second_angles = np.einsum("abc, dc -> adb", rotations, xy_angles)

    for i_a, a in enumerate(first_angles):
        subgrp = hdf_group.create_group(str(tuple(a)))
        ang_visc = np.empty(N_ANGLES, dtype=float)
        for n, sa in enumerate(second_angles[i_a]):
            print("\rangle_1 = %1d / %1d,     angle_2 = %3d / %3d"
                  % (i_a + 1, first_angles.shape[0], n + 1, N_ANGLES),
                  end="")
            ang_visc[n] = m[3].cmp_viscosity(
                number_densities=nd[3],
                temperature=temp[3],
                directions=[a, sa],
                dt=dt)
        subgrp["rad_angle"] = ls
        subgrp["first_angle"] = a
        subgrp["rotations"] = rotations[i_a]
        subgrp["visc"] = ang_visc
        file.flush()


print("#################################\n"
      "Test Range of Values of Viscosity\n"
      "#################################")
NUMBER_OF_TESTS = 100000000
if COMPUTE["range_test"]:
    print("Compute Minimum and Maximum Values")
    MIN = m[3].cmp_viscosity(nd[3], temp[3], dt,
                             directions=[[0,0,1], [0,1,0]])
    MAX = m[3].cmp_viscosity(nd[3], temp[3], dt,
                             directions=[[0, 1, 1], [0, 1, -1]])
    for i_test in range(1, NUMBER_OF_TESTS):
        print("\r" + str(i_test) + "\t/  ", NUMBER_OF_TESTS, end="")
        # create two random integer angles
        orig_angles = np.random.randint(0, 1000, size=6).reshape((2, 3))
        angles = np.array(orig_angles, dtype=float)
        for i in [0, 1]:
            # if an angle is zero, replace by np.ones
            if np.all(orig_angles[i] == 0):
                orig_angles[i] = 1
                angles[i] = 1
            # normalize
            angles[i] /= np.linalg.norm(angles[i])
        # Gram Schmidt Orthogonalization
        angles[1] = angles[1] - np.sum(angles[0] * angles[1]) * angles[0]
        result = m[3].cmp_viscosity(nd[3], temp[3], dt,
                                    directions=angles)
        CONJECTURE = MIN <= result <= MAX
        if not CONJECTURE:
            print(orig_angles)
            print(angles)
            print(MIN, "\t", result, "\t", MAX, "\n\n")
    print("DONE!\n")
else:
    print("SKIPPED!\n")

print("##############################\n"
      "Plot: Angular Dependency in 3D\n"
      "##############################")
print("Create left plot Plot: effects of weight adjustments for specific  angles")
plt.cla()
plt.clf()
fig = plt.figure(constrained_layout=True, figsize=(12.75, 6.25))
axes = [plt.subplot(1, 2, a, projection=proj)
        for a, proj in zip([1,2], [None, "polar"])]
# fig.suptitle("Influence of Collision Angles on Viscosity, Heat Transfer, and Prandtl Number",
#              fontsize=16)
# axes[0].set_title(r"Weight Adjustments for Selected Angles in  3D",
#                   fontsize=14)
axes[0].text(0.5, 1.08, r"Weight Adjustments for Selected Angles in  3D",
             horizontalalignment='center',
             fontsize=14,
             transform=axes[0].transAxes)

print("load stored results for left plot")
angles = {dim: file["angle_effect/angles/" + str(dim)][()]
          for dim in [2, 3]}
visc = {dim: file["angle_effect/visc/" + str(dim)][()]
        for dim in [2, 3]}
heat = {dim: file["angle_effect/heat/" + str(dim)][()]
        for dim in [2, 3]}
BAR_WIDTH = 0.2
OFFSET = {2: 0.125 * np.array([-1, 1], dtype=float),
          3: 0.25 * np.array([-1, 0, 1], dtype=float)}
colors = {2: ["tab:green", "tab:blue",
              "limegreen", "cornflowerblue",
              "tab:red", ],
          3: ["tab:orange", "tab:green", "tab:blue",
              "orange", "limegreen", "cornflowerblue",
              "tab:red", ]}
labels = {2: [r"$\lambda\left(\begin{pmatrix}1 \\ 1 \end{pmatrix}, \begin{pmatrix}1 \\ -1 \end{pmatrix}\right)$",
              r"$\lambda\left(\begin{pmatrix}0 \\ 1 \end{pmatrix}, \begin{pmatrix}1 \\  0 \end{pmatrix}\right)$",
              "Heat Transfers"],
          3: [r"$\lambda\left(\begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}, \begin{pmatrix}0 \\ 1 \\ -1 \end{pmatrix}\right)$",
              r"$\lambda\left(\begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}, \begin{pmatrix}0 \\ 1 \\ -1 \end{pmatrix}\right)$",
              r"$\lambda\left(\begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}, \begin{pmatrix}0 \\ 1 \\ 0 \end{pmatrix}\right)$",
              "Heat Transfers"]}
axes[0].set_ylabel(r"Viscosity and Heat Transfer", fontsize=14)
a = 0
dim = 3
axes[a].set_xlabel(r"Affected Angles", fontsize=14)
pos = np.arange(angles[dim].shape[0])
twax = axes[a].twinx()
twax.set_ylabel(r"Prandtl Number", fontsize=14)
for i in range(dim):
    twax.scatter(pos + OFFSET[dim][i],
                 visc[dim][:, i] / heat[dim][:, i],
                 color=colors[dim][dim + i],
                 edgecolor="black",
                 lw=2,
                 s=75,)
twax.set_ylim(0, 1)

axes[a].scatter([],
                [],
                color="white",
                edgecolor="black",
                label="Prandtl Numbers",
                lw=2,
                s=75)

for i in range(dim):
    axes[a].bar(x=pos + OFFSET[dim][i],
                height=visc[dim][:, i],
                width=BAR_WIDTH,
                bottom=0.0,
                color=colors[dim][i],
                label=labels[dim][i])
    cur_label = labels[dim][dim] if i == 0 else "_nolegend_"
    axes[a].plot(pos,
                 heat[dim][:, i],
                 color=colors[dim][2*dim],
                 label=cur_label,
                 marker="o",
                 markersize=6)
if dim == 2:
    ticklabels = [str(tuple(a)) for a in angles[dim]]
else:
    ticklabels = [r"$\begin{pmatrix} %1d, %1d, %1d \\ %1d, %1d, %1d \end{pmatrix}$"
                  % tuple([int(n) for n in a])
                  for a in angles[dim]]
ticklabels[0] = "Original"
axes[a].set_xticks(pos)
axes[a].set_xticklabels(ticklabels)
# axes[a].legend(loc="upper right")
axes[a].legend(loc="lower center", bbox_to_anchor=(0.40, -0.35), ncol=4,
               fontsize=8)


print("Create right plot: Angle Dependency of the Viscosity in a 3D Model")
axes[1].set_title(r"Angular Dependencies of $\lambda$ in a 3D Model",
                  fontsize=14)

key = "dependency_second_angle"
styles = ["solid", "solid", "solid", "dashed"]
colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
labels = [
    r"$\lambda\left(\begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}, \bullet\right)$",
    r"$\lambda\left(\begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}, \bullet\right)$",
    r"$\lambda\left(\begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}, \bullet\right)$",
    r"$\lambda\left(\begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}, \bullet\right)$",
]
# iterate over this, to have a fixed order!
a_keys = [(1,1,1), (0,1,1), (0,0,1), (1,2,3)]
hdf_group = file[key]
for k, first_angle in enumerate(a_keys):
    subgrp = hdf_group[str(first_angle)]
    if k == 3:
        rad = subgrp["rad_angle"][()] + 0.9
    else:
        rad = subgrp["rad_angle"][()]
    visc = subgrp["visc"][()]
    axes[1].plot(rad,
                 visc,
                 ls=styles[k],
                 c=colors[k],
                 label=labels[k],
                 lw=3)
axes[1].set_rlabel_position(225)
# axes[1].set_rlim(0, 1.4)
axes[1].set_rticks([0, 0.00005, 0.0001, 0.00015])
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=4,
               fontsize=8)
axes[1].grid(linestyle="dotted")
fig.subplots_adjust(bottom=0.225)
plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity3D.pdf")


#################################################
#       compute weight_adjustment_effects       #
#################################################
N_POINTS = 100
MIN_W = 0.2
MAX_W = 4.0
print("Compute weight_adjustment_effects on viscosities")
if "weight_adjustment_effects" not in file.keys():
    hdf_group = file.create_group("weight_adjustment_effects")
    hdf_group["weights"] = np.linspace(MIN_W, MAX_W, N_POINTS)
    for dim in [2, 3]:
        ds = hdf_group.create_dataset(str(dim),
                                      (dim, N_POINTS, dim), dtype=float)
        for a in range(dim):
            new_weights = np.ones(dim, dtype=float)
            for i_w, w in enumerate(np.linspace(MIN_W, MAX_W, N_POINTS)):
                print("\rdim = %1d, a = %1d, weight = %3d / %3d"
                      % (dim, a, i_w, N_POINTS),
                      end="")
                m[dim].collision_weights[...] = DEFAULT_WEIGHT
                new_weights[a] = w
                adjust_weight_by_angle(m[dim], new_weights)
                ds[a, i_w] = cmp_visc(dim)
                file.flush()
    print("DONE!\n")
else:
    print("SKIPPED!\n")
visc = {dim: file["weight_adjustment_effects/" + str(dim)][()]
        for dim in [2, 3]}
weights = file["weight_adjustment_effects/weights"][()]


##############################################
#       plot weight_adjustment_effects       #
##############################################
print("Create Plot for weight_adjustment_effects")
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25))
fig.suptitle("Effect of Componentwise Simplified  Weight Adjustments on Directional Viscosities",
             fontsize=16)
axes[0].set_title(r"2D Collision Model",
                  fontsize=14)
axes[1].set_title(r"3D Collision Model",
                  fontsize=14)

colors = {3: np.array([["darkred", "red", "orange"],
                       ["darkgreen", "limegreen", "lime"],
                       ["navy", "blue", "dodgerblue"],])}
colors[2] = colors[3][1:, 1:]
labels = {3: np.array([["Viscosity[1,1,1]", "Viscosity[0,1,1]", "Viscosity[0,0,1]"],
                       ["Viscosity[1,1,1]", "Viscosity[0,1,1]", "Viscosity[0,0,1]"],
                       ["Viscosity[1,1,1]", "Viscosity[0,1,1]", "Viscosity[0,0,1]"]]),
          2: np.array([["Viscosity[1,1]", "Viscosity[0,1]"],
                       ["Viscosity[1,1]", "Viscosity[0,1]"]])
          }

styles = {3: np.array(["dotted", "dashed", "solid"])}
styles[2] = styles[3][1:]

axes[0].set_ylabel("Directional Viscosities", fontsize=14)
for ax, dim in enumerate([2, 3]):
    axes[ax].set_xlabel(r"Weight factor $\omega_\bullet$")
    for a in range(dim):
        for i in range(dim):
            axes[ax].plot(weights,
                          visc[dim][a, :, i],
                          color=colors[dim][a, i],
                          label=labels[dim][a, i],
                          ls=styles[dim][a],
                          lw=3,
                          zorder=5 - i)
    # create legend, with colums and per-column-title
    leg = axes[ax].get_legend_handles_labels()
    new_leg = []
    for i_l in [0,1]:
        l = np.array(leg[i_l]).reshape((dim, dim))
        new_l = np.empty((dim, dim+1), dtype=object)
        new_l[:, 1:] = l
        if i_l == 0:
            new_l[:, 0] = plt.plot([],marker="", ls="")[0]
        elif dim == 2:
            new_l[:, 0] = [r"$\boldmath{\omega = (\omega_1, 1):}$",
                           r"$\boldmath{\omega = (1, \omega_2):}$"]
        else:
            new_l[:, 0] = [r"$\boldmath{\omega = (\omega_1,1,1):}$",
                           r"$\boldmath{\omega = (1, \omega_2,1):}$",
                           r"$\boldmath{\omega = (1, 1, \omega_3):}$",]
        new_leg.append(new_l.flatten())
    axes[ax].legend(*new_leg, ncol=dim, loc="upper right", fontsize=10)
plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_adjustment_Effect.pdf")


# def harmonize_directional_viscosities(model, atol = 0)
# result = np.ones(model.ndim)
# weights = np.ones(model.ndim)
# diff = 1
# while diff > 1e-8:
#     p = np.argsort(result)
#     w = np.ones(model.ndim, dtype=float)
#     mean = np.sum(result) / result.size
#     w[p[0]] = result[p[0]] / mean
#     w[p[-1]] = result[p[-1]] / mean
#     weights *= w
#     result = cmp_visc(w, model)
#     diff = np.max(result) - np.min(result)
#     print("Difference = ", diff, "\n")
#
#
# print("Used weights = ", weights)
# tic = time()
# if model.ndim == 3:
#     viscosity_001 = model.cmp_viscosity(
#         number_densities=number_densities,
#         temperature=temperature,
#         directions=[[0, 0, 1], [1, 0, 0]],
#         dt=dt)
#     print("viscose_001 = ", "{:e}".format(viscosity_001))
#
#     viscosity_011 = model.cmp_viscosity(
#         number_densities=number_densities,
#         temperature=temperature,
#         directions=[[0, 1, 1], [0, 1, -1]],
#         dt=dt)
#     print("viscose_011 = ", "{:e}".format(viscosity_011))
#
#     # viscosity_011b = model.cmp_viscosity(
#     #     number_densities=number_densities,
#     #     temperature=temperature,
#     #     directions=[[0, 1, 1], [1, 0, 0]],
#     #     dt=dt)
#     # print("viscose_011x100 = ", "{:e}".format(viscosity_011b))
#
#
#     viscosity_111 = model.cmp_viscosity(
#         number_densities=number_densities,
#         temperature=temperature,
#         directions=[[1, 1, 1], [0, 1, -1]],
#         dt=dt)
#     print("viscose_111 = ", "{:e}".format(viscosity_111))
#
#     heat_transter_001 = model.cmp_heat_transfer(
#         number_densities=number_densities,
#         temperature=temperature,
#         direction=[0, 0, 1],
#         dt=dt)
#     print("thermal_001 = ", "{:e}".format(heat_transter_001))
#
#     heat_transter_011 = model.cmp_heat_transfer(
#         number_densities=number_densities,
#         temperature=temperature,
#         dt=dt,
#         direction=[0, 1, 1])
#     print("thermal_011 = ", "{:e}".format(heat_transter_011))
#
#     heat_transter_111 = model.cmp_heat_transfer(
#         number_densities=number_densities,
#         temperature=temperature,
#         dt=dt,
#         direction=[1, 1, 1])
#     print("thermal_111 = ", "{:e}".format(heat_transter_111))
#
#     # print("Diff_ther  = ", "{:e}".format(heat_transter_001 - heat_transter_011))
#     print()
#     print("Prandtl_001 = ", "{:e}".format(viscosity_001 / heat_transter_001))
#     print("Prandtl_011 = ", "{:e}".format(viscosity_011 / heat_transter_001))
#     print("Prandtl_111 = ", "{:e}".format(viscosity_111 / heat_transter_001))
#     print()
# elif model.ndim == 2:
#     viscosity_01 = model.cmp_viscosity(
#         number_densities=number_densities,
#         temperature=temperature,
#         directions=[[0, 1], [1, 0]],
#         dt=dt)
#     print("viscose_01 = ", "{:e}".format(viscosity_01))
#
#     viscosity_11 = model.cmp_viscosity(
#         number_densities=number_densities,
#         temperature=temperature,
#         directions=[[1, 1], [1, -1]],
#         dt=dt)
#     print("viscose_11 = ", "{:e}".format(viscosity_11))
#
#     heat_transter_01 = model.cmp_heat_transfer(
#         number_densities=number_densities,
#         temperature=temperature,
#         direction=[0, 1],
#         dt=dt)
#     print("thermal_01 = ", "{:e}".format(heat_transter_01))
#
#     heat_transter_11 = model.cmp_heat_transfer(
#         number_densities=number_densities,
#         temperature=temperature,
#         dt=dt,
#         direction=[1, 1])
#     print("thermal_11 = ", "{:e}".format(heat_transter_11))
#
#     # print("Diff_ther  = ", "{:e}".format(heat_transter_001 - heat_transter_011))
#     print()
#     print("Prandtl_01 = ", "{:e}".format(viscosity_01 / heat_transter_01))
#     print("Prandtl_11 = ", "{:e}".format(viscosity_11 / heat_transter_11))
#     print()
#
