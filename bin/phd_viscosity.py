
import boltzpy as bp
import numpy as np
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

file = h5py.File(bp.SIMULATION_DIR + "/phd_viscosity.hdf5", mode="a")
for ndim in [2, 3]:
    if str(ndim) not in file.keys():
        file.create_group(str(ndim))

COMPUTE = {"angle_dependency_2D": False,
           "angle_effect_2D": False,
           "bisection_2D": False,
           "angular_invariance_2D": False,
           "weight_adjustment_effects": False,
           "angle_dependency_3D": False,
           "angle_effect_3D": False,
           "simplified_angular_weight_adjustment_effects": False,
           "heuristic": False,
           "persistence_over_altered_temperature": False,
           "multispecies": False,
           "bisection_3D": False,
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


def adjust_weight_by_angle(model, weights, relation_choice=None):
    assert weights.shape == (model.ndim,)
    if relation_choice is None:
        relations = model.collision_relations
        col_weights = model.collision_weights
    else:
        relations = model.collision_relations[relation_choice]
        col_weights = model.collision_weights[relation_choice]

    if model.ndim == 2:
        W_MAT = np.array([[1, 0], [-1, 1]])
    else:
        W_MAT = np.array([[1, 0, 0], [-1, 1, 0], [-1, -1, 1]])
    weight_factor = np.dot(weights, W_MAT)

    key_spc = model.key_species(relations)[:, 1:3]
    key_angles = model.key_angle(relations)
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
        col_weights[pos] *= w

    # apply changes to model
    model.collision_weights[relation_choice] = col_weights
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
            dt=dt,
            normalize=False)
        result[1] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 1, 1], [0, 1, -1]],
            dt=dt,
            normalize=False)
        result[2] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 0, 1], [1, 0, 0]],
            dt=dt,
            normalize=False)

    elif m[i].ndim == 2:
        result[0] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[1, 1], [1, -1]],
            dt=dt,
            normalize=False)
        result[1] = m[i].cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 1], [1, 0]],
            dt=dt,
            normalize=False)

    return result


def cmp_heat(i):
    result = np.empty(m[i].ndim, dtype=float)
    if m[i].ndim == 3:
        result[0] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            dt=dt,
            direction=[1, 1, 1],
            normalize=False)
        result[1] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            dt=dt,
            direction=[0, 1, 1],
            normalize=False)
        result[2] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            direction=[0, 0, 1],
            dt=dt,
            normalize=False)

    elif m[i].ndim == 2:
        result[0] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            direction=[1, 1],
            dt=dt,
            normalize=False)
        result[1] = m[i].cmp_heat_transfer(
            number_densities=nd[i],
            temperature=temp[i],
            direction=[0, 1],
            dt=dt,
            normalize=False)
    return result


print("#################################\n"
      "Compute Angle dependencies in 2D \n"
      "#################################\n")
key = "angle_dependency_2D"
N_ANGLES = 101
if key not in file.keys():
    hdf_group = file.create_group(key)
    hdf_group["viscosity"] = np.zeros(N_ANGLES)
    hdf_group["heat_conductivity"] = np.zeros(N_ANGLES)
    hdf_group["visc_corr"] = np.zeros(N_ANGLES)
    hdf_group["heat_corr"] = np.zeros(N_ANGLES)
    # set up orthogonal pairs of angles
    first_angles = np.zeros((N_ANGLES, 2))
    ls = np.linspace(0, 2 * np.pi, N_ANGLES)
    first_angles[:, 0] = np.cos(ls)
    first_angles[:, 1] = np.sin(ls)
    second_angles = np.zeros((N_ANGLES, 2))
    second_angles[:, 0] = -first_angles[:, 1]
    second_angles[:, 1] = first_angles[:, 0]
    # check orthogonality
    assert np.allclose(np.einsum("ij, ij -> i", first_angles, second_angles),
                       0)
    # store angles
    hdf_group["rad_angle"] = ls
    hdf_group["first_angles"] = first_angles
    hdf_group["second_angles"] = second_angles

    for i_a, eta in enumerate(first_angles):
        print("\rangle = %1d / %1d" % (i_a + 1, first_angles.shape[0]), end="")
        theta = second_angles[i_a]
        hdf_group["viscosity"][i_a] = m[2].cmp_viscosity(
            number_densities=nd[2],
            temperature=temp[2],
            directions=[eta, theta],
            dt=dt,
            normalize=False)
        hdf_group["heat_conductivity"][i_a] = m[2].cmp_heat_transfer(
            number_densities=nd[2],
            temperature=temp[2],
            direction=eta,
            dt=dt,
            normalize=False)
        # compute correction terms separately
        maxwellian = m[2].cmp_initial_state(nd[2], 0, temp[2])
        visc_mf = m[2].mf_stress(np.zeros(2),
                                 np.array([eta, theta]),
                                 orthogonalize=True)
        hdf_group["visc_corr"][i_a] = np.sum(visc_mf**2 * maxwellian)
        heat_mf = m[2].mf_heat_flow(np.zeros(2),
                                    eta,
                                    orthogonalize_state=maxwellian)
        hdf_group["heat_corr"][i_a] = np.sum(heat_mf**2 * maxwellian)
        file.flush()
    print("DONE!\n")
else:
    print("SKIPPED!\n")


print("######################################\n"
      "Plot: Angular Dependencies in 2D\n"
      "######################################")
plt.cla()
plt.clf()
fig, axes = plt.subplots(2, 2, constrained_layout=True,
                         figsize=(12.75, 13),
                         subplot_kw={"projection": "polar"})

st = fig.suptitle(r"Angular Dependencies of Flow Parameters in a 2D Model",
                  fontsize=fs_suptitle)

key = "angle_dependency_2D"
colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
labels = [
    r"Viscosity Coefficient $\widetilde{\mu}$",
    r"Heat Conductivity Coefficient $\widetilde{\kappa}$",
    r"Viscosity Correction Term",
    r"Heat Conductivity Correction Term",
]
hdf_group = file[key]
plotted_parameter = ["viscosity", "heat_conductivity", "visc_corr", "heat_corr"]
for a, ax in enumerate(axes.flatten()):
    ax.plot(hdf_group["rad_angle"][()],
            hdf_group[plotted_parameter[a]][()],
            ls="solid",
            c=colors[a],
            label=labels[a],
            lw=4)
    ax.set_rlabel_position(225)
    min_val = np.min( hdf_group[plotted_parameter[a]][()])
    max_val = np.max( hdf_group[plotted_parameter[a]][()])
    ax.set_rlim(0, max(1.18 * min_val, 1.1 * max_val))
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(linestyle="dotted")
    ax.set_title(labels[a], fontsize=fs_title)
    print("Rdiff for ", labels[a], max_val / min_val)
axes[0, 0].set_rticks([0, 0.004, 0.003, 0.002, 0.001])
plt.savefig(bp.SIMULATION_DIR + "/phd_angle_dependency_2D.pdf",
            bbox_extra_artists=(st, ),
            bbox_inches='tight')
print("Done!\n")


print("############################################################################\n"
      "Compute Viscosity and Heat Transfer for increased weights on specific angles\n"
      "############################################################################")
key = "angle_effect_2D"
if key not in file.keys():
    hdf_group = file.create_group(key)
    dim = 2
    grp = m[dim].group(m[dim].key_angle(m[dim].collision_relations))
    n_keys = len(grp.keys())
    angles = np.empty((n_keys + 1, dim * (dim - 1)), dtype=int)
    visc = np.empty((n_keys + 1, dim))
    heat = np.empty((n_keys + 1, dim))
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
        angles[idx] = key
        visc[idx] = cmp_visc(dim)
        heat[idx] = cmp_heat(dim)
        m[dim].collision_weights[val] = DEFAULT_WEIGHT
    # reset collision weights
    m[dim].collision_weights[...] = DEFAULT_WEIGHT
    m[dim].update_collisions(m[dim].collision_relations,
                             m[dim].collision_weights)
    angles[0] = [0.0]
    visc[0] = cmp_visc(dim)
    heat[0] = cmp_heat(dim)
    # store results in file
    hdf_group["angles"] = angles
    hdf_group["visc"] = visc
    hdf_group["heat"] = heat
    file.flush()

    # compute correction terms separately
    hdf_group["visc_corr"] = np.zeros(2)
    hdf_group["heat_corr"] = np.zeros(2)
    eta = np.array([[1, 1], [0, 1]], dtype=float)
    theta = np.array([[1, -1], [1, 0]], dtype=float)
    maxwellian = m[2].cmp_initial_state(nd[2], 0, temp[2])
    for i in range(2):
        visc_mf = m[2].mf_stress(np.zeros(2),
                                 np.array([eta[i], theta[i]]),
                                 orthogonalize=True)
        hdf_group["visc_corr"][i] = np.sum(visc_mf ** 2 * maxwellian)
        heat_mf = m[2].mf_heat_flow(np.zeros(2),
                                    theta[i],
                                    orthogonalize_state=maxwellian)
        hdf_group["heat_corr"][i] = np.sum(heat_mf ** 2 * maxwellian)

    file.flush()
    print("DONE!\n")
else:
    print("SKIPPED!\n")


print("############################################################################\n"
      "Plot: Feasibility of Angular Weight Adjustments in 2D\n"
      "############################################################################")
fig = plt.figure(constrained_layout=True, figsize=(12.75, 5.25))
ax = fig.add_subplot()

ax.set_title("Influence of Collision Angles on Viscosity, "
             "Heat Transfer, and Prandtl Number",
             fontsize=fs_title)

BAR_WIDTH = 0.2
OFFSET = 0.125 * np.array([-1, 1], dtype=float)
colors = ["tab:green", "tab:blue",
          "limegreen", "cornflowerblue",
          "tab:red", ]
labels = [r"$\Lambda_1$", r"$\Lambda_2$", "Heat Transfers"]

print("load stored results for left plot")
hdf_group = file["angle_effect_2D"]
angles = hdf_group["angles"][()]
visc = hdf_group["visc"][()]
heat = hdf_group["heat"][()]
visc_corr = hdf_group["visc_corr"][()]
heat_corr = hdf_group["heat_corr"][()]

dim = 2
ax.set_xlabel(r"Weight Collision Angles", fontsize=fs_label)
pos = np.arange(angles.shape[0])
twax = ax.twinx()
# twax.set_ylabel(r"Prandtl Number", fontsize=fs_label, rotation=270, labelpad=18)
for i in range(dim):
    twax.scatter(pos + OFFSET[i],
                 visc[:, i] * heat_corr[i] / (visc_corr[i] * heat[:, i]),
                 color=colors[dim + i],
                 edgecolor="black",
                 label=labels[i],
                 lw=2,
                 s=75,)
twax.set_ylim(0, 1)
twax.tick_params(axis="y", labelsize=fs_ticks)

ax.scatter([],
           [],
           color="white",
           edgecolor="black",
           label="Prandtl Numbers",
           lw=2,
           s=75)

for i in range(dim):
    ax.bar(x=pos + OFFSET[i],
           height=visc[:, i],
           width=BAR_WIDTH,
           bottom=0.0,
           color=colors[i],
           label=labels[i])
    cur_label = labels[dim] if i == 0 else "_nolegend_"
    ax.plot(pos,
            heat[:, i],
            color=colors[2*dim],
            label=cur_label,
            marker="o",
            markersize=6)

ticklabels = [str(tuple(a)) for a in angles]
ticklabels[0] = "Original"
ax.set_xticks(pos)
ax.set_xticklabels(ticklabels)
ax.legend(loc="upper right", fontsize=fs_legend+2)
ax.tick_params(axis="x", labelsize=fs_ticks)
ax.tick_params(axis="y", labelsize=fs_ticks)

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_feasibility_2D.pdf")
print("Done!\n")


print("######################################################\n"
      "#       compute bisection Scheme for 2D models       #\n"
      "######################################################")
atol = 1e-6
rtol = 1e-4
if "bisection_2D" not in file.keys():
    hdf_group = file.create_group("bisection_2D")
    grp = m[2].group(m[2].key_angle(m[2].collision_relations))
    for key in [(0,1), (1,1), (1,2)]:
        print("Angular Weight Adjustment for " + str(key))
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
            print("\rWeight = {:10e} - Absolute = {:6e} - Realtive = {:6e}"
                  "".format(w, adiff, rdiff),
                  end="")
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
        print()
    print("DONE!\n")
else:
    print("SKIPPED!\n")

print("####################################\n"
      "Plot: bisection Scheme for 2D models\n"
      "####################################")
# fig = plt.figure(constrained_layout=True, figsize=(12.75, 6.25))
# ax = fig.add_subplot()
fig, axes = plt.subplots(1, 3, constrained_layout=True,
                         sharex="all",
                         sharey="all",
                         figsize=(12.75, 5.25))

fig.suptitle("Bisecting Viscosity Coefficient with Angular Weight Adjustments",
             fontsize=fs_suptitle)

axes[0].set_ylabel(r"Directional Viscosities", fontsize=fs_label)
colors = ["tab:green", "tab:blue"]
labels = [r"$\eta = \begin{pmatrix}0\\1\end{pmatrix}$",
          r"$\eta = \begin{pmatrix}1\\1\end{pmatrix}$"]
max_steps = 0

for k, key in enumerate([(0,1), (1,1), (1,2)]):
    print("load stored results for key = ", key)
    subgrp = file["bisection_2D/" + str(key)]
    visc = subgrp["visc"][()]
    final_weight = 0.5 * (subgrp["w_hi"][-1] + subgrp["w_lo"][-1])
    print("final weight = ",
          final_weight)
    pos = np.arange(visc.shape[0]) + 1
    max_steps = max(max_steps, visc.shape[0])
    for j in [0, 1]:
        axes[k].plot(pos,
                     visc[:, j],
                     color=colors[j],
                     lw=2,
                     ls="solid",
                     marker="o",
                     markersize=4,
                     label=labels[j])
    # axes[k].plot(pos,
    #              np.abs(visc[:, 0] - visc[:, 1]),
    #              # color=colors[j],
    #              lw=2,
    #              ls="solid",
    #              marker="o",
    #              markersize=4
    #              )
    # axes[k].set_yscale("log")
    axes[k].legend(loc="upper right", fontsize=fs_legend)
    pos = np.arange(max_steps) + 1
    axes[k].set_xticks(pos[4::5])
    axes[k].tick_params(axis="x", labelsize=fs_ticks)
    axes[k].tick_params(axis="y", labelsize=fs_ticks)
    axes[k].set_title(r"$\phi = " + str(key) +
                      ",\: \gamma_\phi = " + str(final_weight)[:5] + "$",
                      fontsize=fs_title)
    axes[k].set_xlabel(r"Algorithmic Time Step", fontsize=fs_label)
    axes[k].grid(linestyle="dotted")

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_bisection_2D.pdf")
print("Done!\n")

print("#############################################################\n"
      "#       Compute Angular Dependencies on Bisected Model      #\n"
      "#############################################################")
key = "angular_invariance_2D"
if key not in file.keys():
    angle = (0, 1)
    bisection_results = file["bisection_2D/" + str(angle)]
    final_weight = 0.5 * (bisection_results["w_hi"][-1]
                          + bisection_results["w_lo"][-1])
    # adjust weights
    grp = m[2].group(m[2].key_angle(m[2].collision_relations))
    m[2].collision_weights[:] = DEFAULT_WEIGHT
    m[2].collision_weights[grp[angle]] *= final_weight
    m[2].update_collisions(m[2].collision_relations,
                           m[2].collision_weights)

    hdf_group = file.create_group(key)
    hdf_group["rad_angle"] = file["angle_dependency_2D/rad_angle"]
    hdf_group["first_angles"] = file["angle_dependency_2D/first_angles"]
    hdf_group["second_angles"] = file["angle_dependency_2D/second_angles"]
    hdf_group["viscosity"] = np.zeros(N_ANGLES)
    hdf_group["heat_conductivity"] = np.zeros(N_ANGLES)
    # set up orthogonal pairs of angles
    first_angles = hdf_group["first_angles"][()]
    second_angles = hdf_group["second_angles"] [()]

    for i_a, eta in enumerate(first_angles):
        print("\rangle = %1d / %1d" % (i_a + 1, first_angles.shape[0]), end="")
        theta = second_angles[i_a]
        hdf_group["viscosity"][i_a] = m[2].cmp_viscosity(
            number_densities=nd[2],
            temperature=temp[2],
            directions=[eta, theta],
            dt=dt,
            normalize=False)
        hdf_group["heat_conductivity"][i_a] = m[2].cmp_heat_transfer(
            number_densities=nd[2],
            temperature=temp[2],
            direction=eta,
            dt=dt,
            normalize=False)
        file.flush()
    print("DONE!\n")
else:
    print("SKIPPED!\n")
# restore original model
m[2].collision_weights[:] = DEFAULT_WEIGHT
m[2].update_collisions(m[2].collision_relations,
                       m[2].collision_weights)


print("##########################################################\n"
      "#    Plot Angular Dependencies on Bisected Model in 2D   #\n"
      "##########################################################")
plt.cla()
plt.clf()
fig, axes = plt.subplots(1, 2,
                         figsize=(12.75, 15),
                         subplot_kw={"projection": "polar"})

fig.suptitle(r"Angular Dependencies in the Adjusted 2D Model",
                  fontsize=fs_suptitle)
colors = ["tab:orange", "tab:green"]
key = "angular_invariance_2D"
hdf_group = file[key]
for a, subkey in enumerate(["viscosity", "heat_conductivity"]):
    axes[a].plot(hdf_group["rad_angle"][()],
                 hdf_group[subkey][()],
                 ls="solid",
                 c=colors[a],
                 lw=4)
    axes[a].set_rlabel_position(225)
    min_val = np.min(hdf_group[subkey][()])
    max_val = np.max(hdf_group[subkey][()])
    axes[a].set_rlim(0, max(1.18 * min_val, 1.1 * max_val))
    axes[a].tick_params(axis="both", labelsize=16)
    axes[a].grid(linestyle="dotted")
    print(subkey,
          ": diff = ", max_val - min_val,
          "rdiff = ", max_val / min_val - 1)
axes[0].set_rticks([0, 0.004, 0.003, 0.002, 0.001])
axes[1].set_rticks([0, 0.01, 0.03, 0.05, 0.07])
axes[0].set_title(r"Viscosity Coefficient $\widetilde{\mu}$",
                  fontsize=fs_title)
axes[1].set_title(r"Heat Conductivity Coefficient $\widetilde{\kappa}$",
                  fontsize=fs_title)

plt.subplots_adjust(bottom=0.575)
plt.savefig(bp.SIMULATION_DIR + "/phd_angular_invariance_2D.pdf",
            bbox_inches='tight')
print("Done!\n")


print("#################################################\n"
      "#    Print out Prandtl Numbers of adjusted DVM  #\n"
      "#################################################")
for angle in [(0, 1),  (1,1), (1,2)]:
    bisection_results = file["bisection_2D/" + str(angle)]
    final_weight = 0.5 * (bisection_results["w_hi"][-1]
                          + bisection_results["w_lo"][-1])
    # adjust weights
    grp = m[2].group(m[2].key_angle(m[2].collision_relations))
    m[2].collision_weights[:] = DEFAULT_WEIGHT
    m[2].collision_weights[grp[angle]] *= final_weight
    m[2].update_collisions(m[2].collision_relations,
                           m[2].collision_weights)

    visc = m[2].cmp_viscosity(
        number_densities=nd[2],
        temperature=temp[2],
        directions=np.eye(2),
        dt=dt,
        normalize=True)
    heat = m[2].cmp_heat_transfer(
        number_densities=nd[2],
        temperature=temp[2],
        direction=np.eye(2)[0],
        dt=dt,
        normalize=True)
    print("Prandlt", angle, " = ", visc / heat)
# restore original model
m[2].collision_weights[:] = DEFAULT_WEIGHT
m[2].update_collisions(m[2].collision_relations,
                       m[2].collision_weights)


print("Done!\n")
print("##########################################################\n"
      "#    compute Angular Dependency (2 Angles) for 3D DVM    #\n"
      "##########################################################")
key = "angle_dependency_3D"
N_ANGLES = 101
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
        subgrp["viscosity"] = np.zeros(N_ANGLES)
        subgrp["heat_conductivity"] = np.zeros(N_ANGLES)
        subgrp["visc_corr"] = np.zeros(N_ANGLES)
        subgrp["heat_corr"] = np.zeros(N_ANGLES)
        subgrp["rad_angle"] = ls
        subgrp["first_angle"] = a
        subgrp["rotations"] = rotations[i_a]
        for n, sa in enumerate(second_angles[i_a]):
            print("\rangle_1 = %1d / %1d,     angle_2 = %3d / %3d"
                  % (i_a + 1, first_angles.shape[0], n + 1, N_ANGLES),
                  end="")
            subgrp["viscosity"] [n] = m[3].cmp_viscosity(
                number_densities=nd[3],
                temperature=temp[3],
                directions=[a, sa],
                dt=dt,
                normalize=False)
            subgrp["heat_conductivity"][n] = m[3].cmp_heat_transfer(
                number_densities=nd[3],
                temperature=temp[3],
                direction=sa,
                dt=dt,
                normalize=False)
            # compute correction terms separately
            maxwellian = m[3].cmp_initial_state(nd[3], 0, temp[3])
            visc_mf = m[3].mf_stress(np.zeros(3),
                                     np.array([a, sa]),
                                     orthogonalize=True)
            subgrp["visc_corr"][i_a] = np.sum(visc_mf ** 2 * maxwellian)
            heat_mf = m[3].mf_heat_flow(np.zeros(3),
                                        sa,
                                        orthogonalize_state=maxwellian)
            subgrp["heat_corr"][i_a] = np.sum(heat_mf ** 2 * maxwellian)
            file.flush()
    print("\nDONE!\n")
else:
    print("SKIPPED!\n")

print("######################################\n"
      "Plot: Dependency of Second Angle in 3D\n"
      "######################################")
plt.cla()
plt.clf()
fig = plt.figure(constrained_layout=True, figsize=(10, 10))
ax = fig.add_subplot(projection="polar")
print("Create  plot: Angle Dependency of the Viscosity in a 3D Model")
ax.set_title(r"Angular Dependencies of the Viscosity  $\lambda$ in a 3D Model",
             fontsize=fs_title)

key = "angle_dependency_3D"
styles = ["solid", "solid", "solid", "dashed"]
colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
labels = [
    r"$\lambda\left(\begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}, \psi\right)$",
    r"$\lambda\left(\begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}, \psi\right)$",
    r"$\lambda\left(\begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}, \psi\right)$",
    r"$\lambda\left(\begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}, \psi\right)$",
]
# iterate over this, to have a fixed order!
a_keys = [(1, 1, 1), (0, 1, 1), (0, 0, 1), (1, 2, 3)]
hdf_group = file[key]
for k, first_angle in enumerate(a_keys):
    subgrp = hdf_group[str(first_angle)]
    if k == 3:
        rad = subgrp["rad_angle"][()] + 0.9
    else:
        rad = subgrp["rad_angle"][()]
    visc = subgrp["viscosity"][()]
    ax.plot(rad,
            visc,
            ls=styles[k],
            c=colors[k],
            label=labels[k],
            lw=4)
ax.set_rlabel_position(225)
# axes[1].set_rlim(0, 1.4)
ax.set_rticks([0, 0.00005, 0.0001, 0.00015])
ax.tick_params(axis="both", labelsize=16)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
          fontsize=fs_legend
          )
ax.grid(linestyle="dotted")

plt.savefig(bp.SIMULATION_DIR + "/phd_angle_dependency_3D_single.pdf")
print("Done!\n")


print("######################################\n"
      "Plot: Both Angular Dependencies in 2D\n"
      "######################################")
plt.cla()
plt.clf()
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 7),
                         subplot_kw={"projection": "polar"})

fig.suptitle(r"Angular Dependencies of Flow Parameters in a 2D Model",
                  fontsize=fs_suptitle)

key = "angle_dependency_3D"
styles = ["solid", "solid", "solid", "dashed"]
colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
labels = [
    r"$\eta = \begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}$",
    r"$\eta = \begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}$",
    r"$\eta = \begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}$",
    r"$\eta = \begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}$"
]
# iterate over this, to have a fixed order!
a_keys = [(1, 1, 1), (0, 1, 1), (0, 0, 1), (1, 2, 3)]
hdf_group = file[key]
for k, first_angle in enumerate(a_keys):
    subgrp = hdf_group[str(first_angle)]
    if k == 3:
        rad = subgrp["rad_angle"][()] + 0.9
    else:
        rad = subgrp["rad_angle"][()]
    for a, coeff in enumerate(["viscosity", "heat_conductivity"]):
        axes[a].plot(rad,
                     subgrp[coeff][()],
                     ls=styles[k],
                     c=colors[k],
                     label=labels[k] if a == 0 else "_nolegend_",
                     lw=4)
        if a == 1:
            max_val = np.max(subgrp[coeff][()])
            axes[a].set_rticks([0, 0.005, 0.01, 0.015, 0.02])
            axes[a].set_rlim(0, 1.15 * max_val)
        else:
            axes[a].set_rticks([0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125])
for ax in axes:
    ax.set_rlabel_position(225)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(linestyle="dotted")

fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
           fontsize=fs_legend
           )

plt.savefig(bp.SIMULATION_DIR + "/phd_angle_dependency_3D_both.pdf",
            bbox_inches='tight')
print("Done!\n")


print("########################################################\n"
      "#    Compute Feasibility for Angular Weights in 3D     #\n"
      "########################################################")
key = "angle_effect_3D"
if key not in file.keys():
    hdf_group = file.create_group(key)
    dim = 3
    grp = m[dim].group(m[dim].key_angle(m[dim].collision_relations))
    grp = {key: grp[key]
           for key in [(0, 0, 1, 0, 0, 1),
                       (0, 0, 1, 0, 1, 1),
                       (0, 0, 1, 0, 1, 4),
                       (0, 1, 1, 0, 1, 1),
                       (0, 1, 1, 1, 1, 1),
                       (1, 1, 1, 1, 1, 2)
                       ]}
    n_keys = len(grp.keys())
    angles = np.empty((n_keys + 1, dim * (dim - 1)), dtype=int)
    visc = np.empty((n_keys + 1, dim))
    heat = np.empty((n_keys + 1, dim))
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
        angles[idx] = key
        visc[idx] = cmp_visc(dim)
        heat[idx] = cmp_heat(dim)
        m[dim].collision_weights[val] = DEFAULT_WEIGHT
    # reset collision weights
    m[dim].collision_weights[...] = DEFAULT_WEIGHT
    m[dim].update_collisions(m[dim].collision_relations,
                             m[dim].collision_weights)
    angles[0] = [0.0]
    visc[0] = cmp_visc(dim)
    heat[0] = cmp_heat(dim)
    # store results in file
    hdf_group["angles"] = angles
    hdf_group["visc"] = visc
    hdf_group["heat"] = heat
    file.flush()

    # compute correction terms separately
    hdf_group["visc_corr"] = np.zeros(dim)
    hdf_group["heat_corr"] = np.zeros(dim)
    eta = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=float)
    theta = np.array([[0, 1, -1], [0, 1, -1], [1, 0, 0]], dtype=float)
    maxwellian = m[dim].cmp_initial_state(nd[dim], 0, temp[dim])
    for i in range(dim):
        visc_mf = m[dim].mf_stress(np.zeros(dim),
                                   np.array([eta[i], theta[i]]),
                                   orthogonalize=True)
        hdf_group["visc_corr"][i] = np.sum(visc_mf ** 2 * maxwellian)
        heat_mf = m[dim].mf_heat_flow(np.zeros(dim),
                                      theta[i],
                                      orthogonalize_state=maxwellian)
        hdf_group["heat_corr"][i] = np.sum(heat_mf ** 2 * maxwellian)

    print("\nDONE!\n")
else:
    print("SKIPPED!\n")

print("###########################################\n"
      "Plot: Feasibility for Angular Weights in 3D\n"
      "###########################################")
print("Create  Plot: effects of weight adjustments for specific  angles")
plt.cla()
plt.clf()
fig = plt.figure(constrained_layout=True, figsize=(12.75, 6.25))
ax = fig.add_subplot()
# fig.suptitle("Influence of Collision Angles on Viscosity, Heat Transfer, and Prandtl Number",
#              fontsize=16)
ax.set_title(r"Weight Adjustments for Selected Angles in  3D",
             fontsize=fs_title)
# ax.text(0.5, 1.08, r"Weight Adjustments for Selected Angles in  3D",
#              horizontalalignment='center',
#              fontsize=20,
#              transform=ax.transAxes)

print("load stored results for left plot")
hdf_group = file["angle_effect_3D"]
angles = hdf_group["angles"][()]
visc = hdf_group["visc"][()]
heat = hdf_group["heat"][()]
visc_corr = hdf_group["visc_corr"][()]
heat_corr = hdf_group["heat_corr"][()]

BAR_WIDTH = 0.2
OFFSET = 0.25 * np.array([-1, 0, 1], dtype=float)
colors = ["tab:green", "tab:blue", "tab:orange",
          "limegreen", "cornflowerblue", "orange",
          "tab:red"]
labels = [r"$\Lambda_1$", r"$\Lambda_2$", r"$\Lambda_3$", "Heat Transfers"]
ax.set_ylabel(r"Viscosity and Heat Transfer", fontsize=fs_label)
dim = 3
ax.set_xlabel(r"Affected Angles", fontsize=fs_label)
pos = np.arange(angles.shape[0])
twax = ax.twinx()
twax.set_ylabel(r"Prandtl Number", fontsize=fs_label,
                # rotation=270, labelpad=20
                )
for i in range(dim):
    twax.scatter(pos + OFFSET[i],
                 visc[:, i] * heat_corr[i] / (visc_corr[i] * heat[:, i]),
                 color=colors[i],
                 edgecolor="black",
                 lw=2,
                 s=75,)
twax.set_ylim(0, 1)

ax.scatter([],
           [],
           color="white",
           edgecolor="black",
           label="Prandtl Numbers",
           lw=2,
           s=75)

for i in range(dim):
    ax.bar(x=pos + OFFSET[i],
           height=visc[:, i],
           width=BAR_WIDTH,
           bottom=0.0,
           color=colors[i],
           label=labels[i])
    cur_label = labels[3] if i == 0 else "_nolegend_"
    ax.plot(pos,
            heat[:, i],
            color=colors[2*dim],
            label=cur_label,
            marker="o",
            markersize=6)
ticklabels = [(r"$\begin{pmatrix}"
               + "{}, {}, {}".format(*[int(a) for a in angle_pair[:3]])
               + r"\\"
               + "{}, {}, {}".format(*[int(a) for a in angle_pair[3:]])
               + r"\end{pmatrix}$")
              for angle_pair in angles]
ticklabels[0] = "Original"
ax.set_xticks(pos)
ax.set_xticklabels(ticklabels)
ax.tick_params(axis="both", labelsize=16)
ax.legend(loc="lower center", bbox_to_anchor=(0.50, -0.335), ncol=5,
          fontsize=fs_legend)

plt.savefig(bp.SIMULATION_DIR + "/phd_angle_effect_3D.pdf")
print("Done!\n")


print("############################################################\n"
      "#    Compute effects of simplified weights adjustments     #\n"
      "############################################################")
N_POINTS = 100
MIN_W = 0.2
MAX_W = 4.0

key = "simplified_angular_weight_adjustment_effects"
if key not in file.keys():
    hdf_group = file.create_group(key)
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

print("####################################################\n"
      "     Create Plot for weight_adjustment_effects      \n"
      "####################################################")
key = "simplified_angular_weight_adjustment_effects"
visc = {dim: file[key][str(dim)][()]
        for dim in [2, 3]}
weights = file[key + "/weights"][()]


fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25))
fig.suptitle("Effect of Componentwise Simplified  Weight Adjustments on Directional Viscosities",
             fontsize=fs_suptitle)
axes[0].set_title(r"2D Collision Model",
                  fontsize=fs_title)
axes[1].set_title(r"3D Collision Model",
                  fontsize=fs_title)

colors = {3: np.array([["darkred", "red", "orange"],
                       ["darkgreen", "limegreen", "lime"],
                       ["navy", "blue", "dodgerblue"],])}
colors[2] = colors[3][1:, 1:]
labels = {3: np.array([[r"$\Lambda_1$", r"$\Lambda_2$", r"$\Lambda_3$"],
                       [r"$\Lambda_1$", r"$\Lambda_2$", r"$\Lambda_3$"],
                       [r"$\Lambda_1$", r"$\Lambda_2$", r"$\Lambda_3$"]]),
          2: np.array([[r"$\Lambda_1$", r"$\Lambda_2$"],
                       [r"$\Lambda_1$", r"$\Lambda_2$"]])
          }

styles = {3: np.array(["dotted", "dashed", "solid"])}
styles[2] = styles[3][1:]

axes[0].set_ylabel("Directional Viscosities", fontsize=fs_label)
for ax, dim in enumerate([2, 3]):
    axes[ax].set_xlabel(r"Weight factor $\omega_\bullet$", fontsize=fs_label)
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
            new_l[:, 0] = [r"$\boldmath{\omega_1:}$",
                           r"$\boldmath{\omega_2:}$"]
        else:
            new_l[:, 0] = [r"$\boldmath{\omega_1:}$",
                           r"$\boldmath{\omega_2:}$",
                           r"$\boldmath{\omega_3:}$"]
        new_leg.append(new_l.flatten())
    axes[ax].legend(*new_leg, ncol=dim, loc="upper right",
                    # fontsize=fs_legend
                    )
plt.savefig(bp.SIMULATION_DIR + "/phd_simplified_angular_weight_adjustment_effects.pdf")
print("Done!\n")


print("######################################################\n"
      "Harmonize 3D Model, and compute its angle dependencies\n"
      "######################################################\n")
key = "heuristic"
atol = 1e-6
rtol= 1e-3
if key not in file.keys():
    hdf_group = file.create_group(key)
    for dim in [2,3]:
        print("Harmonize %1dD model..." % dim)
        subgrp = hdf_group.create_group(str(dim))
        m[dim].collision_weights[...] = DEFAULT_WEIGHT
        visc = [cmp_visc(dim)]
        weights = [np.ones(dim)]
        ctr = 0
        while True:
            pos = np.argsort(visc[-1])
            # w = np.ones(dim, dtype=float)
            mean = np.sum(visc[-1]) / dim
            # w[pos[0]] = visc[-1][pos[0]] / mean
            # w[pos[-1]] = visc[-1][pos[-1]] / mean
            w = visc[-1] / mean
            weights.append(weights[-1] * w)
            # reset previous weight adjustments, to keep weights linear
            m[dim].collision_weights[...] = DEFAULT_WEIGHT
            adjust_weight_by_angle(m[dim], weights[-1])
            visc.append(cmp_visc(dim))
            adiff = np.max(visc[-1]) - np.min(visc[-1])
            rdiff = np.max(visc[-1]) / np.min(visc[-1]) - 1
            print("\rweights = ", ["{:3e}".format(w) for w in weights[-1]],
                  " - diffs = ({:3e}, {:3e})".format(adiff, rdiff),
                  " - i = ", ctr,
                  end="")
            ctr += 1
            if adiff < atol and rdiff < rtol:
                break
        # store results in h5py
        subgrp["weights"] = np.array(weights)
        subgrp["visc"] = np.array(visc)
        print("\nDONE!\n")

        if dim == 3:
            print("compute angular dependecies for adjusted 3D model ")
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
                ang_heat = np.empty(N_ANGLES, dtype=float)
                for n, sa in enumerate(second_angles[i_a]):
                    print("\rangle_1 = %1d / %1d,     angle_2 = %3d / %3d"
                          % (i_a + 1, first_angles.shape[0], n + 1, N_ANGLES),
                          end="")
                    ang_visc[n] = m[3].cmp_viscosity(
                        number_densities=nd[3],
                        temperature=temp[3],
                        directions=[a, sa],
                        dt=dt,
                        normalize=False)
                    ang_heat[n] = m[3].cmp_heat_transfer(
                        number_densities=nd[3],
                        temperature=temp[3],
                        direction=sa,
                        dt=dt,
                        normalize=False)
                subgrp["rad_angle"] = ls
                subgrp["first_angle"] = a
                subgrp["rotations"] = rotations[i_a]
                subgrp["angular_visc"] = ang_visc
                subgrp["angular_heat"] = ang_heat
                file.flush()
            print("\nDONE!\n")
else:
    print("SKIPPED!\n")


print("#########################\n"
      "Plot 3D Heuristic results\n"
      "#########################")
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25))
fig.suptitle(r"Heuristic Harmonization of the Directional Viscosities $\Lambda$",
             fontsize=fs_suptitle)
axes[0].set_title(r"$\Lambda$ and  $\omega$ in a 2D Model",
                  fontsize=fs_title)
axes[1].set_title(r"$\Lambda$ and $\omega$ in a 3D Model",
                  fontsize=fs_title)

colors = {2: ["tab:green", "tab:blue" ],
          3: ["tab:orange", "tab:green", "tab:blue"]}
labels = {2: [r"$\Lambda_1$",
              r"$\Lambda_2$"],
          3: [r"$\Lambda_1$",
              r"$\Lambda_2$",
              r"$\Lambda_3$"]}
axes[0].set_ylabel(r"Directional Viscosities", fontsize=fs_label)

for a, dim in enumerate([2, 3]):
    visc = file["heuristic"][str(dim)]["visc"]
    weights =  file["heuristic"][str(dim)]["weights"]
    pos = np.arange(visc.shape[0]) + 1
    axes[a].set_xlabel(r"Algorithmic Time Step", fontsize=fs_label)
    twax = axes[a].twinx()
    if a == 1:
        twax.set_ylabel(r"Weight Factor $\omega$", fontsize=fs_label)
    for i in range(dim):
        axes[a].plot(pos,
                     visc[:, i],
                     color=colors[dim][i],
                     label=labels[dim][i],
                     marker="o",
                     markersize=6)
        twax.plot(pos,
                  weights[:, i],
                  color=colors[dim][i],
                  ls="dotted",
                  markersize=6)
    for i in range(dim):
        axes[a].plot([], [],
                     color=colors[dim][i],
                     label=r"$\omega_{%1d}$" % (i + 1),
                     ls="dotted",
                     markersize=6)
        twax.set_ylim(0, int(np.max(weights)) + 1)

    axes[a].set_xticks(pos[::2])
    axes[a].legend(loc="upper right", ncol=2, fontsize=fs_legend)

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_heuristic.pdf")
print("Done!\n")


print("######################################################\n"
      "Plot: Dependency of Second Angle in adjusted  3D Model\n"
      "######################################################")
plt.cla()
plt.clf()
fig = plt.figure(constrained_layout=True, figsize=(10, 10))
ax = fig.add_subplot(projection="polar")
ax.set_title(r"Angular Dependencies of the Viscosity $\lambda$ in the Adjusted 3D Model",
             fontsize=fs_title)

key = "heuristic"
styles = ["solid", "solid", "solid", "dashed"]
colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
labels = [
    r"$\lambda\left(\begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}, \psi\right)$",
    r"$\lambda\left(\begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}, \psi\right)$",
    r"$\lambda\left(\begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}, \psi\right)$",
    r"$\lambda\left(\begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}, \psi\right)$",
]
# iterate over this, to have a fixed order!
a_keys = [(1,1,1), (0,1,1), (0,0,1), (1,2,3)]
hdf_group = file[key]
for k, first_angle in enumerate(a_keys):
    subgrp = hdf_group[str(first_angle)]
    visc = subgrp["angular_visc"][()]
    if k == 3:
        rad = subgrp["rad_angle"][()] + 0.9
    else:
        rad = subgrp["rad_angle"][()]
    ax.plot(rad,
            visc,
            ls=styles[k],
            c=colors[k],
            label=labels[k],
            lw=4)
ax.set_rlabel_position(225)
# axes[1].set_rlim(0, 1.4)
ax.set_rticks([0, 0.00005, 0.0001, 0.00015])
ax.tick_params(axis="both", labelsize=16)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
          fontsize=fs_legend)
ax.grid(linestyle="dotted")

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity3D_heuristic_angle_dependency.pdf")
print("Done!\n")


print("#####################################################################\n"
      "   Compute harmonized Angular viscosities for altered temperatures\n"
      "#####################################################################")
N_TEMPS = 50
key = "persistence_over_altered_temperature"
if key not in file.keys():
    hdf_group = file.create_group(key)
    # copy old parameters for viscosity
    nd_orig = nd.copy()
    mean_v_orig = mean_v.copy()
    temp_orig = temp.copy()
    for dim in [2,3]:
        print("\nCheck %1dD model..." % dim)
        subgrp = hdf_group.create_group(str(dim))
        m[dim].collision_weights[...] = DEFAULT_WEIGHT
        weights = file["heuristic"][str(dim)]["weights"][-1]
        adjust_weight_by_angle(m[dim], weights)
        temp_range = m[dim].temperature_range(atol=0.1)
        temperatures = np.linspace(*temp_range, N_TEMPS)
        visc = np.zeros((N_TEMPS, dim))
        for i_t, t in enumerate(temperatures):
            print("\r%3d / %3d" % (i_t+1, N_TEMPS), end="")
            temp[dim][...] = t
            visc[i_t] = cmp_visc(dim)
        subgrp["temperatures"] = temperatures
        subgrp["visc"] = visc
    print("\nDONE!\n")
else:
    print("SKIPPED!\n")

print("########################################################################\n",
      "#     Plot harmonized angular viscosities for altered temperatures     #\n",
      "########################################################################")
print("Create Plot for weight_adjustment_effects")
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25))
fig.suptitle("Angular Viscosities of Equalized DVM for Altered Temperatures",
             fontsize=fs_suptitle)
axes[0].set_title(r"2D Collision Model",
                  fontsize=fs_title)
axes[1].set_title(r"3D Collision Model",
                  fontsize=fs_title)

colors = {3: np.array(["tab:red",
                       "tab:green",
                       "tab:blue"])}
colors[2] = colors[3][1:]
labels = {3: np.array([r"$\Lambda_1$", r"$\Lambda_2$", r"$\Lambda_3$"]),
          2: np.array([r"$\Lambda_1$", r"$\Lambda_2$"])
          }

styles = {3: np.array(["dotted", "dashed", "solid"])}
styles[2] = styles[3][1:]

axes[0].set_ylabel("Directional Viscosities", fontsize=fs_label)

key = "persistence_over_altered_temperature"
hdf_group = file[key]
for ax, dim in enumerate([2, 3]):
    axes[ax].set_xlabel(r"Temperature", fontsize=fs_label)
    for i in range(dim):
        temperatures = hdf_group[str(dim)]["temperatures"][()]
        visc = hdf_group[str(dim)]["visc"][:, i]
        axes[ax].plot(temperatures,
                      visc,
                      color=colors[dim][i],
                      label=labels[dim][i],
                      ls=styles[dim][i],
                      lw=3)
    axes[ax].set_ylim(0, 1.5 * np.max(visc))
    axes[ax].tick_params(axis="both", labelsize=fs_ticks)
    axes[ax].legend(loc="upper right", fontsize=fs_legend)

plt.savefig(bp.SIMULATION_DIR + "/phd_persistence_over_altered_temperature.pdf")
print("Done!\n")


print("##################################################################\n"
      "#   Multispecies: Apply 3D-Heuristic Species-Wise to 3-Mixture   #\n"
      "##################################################################")

def cmp_visc_ext(model, nd, temp):
    result = np.empty(3)
    assert model.ndim == 3
    result[0] = model.cmp_viscosity(
        number_densities=nd,
        temperature=temp,
        directions=[[1, 1, 1], [0, 1, -1]],
        dt=dt)
    result[1] = model.cmp_viscosity(
        number_densities=nd,
        temperature=temp,
        directions=[[0, 1, 1], [0, 1, -1]],
        dt=dt)
    result[2] = model.cmp_viscosity(
        number_densities=nd,
        temperature=temp,
        directions=[[0, 0, 1], [1, 0, 0]],
        dt=dt)
    return result


atol = 1e-6
rtol = 1e-3
MAX_ITER = 250
key = "multispecies"
if key not in file.keys():
    hdf_group = file.create_group(key)
    print("Generate Collision Model")
    masses = [2,3,4]
    ndim = 3
    model = bp.CollisionModel(masses,
                              [[5] * ndim, [7] * ndim, [7] * ndim],
                              0.125,
                              [12, 8, 6],
                              np.full((len(masses), len(masses)), DEFAULT_WEIGHT),
                              )
    print(model.ncols, " total collisions")
    print("DONE!\n")
    sup_grp = model.group((model.key_species(model.collision_relations)[:, 1:3]))
    print("Apply Heuristic specieswise")
    # process intraspecies collisions first,
    # then harmonize interspecies collisions based on harmonized Single Species
    spc_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for (s1, s2) in spc_pairs:
        print("Harmonize (%1d, %1d)-Collisions..." % (s1, s2))
        subgrp = hdf_group.create_group(str((s1, s2)))
        # setuo new initial states
        nd3 = np.zeros(3)
        nd3[[s1, s2]] = 1
        temp3 = np.array([2.25] * 3)
        sm = model.submodel(list({s1, s2}))
        grp = sm.group((sm.key_species(sm.collision_relations)[:, 1:3]))
        if s1 == s2:
            cur_spc = (0,0)
        else:
            cur_spc = (0, 1)
        cur_iter = 0
        weights = [np.ones(sm.ndim)]
        visc = []
        while True:
            # apply new collision weights
            sm.collision_weights[grp[cur_spc]] = DEFAULT_WEIGHT
            adjust_weight_by_angle(sm, weights[-1], grp[cur_spc])

            # compute viscosity
            visc.append(cmp_visc_ext(sm,
                                     nd3[list({s1, s2})],
                                     temp3[list({s1, s2})]))

            # compute weight factor for new weight
            mean_visc = np.sum(visc[-1]) / dim
            w_factor = visc[-1] / mean_visc

            # check tolerance break conditions
            adiff = np.max(visc[-1]) - np.min(visc[-1])
            rdiff = np.max(visc[-1]) / np.min(visc[-1]) - 1
            print("\rw = [{:3e}, {:3e}, {:3e}] - MaxChange = {:3e} - "
                  "diffs = [{:6e} {:6e}] - i = {}"
                  "".format(*(weights[-1]),
                            np.max(np.abs(w_factor - 1)),
                            adiff,
                            rdiff,
                            cur_iter),
                  end="")
            if adiff < atol and rdiff < rtol:
                break
            if cur_iter > MAX_ITER:
                raise ValueError("Harmonization of DVM failed!")
            # adjust weights, multiply new correction factor
            weights.append(w_factor * weights[-1])
            cur_iter += 1

        # apply weights to model
        adjust_weight_by_angle(model, weights[-1], sup_grp[(s1, s2)])

        # store results in h5py
        subgrp["weights"] = np.array(weights)
        subgrp["visc"] = np.array(visc)
        file.flush()
        print("\nFinal Weights = ", weights[-1])
        print("Directional Viscosities = ", visc[-1], "\n")
    print("DONE!\n")
else:
    print("SKIPPED!\n")



# print("Compute viscosities for complete mixture")
# print("Restore original weights")
# # print this should be unnecessary, only weights should have changed
# model.collision_relations = hdf_group["Collisions"][()]
# model.collision_weights = np.full(model.collision_relations.shape[0],
#                                   DEFAULT_WEIGHT)
# print("apply computed weight adjustments, per species pair")
# for (s1, s2) in spc_pairs:
#     weights = hdf_group[str((s1, s2))]["weights"][-1]
#     col_choice = grp[(s1, s2)]
#     adjust_weight_by_angle(model, weights, col_choice)
# model.update_collisions(model.collision_relations,
#                         model.collision_weights)
# # compute viscosity
# nd3 = np.ones(3)
# temp3 = np.array([2.25] * 3)
# hdf_group["Viscosities"] = cmp_visc_ext(model, nd3, temp3)
# file.flush()
# print(hdf_group["Viscosities"][()])
#
# print("################################################################\n"
#       "harmonize 3D model with bisection and compute angular dependency\n"
#       "################################################################\n")
# key = "bisection_3D"
# atol = 1e-8
# rtol= 1e-3
# dim = 3
#
# if key not in file.keys():
#     hdf_group = file.create_group(key)
#     for i_w in range(3):
#         subgrp = hdf_group.create_group(str(i_w))
#         if i_w == 2:
#             weights = [0.1, 3.0]
#         else:
#             weights = [0.1, 10.0]
#
#         visc = []
#         for val in weights:
#             m[3].collision_weights[:] = DEFAULT_WEIGHT
#             new_w = np.ones(3)
#             new_w[i_w] = val
#             adjust_weight_by_angle(m[3], new_w)
#             m[3].update_collisions(m[3].collision_relations,
#                                    m[3].collision_weights)
#             visc.append(cmp_visc(3))
#         # the current weights and diffs describe the best results so far
#         print([visc[i][2] - visc[i][1] for i in [0, 1]])
#         order = np.argsort([visc[i][2] - visc[i][1] for i in [0, 1]])
#         # store all weights and viscosities here, divided into upper and lower bounds
#         visc_lo = [visc[order[0]], ] * 2
#         visc_hi = [visc[order[1]], ] * 2
#         w_lo = [weights[order[0]]] * 2
#         w_hi = [weights[order[1]]] * 2
#         assert visc_lo[-1][2] - visc_lo[-1][1] < 0
#         assert (visc_hi[-1][2] - visc_hi[-1][1] > 0)
#         assert len(w_hi) == len(w_lo)
#         assert len(visc_hi) == len(visc_lo)
#         assert len(w_hi) == len(visc_lo)
#         assert len(visc) == len(visc_lo)
#
#         while True:
#             new_w = np.ones(3)
#             new_w[i_w] = (w_lo[-1] + w_hi[-1]) / 2
#             # compute new viscosity
#             m[3].collision_weights[:] = DEFAULT_WEIGHT
#             adjust_weight_by_angle(m[3], new_w)
#             m[3].update_collisions(m[3].collision_relations,
#                                    m[3].collision_weights)
#             visc.append(cmp_visc(3))
#             diff = visc[-1][2] - visc[-1][1]
#             # update hi/lo lists
#             visc_lo.append(visc_lo[-1])
#             w_lo.append(w_lo[-1])
#             visc_hi.append(visc_hi[-1])
#             w_hi.append(w_hi[-1])
#             if diff < 0:
#                 visc_lo[-1] = visc[-1]
#                 w_lo[-1] = new_w[i_w]
#             else:
#                 visc_hi[-1] = visc[-1]
#                 w_hi[-1] = new_w[i_w]
#             assert len(w_hi) == len(w_lo)
#             assert len(visc_hi) == len(visc_lo)
#             assert len(w_hi) == len(visc_lo)
#             assert len(visc) == len(visc_lo)
#             assert (visc_lo[-1][2] - visc_lo[-1][1] < 0)
#             assert (visc_hi[-1][2] - visc_hi[-1][1] > 0)
#
#             adiff = np.max(visc[-1]) - np.min(visc[-1])
#             rdiff = np.max(visc[-1]) / np.min(visc[-1]) - 1
#             print("Weight = ", new_w[i_w], "\tAbsolute: ",
#                   adiff, "\tRealtive: ", rdiff)
#             if adiff < atol and rdiff < rtol:
#                 break
#         # store results in h5py
#         subgrp["w_hi"] = np.array(w_hi)
#         subgrp["w_lo"] = np.array(w_lo)
#         subgrp["visc_hi"] = np.array(visc_hi)
#         subgrp["visc_lo"] = np.array(visc_lo)
#         subgrp["visc"] = np.array(visc)
#         file.flush()
#
#         print("compute angular dependecies for adjusted model")
#         rotations = []
#         rotations.append([[1, 0, 0],
#                           [0, 1, 0],
#                           [0, 0, 1]])
#         rotations.append([[1,  0, 0],
#                           [0,  1, 1],
#                           [0, -1, 1]])
#         rotations.append([[-1,  1, 1],
#                           [-1, -1, 1],
#                           [ 2,  0, 1]])
#         rotations.append([[-3, -2, 1],
#                           [ 0, 10, 2],
#                           [ 1, -6, 3]])
#
#         rotations = np.array(rotations, dtype=int)
#         # store integer first angles, to use as keys
#         first_angles = np.copy(rotations[:, :, -1])
#         # change data type to float and normalize each column
#         rotations = np.array(rotations, dtype=float)
#         for a in range(rotations.shape[0]):
#             for col in range(rotations.shape[-1]):
#                 rotations[a, :, col] /= np.linalg.norm(rotations[a, :, col])
#         # base angles in xy plane
#         xy_angles = np.zeros((N_ANGLES, 3))
#         ls = np.linspace(0, 2 * np.pi, N_ANGLES)
#         xy_angles[:, 0] = np.cos(ls)
#         xy_angles[:, 1] = np.sin(ls)
#         # compute second angles from rotating base_angles
#         second_angles = np.einsum("abc, dc -> adb", rotations, xy_angles)
#
#         for i_a, a in enumerate(first_angles):
#             ssubgrp = subgrp.create_group(str(tuple(a)))
#             ang_visc = np.empty(N_ANGLES, dtype=float)
#             for n, sa in enumerate(second_angles[i_a]):
#                 print("\rangle_1 = %1d / %1d,     angle_2 = %3d / %3d"
#                       % (i_a + 1, first_angles.shape[0], n + 1, N_ANGLES),
#                       end="")
#                 ang_visc[n] = m[3].cmp_viscosity(
#                     number_densities=nd[3],
#                     temperature=temp[3],
#                     directions=[a, sa],
#                     dt=dt)
#             ssubgrp["rad_angle"] = ls
#             ssubgrp["first_angle"] = a
#             ssubgrp["rotations"] = rotations[i_a]
#             ssubgrp["angular_visc"] = ang_visc
#             file.flush()
#     print("\nDONE!\n")
# else:
#     print("SKIPPED!\n")
#

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
