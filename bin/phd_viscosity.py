
import boltzpy as bp
import numpy as np
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
           "print prandtl numbers": False,
           "angle_dependency_3D": False,
           "angle_effect_3D": False,
           "simplified_angular_weight_adjustment_effects": False,
           "heuristic": False,
           "multispecies": False,
           "bisection_3D": False,
           "bisection3D_angular_invariance": False,
           "bisection_3D_free_parameter": False,
           "prandtl_shape_Effect_2D": False,
           "persistence_over_altered_temperature": False,
           "persistence_over_altered_number_densities_1": False,
           "bisection_3d_pairwise": False,
           "persistence_over_altered_number_densities_2": False,
           "bisection_3d_multispecies_pairwise": True,
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
        W_MAT = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1]])
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


def cmp_visc(i, normalize=False):
    result = np.empty(m[i].ndim)
    if m[i].ndim == 3:
        result[0] = m[i]._cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[1, 1, 1], [0, 1, -1]],
            dt=dt,
            normalize=normalize)
        result[1] = m[i]._cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 1, 1], [0, 1, -1]],
            dt=dt,
            normalize=normalize)
        result[2] = m[i]._cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 0, 1], [1, 0, 0]],
            dt=dt,
            normalize=normalize)

    elif m[i].ndim == 2:
        result[0] = m[i]._cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[1, 1], [1, -1]],
            dt=dt,
            normalize=normalize)
        result[1] = m[i]._cmp_viscosity(
            number_densities=nd[i],
            temperature=temp[i],
            directions=[[0, 1], [1, 0]],
            dt=dt,
            normalize=normalize)

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


def cmp_visc_ext(model, nd, temp, normalize=False):
    result = np.empty(model.ndim)
    if model.ndim == 3:
        result[0] = model._cmp_viscosity(
            number_densities=nd,
            temperature=temp,
            directions=[[1, 1, 1], [0, 1, -1]],
            dt=dt)
        result[1] = model._cmp_viscosity(
            number_densities=nd,
            temperature=temp,
            directions=[[0, 1, 1], [0, 1, -1]],
            dt=dt)
        result[2] = model._cmp_viscosity(
            number_densities=nd,
            temperature=temp,
            directions=[[0, 0, 1], [1, 0, 0]],
            dt=dt)
    elif model.ndim == 2:
        result[0] = model._cmp_viscosity(
            number_densities=nd,
            temperature=temp,
            directions=[[1, 1], [1, -1]],
            dt=dt,
            normalize=normalize)
        result[1] = model._cmp_viscosity(
            number_densities=nd,
            temperature=temp,
            directions=[[0, 1], [1, 0]],
            dt=dt,
            normalize=normalize)
    else:
        raise NotImplementedError
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
        hdf_group["viscosity"][i_a] = m[2]._cmp_viscosity(
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
plt.close("all")
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
    ndim = 2
    grp = m[ndim].group(m[ndim].key_angle(m[ndim].collision_relations))
    n_keys = len(grp.keys())
    angles = np.empty((n_keys + 1, ndim * (ndim - 1)), dtype=int)
    visc = np.empty((n_keys + 1, ndim))
    heat = np.empty((n_keys + 1, ndim))
    ncols = np.empty(n_keys, dtype=int)
    for idx, (key, val) in enumerate(grp.items()):
        print("\rdim = %1d    -   angle = %3d / %3d"
              % (ndim, idx, n_keys),
              end="")
        # first element is without any changes
        ncols[idx] = val.size
        idx = idx + 1
        w_factor = 10
        m[ndim].collision_weights[val] *= w_factor
        m[ndim].update_collisions(m[ndim].collision_relations,
                                 m[ndim].collision_weights)
        angles[idx] = key
        visc[idx] = cmp_visc(ndim)
        heat[idx] = cmp_heat(ndim)
        m[ndim].collision_weights[val] = DEFAULT_WEIGHT
    # reset collision weights
    m[ndim].collision_weights[...] = DEFAULT_WEIGHT
    m[ndim].update_collisions(m[ndim].collision_relations,
                             m[ndim].collision_weights)
    angles[0] = [0.0]
    visc[0] = cmp_visc(ndim)
    heat[0] = cmp_heat(ndim)
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
    print("\nDONE!\n")
else:
    print("\nSKIPPED!\n")


print("############################################################################\n"
      "Plot: Feasibility of Angular Weight Adjustments in 2D\n"
      "############################################################################")
plt.close("all")
fig = plt.figure(figsize=(12.75, 5.25))
ax = fig.add_subplot()

ax.set_title("Effect of Amplified Collision Angles on Flow Parameters in a 2D Model",
             fontsize=fs_title)

BAR_WIDTH = 0.2
OFFSET = 0.125 * np.array([-1, 1], dtype=float)
colors = ["tab:green", "tab:blue",
          "limegreen", "cornflowerblue",
          "tab:red", ]
labels = [r"$\widetilde{\mu}_1$",
          r"$\widetilde{\mu}_2$",
          r"Rescaled Heat Conductivity $\frac{\kappa}{10}$"]

print("load stored results for left plot")
hdf_group = file["angle_effect_2D"]
angles = hdf_group["angles"][()]
visc = hdf_group["visc"][()]
heat = hdf_group["heat"][()]
visc_corr = hdf_group["visc_corr"][()]
heat_corr = hdf_group["heat_corr"][()]

ndim = 2
ax.set_xlabel(r"Affected Collision Angles", fontsize=fs_label)
pos = np.arange(angles.shape[0])
twax = ax.twinx()
ax.set_ylabel("Flow Parameter Coefficients",
              fontsize=fs_label)
twax.set_ylabel(r"Prandtl Numbers",
                fontsize=fs_label,
                rotation=270,
                labelpad=25)
for i in range(ndim):
    twax.scatter(pos + OFFSET[i],
                 visc[:, i] * heat_corr[i] / (visc_corr[i] * heat[:, i]),
                 color=colors[ndim + i],
                 edgecolor="black",
                 # label=labels[i],
                 lw=3,
                 s=100,)
twax.set_ylim(0, 1)
twax.tick_params(axis="y", labelsize=fs_ticks)

ax.scatter([],
           [],
           color="white",
           edgecolor="black",
           label="Prandtl Numbers",
           lw=3,
           s=75)

for i in range(ndim):
    ax.bar(x=pos + OFFSET[i],
           height=visc[:, i],
           width=BAR_WIDTH,
           bottom=0.0,
           color=colors[i],
           label=labels[i])
    cur_label = labels[ndim] if i == 0 else "_nolegend_"
    ax.plot(pos,
            0.1 * heat[:, i],
            color=colors[2*ndim],
            label=cur_label,
            marker="o",
            markersize=6)

ticklabels = [str(tuple(a)) for a in angles]
ticklabels[0] = "Original"
ax.set_xticks(pos)
ax.set_xticklabels(ticklabels)
# change order in legend
handles, labels = ax.get_legend_handles_labels()
order = [2, 3, 0, 1]
fig.legend([handles[idx] for idx in order],
           [labels[idx] for idx in order],
           loc="lower center",
           ncol=4,
           bbox_to_anchor=(0.5, -0.15),
           fontsize=fs_legend+2)
ax.tick_params(axis="x", labelsize=fs_ticks)
ax.tick_params(axis="y", labelsize=fs_ticks)

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_feasibility_2D.pdf",
            # bbox_extra_artists=(st, ),
            bbox_inches='tight')
print("Done!\n")


print("######################################################\n"
      "#       compute bisection Scheme for 2D models       #\n"
      "######################################################")
atol = 1e-6
rtol = 1e-3
if "bisection_2D" not in file.keys():
    hdf_group = file.create_group("bisection_2D")
    grp = m[2].group(m[2].key_angle(m[2].collision_relations))
    for key in [(0,1), (1,1), (1,2)]:
        print("Angular Weight Adjustment for " + str(key))
        subgrp = hdf_group.create_group(str(key))
        # prepare initial parameters for bisection
        weights = [0.1, 10]
        # store all computed viscosities here
        visc = []
        for w in weights:
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
        w_lo = [weights[order[0]]] * 2
        w_hi = [weights[order[1]]] * 2
        assert visc_lo[-1][1] - visc_lo[-1][0] < 0
        assert (visc_hi[-1][1] - visc_hi[-1][0] > 0)
        assert len(w_hi) == len(w_lo)
        assert len(visc_hi) == len(visc_lo)
        assert len(w_hi) == len(visc_lo)
        assert len(visc) == len(visc_lo)

        while True:
            w = (w_lo[-1] + w_hi[-1]) / 2
            weights.append(w)
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
            print("\rWeight = {:10e} - Absolute = {:3e} - Realtive = {:3e}"
                  " - i = {}".format(w, adiff, rdiff, len(visc)),
                  end="")
            if adiff < atol and rdiff < rtol:
                m[2].collision_weights[:] = DEFAULT_WEIGHT
                break
        # store results in h5py
        subgrp["weights"] = np.array(weights)
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
plt.close("all")
fig, axes = plt.subplots(1, 3, constrained_layout=True,
                         sharey="all",
                         figsize=(12.75, 5.25))
# set up second ax
twax = [axes[k].twinx() for k in [0, 1, 2]]

fig.suptitle("Bisecting "
             r"$\left\lvert \widetilde{\mu}_1 - \widetilde{\mu}_2\right\rvert$"
             " with an Angular Weight Adjustment "
             r"$\gamma_\eta$",
             fontsize=fs_suptitle)

axes[0].set_ylabel(r"Viscosity Coefficients", fontsize=fs_label)
twax[2].set_ylabel(r"Angular Weight Factor $\gamma_\eta$",
                   fontsize=fs_label,
                   rotation=270,
                   labelpad=25)
colors = ["tab:green", "tab:blue"]
labels = [r"$\widetilde{\mu}_1$",
          r"$\widetilde{\mu}_2$"]
max_steps = 0


for k, key in enumerate([(0,1), (1,1), (1,2)]):
    print("load stored results for key = ", key)
    subgrp = file["bisection_2D/" + str(key)]
    visc = subgrp["visc"][()]
    weights = subgrp["weights"][()]
    final_weight = 0.5 * (subgrp["w_hi"][-1] + subgrp["w_lo"][-1])
    print("final weight = ",
          final_weight)
    pos = np.arange(visc.shape[0]) + 1
    # max_steps = max(max_steps, visc.shape[0])
    max_Steps = pos
    for j in [0, 1]:
        axes[k].plot(pos,
                     visc[:, j],
                     color=colors[j],
                     lw=2,
                     ls="solid",
                     marker="o",
                     markersize=4,
                     label=labels[j])
        twax[k].plot(pos,
                     weights[:],
                     color="black",
                     lw=2,
                     ls="dotted",
                     marker="s",
                     markersize=4,
                     label=r"$\gamma_\eta$")
    # add twax to legend
    axes[k].plot([],[],
                 color="black",
                 lw=2,
                 ls="dotted",
                 marker="x",
                 markersize=4,
                 label=r"$\gamma_\eta$")
    # axes[k].plot(pos,
    #              np.abs(visc[:, 0] - visc[:, 1]),
    #              # color=colors[j],
    #              lw=2,
    #              ls="solid",
    #              marker="o",
    #              markersize=4
    #              )
    # axes[k].set_yscale("log")
    pos = np.arange(max_steps) + 1
    # axes[k].set_xticks(pos[4::5])
    axes[k].tick_params(axis="x", labelsize=fs_ticks)
    axes[k].tick_params(axis="y", labelsize=fs_ticks)
    axes[k].set_title(r"Angle $\eta = " + str(key)
                      # + ",\: \gamma_\eta = " + str(final_weight)[:5]
                      + "^T$",
                      fontsize=fs_title)
    axes[k].set_xlabel(r"Algorithmic Time Step", fontsize=fs_label)
    axes[k].grid(linestyle="dotted")
# set up aesthetics for twin axes
for i in [0, 1,2]:
    twax[i].set_ylim(0, 10)
    twax[i].tick_params(axis="y", labelsize=fs_ticks)
    if i != 2:
        twax[i].set_yticks([])
axes[0].legend(loc="upper right", fontsize=fs_legend+6)

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
        hdf_group["viscosity"][i_a] = m[2]._cmp_viscosity(
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
plt.close("all")
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
key = "print prandtl numbers"
if COMPUTE[key]:
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

        visc = m[2]._cmp_viscosity(
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
else:
    print("Skipped!\n")

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
            subgrp["viscosity"] [n] = m[3]._cmp_viscosity(
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

# print("######################################\n"
#       "Plot: Dependency of Second Angle in 3D\n"
#       "######################################")
# plt.close("all")
# fig = plt.figure(constrained_layout=True, figsize=(10, 10))
# ax = fig.add_subplot(projection="polar")
# print("Create  plot: Angle Dependency of the Viscosity in a 3D Model")
# ax.set_title(r"Angular Dependencies of the Viscosity  $\lambda$ in a 3D Model",
#              fontsize=fs_title)
#
# key = "angle_dependency_3D"
# styles = ["solid", "solid", "solid", "dashed"]
# colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
# labels = [
#     r"$\lambda\left(\begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}, \psi\right)$",
#     r"$\lambda\left(\begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}, \psi\right)$",
#     r"$\lambda\left(\begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}, \psi\right)$",
#     r"$\lambda\left(\begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}, \psi\right)$",
# ]
# # iterate over this, to have a fixed order!
# a_keys = [(1, 1, 1), (0, 1, 1), (0, 0, 1), (1, 2, 3)]
# hdf_group = file[key]
# for k, first_angle in enumerate(a_keys):
#     subgrp = hdf_group[str(first_angle)]
#     if k == 3:
#         rad = subgrp["rad_angle"][()] + 0.9
#     else:
#         rad = subgrp["rad_angle"][()]
#     visc = subgrp["viscosity"][()]
#     print("Relative Differences for ", first_angle)
#     print("Viscosity: ", np.max(visc) / np.min(visc) - 1)
#     heat = subgrp["heat_conductivity"][()]
#     print("Heat Conductivity: ", np.max(heat) / np.min(heat) - 1)
#
#     ax.plot(rad,
#             visc,
#             ls=styles[k],
#             c=colors[k],
#             label=labels[k],
#             lw=4)
# ax.set_rlabel_position(225)
# # axes[1].set_rlim(0, 1.4)
# ax.set_rticks([0, 0.00005, 0.0001, 0.00015])
# ax.tick_params(axis="both", labelsize=16)
# ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
#           fontsize=fs_legend
#           )
# ax.grid(linestyle="dotted")
#
# plt.savefig(bp.SIMULATION_DIR + "/phd_angle_dependency_3D_single.pdf")
# print("Done!\n")


print("######################################\n"
      "Plot: Both Angular Dependencies in 3D\n"
      "######################################")
plt.close("all")
fig, axes = plt.subplots(1, 2,
                         figsize=(12.75, 9),
                         subplot_kw={"projection": "polar"})

fig.suptitle(r"Angular Dependencies of Flow Parameters in a 3D Model",
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
titles = [
    r"Viscosity Coefficient $\widetilde{\mu}(\eta, \theta)$",
    r"Heat Conductivity Coefficient $\widetilde{\kappa}(\theta)$"
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
        axes[a].set_title(titles[a], fontsize=fs_title)
        if a == 1:
            max_val = np.max(subgrp[coeff][()])
            axes[a].set_rticks([0, 0.0075, 0.015, 0.0225])
            axes[a].set_rlim(0, 1.15 * max_val)
        else:
            axes[a].set_rticks([0,  0.0005, 0.001])
            axes[a].set_rlim(0, 0.0013)
for ax in axes:
    ax.set_rlabel_position(225)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(linestyle="dotted")

fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.14), ncol=4,
               fontsize=fs_legend+4)
plt.subplots_adjust(bottom=0.3)
plt.savefig(bp.SIMULATION_DIR + "/phd_angle_dependency_3D_both.pdf",
            bbox_inches='tight')
print("Done!\n")


print("########################################################\n"
      "#    Compute Feasibility for Angular Weights in 3D     #\n"
      "########################################################")
key = "angle_effect_3D"
if key not in file.keys():
    hdf_group = file.create_group(key)
    ndim = 3
    grp = m[ndim].group(m[ndim].key_angle(m[ndim].collision_relations))
    grp = {key: grp[key]
           for key in [(0, 0, 1, 0, 0, 1),
                       (0, 0, 1, 0, 1, 1),
                       # (0, 0, 1, 0, 1, 4),
                       (0, 1, 1, 0, 1, 1),
                       (0, 1, 1, 1, 1, 1),
                       (1, 1, 1, 1, 1, 2)
                       ]}
    n_keys = len(grp.keys())
    angles = np.empty((n_keys + 1, ndim * (ndim - 1)), dtype=int)
    visc = np.empty((n_keys + 1, ndim))
    heat = np.empty((n_keys + 1, ndim))
    ncols = np.empty(n_keys, dtype=int)
    for idx, (key, val) in enumerate(grp.items()):
        print("\rndim = %1d    -   angle = %3d / %3d"
              % (ndim, idx, n_keys),
              end="")
        # first element is without any changes
        ncols[idx] = val.size
        idx = idx + 1
        w_factor = 10
        m[ndim].collision_weights[val] *= w_factor
        m[ndim].update_collisions(m[ndim].collision_relations,
                                 m[ndim].collision_weights)
        angles[idx] = key
        visc[idx] = cmp_visc(ndim)
        heat[idx] = cmp_heat(ndim)
        m[ndim].collision_weights[val] = DEFAULT_WEIGHT
    # reset collision weights
    m[ndim].collision_weights[...] = DEFAULT_WEIGHT
    m[ndim].update_collisions(m[ndim].collision_relations,
                             m[ndim].collision_weights)
    angles[0] = [0.0]
    visc[0] = cmp_visc(ndim)
    heat[0] = cmp_heat(ndim)
    # store results in file
    hdf_group["angles"] = angles
    hdf_group["visc"] = visc
    hdf_group["heat"] = heat
    file.flush()

    # compute correction terms separately
    hdf_group["visc_corr"] = np.zeros(ndim)
    hdf_group["heat_corr"] = np.zeros(ndim)
    eta = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=float)
    theta = np.array([[0, 1, -1], [0, 1, -1], [1, 0, 0]], dtype=float)
    maxwellian = m[ndim].cmp_initial_state(nd[ndim], 0, temp[ndim])
    for i in range(ndim):
        visc_mf = m[ndim].mf_stress(np.zeros(ndim),
                                   np.array([eta[i], theta[i]]),
                                   orthogonalize=True)
        hdf_group["visc_corr"][i] = np.sum(visc_mf ** 2 * maxwellian)
        heat_mf = m[ndim].mf_heat_flow(np.zeros(ndim),
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
plt.close("all")
fig = plt.figure(constrained_layout=True, figsize=(12.75, 5.25))
ax = fig.add_subplot()
# fig.suptitle("Influence of Collision Angles on Viscosity, Heat Transfer, and Prandtl Number",
#              fontsize=16)
ax.set_title(r"Effect of Amplified Collision Angles on Flow Parameters in a 3D Model",
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
labels = [r"$\widetilde{\mu}_1$",
          r"$\widetilde{\mu}_2$",
          r"$\widetilde{\mu}_3$",
          r"Rescaled Heat Conductivity $\frac{\kappa}{10}$"]
# ax.set_ylabel(r"Viscosity and Heat Transfer", fontsize=fs_label)
ndim = 3
ax.set_xlabel(r"Affected Collision Angles", fontsize=fs_label)
pos = np.arange(angles.shape[0])
twax = ax.twinx()
ax.set_ylabel("Flow Parameter Coefficients",
              fontsize=fs_label)
twax.set_ylabel(r"Prandtl Numbers",
                fontsize=fs_label,
                rotation=270,
                labelpad=25)
for i in range(ndim):
    twax.scatter(pos + OFFSET[i],
                 visc[:, i] * heat_corr[i] / (visc_corr[i] * heat[:, i]),
                 color=colors[i],
                 edgecolor="black",
                 lw=3,
                 s=100,)
twax.set_ylim(0, 1)
twax.tick_params(axis="y", labelsize=fs_ticks)

ax.scatter([],
           [],
           color="white",
           edgecolor="black",
           label="Prandtl Numbers",
           lw=3,
           s=75)

for i in range(ndim):
    ax.bar(x=pos + OFFSET[i],
           height=visc[:, i],
           width=BAR_WIDTH,
           bottom=0.0,
           color=colors[i],
           label=labels[i])
    cur_label = labels[3] if i == 0 else "_nolegend_"
    ax.plot(pos,
            0.1 * heat[:, i],
            color=colors[2*ndim],
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
# change order in legend
handles, labels = ax.get_legend_handles_labels()
order = [2, 3, 4, 0, 1]
fig.legend([handles[idx] for idx in order],
           [labels[idx] for idx in order],
           loc="lower center",
           bbox_to_anchor=(0.50, -0.15),
           ncol=5,
           fontsize=fs_legend+6)
ax.tick_params(axis="both", labelsize=fs_ticks)

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_feasibility_3D.pdf",
            # bbox_extra_artists=(st, ),
            bbox_inches='tight')
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
    weights = np.linspace(MIN_W, MAX_W, N_POINTS)
    for ndim in [2, 3]:
        ds = hdf_group.create_dataset(str(ndim),
                                      (ndim, N_POINTS, ndim), dtype=float)
        for a in range(ndim):
            for i_w, w in enumerate(weights):
                print("\rndim = %1d, a = %1d, weight = %3d / %3d"
                      % (ndim, a, i_w, N_POINTS),
                      end="")
                m[ndim].collision_weights[...] = DEFAULT_WEIGHT
                new_weights = np.ones(ndim, dtype=float)
                new_weights[a] = w
                adjust_weight_by_angle(m[ndim], new_weights)
                ds[a, i_w] = cmp_visc(ndim)
                file.flush()
    print("DONE!\n")
else:
    print("SKIPPED!\n")

print("####################################################\n"
      "     Create Plot for weight_adjustment_effects      \n"
      "####################################################")
key = "simplified_angular_weight_adjustment_effects"
visc = {ndim: file[key][str(ndim)][()]
        for ndim in [2, 3]}
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
for ax, ndim in enumerate([2, 3]):
    axes[ax].set_xlabel(r"Weight factor $\omega_\bullet$", fontsize=fs_label)
    for a in range(ndim):
        for i in range(ndim):
            axes[ax].plot(weights,
                          visc[ndim][a, :, i],
                          color=colors[ndim][a, i],
                          label=labels[ndim][a, i],
                          ls=styles[ndim][a],
                          lw=3,
                          zorder=5 - i)
    # create legend, with colums and per-column-title
    leg = axes[ax].get_legend_handles_labels()
    new_leg = []
    for i_l in [0,1]:
        l = np.array(leg[i_l]).reshape((ndim, ndim))
        new_l = np.empty((ndim, ndim+1), dtype=object)
        new_l[:, 1:] = l
        if i_l == 0:
            new_l[:, 0] = plt.plot([],marker="", ls="")[0]
        elif ndim == 2:
            new_l[:, 0] = [r"$\boldmath{\omega_1:}$",
                           r"$\boldmath{\omega_2:}$"]
        else:
            new_l[:, 0] = [r"$\boldmath{\omega_1:}$",
                           r"$\boldmath{\omega_2:}$",
                           r"$\boldmath{\omega_3:}$"]
        new_leg.append(new_l.flatten())
    axes[ax].legend(*new_leg, ncol=ndim, loc="upper right",
                    # fontsize=fs_legend
                    )
plt.savefig(bp.SIMULATION_DIR + "/phd_simplified_angular_weight_adjustment_effects.pdf")
print("Done!\n")

print("##########################################################\n"
      "#   Plot: Feasibility of Simplified Weight Adjustments   #\n"
      "##########################################################")
key = "simplified_angular_weight_adjustment_effects"
visc = {ndim: file[key][str(ndim)][()]
        for ndim in [2, 3]}
weights = file[key + "/weights"][()]

colors = {3: [["red", "darkred", "orange"],
              ["limegreen", "darkgreen", "olive"],
              ["blue", "navy", "dodgerblue"]]}
colors[2] = colors[3][1:]
labels = {3: [r"$\widetilde{\mu}_1$",
              r"$\widetilde{\mu}_2$",
              r"$\widetilde{\mu}_3$"],
          2: [r"$\widetilde{\mu}_1$",
              r"$\widetilde{\mu}_2$"]
          }
xlabels = {3: [r"$\gamma_{(1,1,1)}$",
               r"$\gamma_{(0,1,1)}$",
               r"$\gamma_{(0,0,1)}$"],
           2: [r"$\gamma_{(1,1)}$",
               r"$\gamma_{(0,1)}$"]
           }
styles = {3: ["solid", "dashed", "dotted"],
          2: ["solid", "dashed", "dotted"]
          }

for ndim in [2, 3]:
    fig, axes = plt.subplots(1, ndim, constrained_layout=True,
                             sharey="all",
                             figsize=(12.75, 5.25))
    fig.suptitle("Effects of Componentwise Simplified Weight Adjustments "
                 "in " + str(ndim) + "D DVM",
                 fontsize=fs_suptitle)

    axes[0].set_ylabel("Viscosity Coefficients", fontsize=fs_label)
    for a in range(ndim):
        axes[a].set_xlabel("Weight Factor " + xlabels[ndim][a], fontsize=fs_label)
        axes[a].grid(linestyle="dotted")
        axes[a].tick_params(axis="both", labelsize=fs_ticks)
        for i in range(ndim):
            axes[a].plot(weights,
                         visc[ndim][a, :, i],
                         color=colors[ndim][a][i],
                         label=labels[ndim][i],
                         ls=styles[ndim][i],
                         lw=3)
        axes[a].legend(loc="upper right",
                       fontsize=fs_legend+4)
    plt.savefig(bp.SIMULATION_DIR + "/phd_simplified_weight_effects"
                + str(ndim) + "D.pdf")
print("Done!\n")


print("#########################################################\n"
      "    Compute Heuristic for 2D and 3D Model of 2 Species   \n"
      "#########################################################\n")
key = "heuristic"
atol = 1e-6
rtol = 1e-4
if key not in file.keys():
    hdf_group = file.create_group(key)
    for ndim in [2,3]:
        print("Harmonize %1dD model..." % ndim)
        subgrp = hdf_group.create_group(str(ndim))
        m[ndim].collision_weights[...] = DEFAULT_WEIGHT
        visc = [cmp_visc(ndim)]
        weights = [np.ones(ndim)]
        ctr = 0
        while True:
            pos = np.argsort(visc[-1])
            # w = np.ones(ndim, dtype=float)
            mean = np.sum(visc[-1]) / ndim
            # w[pos[0]] = visc[-1][pos[0]] / mean
            # w[pos[-1]] = visc[-1][pos[-1]] / mean
            w = visc[-1] / mean
            weights.append(weights[-1] * w)
            # reset previous weight adjustments, to keep weights linear
            m[ndim].collision_weights[...] = DEFAULT_WEIGHT
            adjust_weight_by_angle(m[ndim], weights[-1])
            visc.append(cmp_visc(ndim))
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

        if ndim == 3:
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
                    ang_visc[n] = m[3]._cmp_viscosity(
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

for a, ndim in enumerate([2, 3]):
    visc = file["heuristic"][str(ndim)]["visc"]
    weights = file["heuristic"][str(ndim)]["weights"]
    pos = np.arange(visc.shape[0]) + 1
    axes[a].set_xlabel(r"Algorithmic Time Step", fontsize=fs_label)
    twax = axes[a].twinx()
    if a == 1:
        twax.set_ylabel(r"Weight Factor $\omega$", fontsize=fs_label)
    for i in range(ndim):
        axes[a].plot(pos,
                     visc[:, i],
                     color=colors[ndim][i],
                     label=labels[ndim][i],
                     marker="o",
                     markersize=6)
        twax.plot(pos,
                  weights[:, i],
                  color=colors[ndim][i],
                  ls="dotted",
                  markersize=6)
    for i in range(ndim):
        axes[a].plot([], [],
                     color=colors[ndim][i],
                     label=r"$\omega_{%1d}$" % (i + 1),
                     ls="dotted",
                     markersize=6)
        twax.set_ylim(0, int(np.max(weights)) + 1)

    axes[a].set_xticks(pos[::2])
    axes[a].legend(loc="upper right", ncol=2, fontsize=fs_legend)

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_heuristic.pdf")
print("Done!\n")


print("######################################################\n"
      "# Plot: Dependency for adjusted 3D Model (heuristic) #\n"
      "######################################################")
plt.close("all")
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
    print("Relative Differences for ", first_angle)
    print("Viscosity: ", np.max(visc) / np.min(visc) - 1)
    heat = subgrp["angular_heat"][()]
    print("Heat Conductivity: ", np.max(heat) / np.min(heat) - 1)

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

plt.savefig(bp.SIMULATION_DIR + "/phd_angular_invariance_3D_heuristic.pdf")
print("Done!\n")


# print("#####################################################################\n"
#       "   Compute harmonized Angular viscosities for altered temperatures\n"
#       "#####################################################################")
# N_TEMPS = 50
# key = "persistence_over_altered_temperature"
# if key not in file.keys():
#     hdf_group = file.create_group(key)
#     # copy old parameters for viscosity
#     nd_orig = nd.copy()
#     mean_v_orig = mean_v.copy()
#     temp_orig = temp.copy()
#     for ndim in [2,3]:
#         print("\nCheck %1dD model..." % ndim)
#         subgrp = hdf_group.create_group(str(ndim))
#         m[ndim].collision_weights[...] = DEFAULT_WEIGHT
#         weights = file["heuristic"][str(ndim)]["weights"][-1]
#         adjust_weight_by_angle(m[ndim], weights)
#         temp_range = m[ndim].temperature_range(atol=0.1)
#         temperatures = np.linspace(*temp_range, N_TEMPS)
#         visc = np.zeros((N_TEMPS, ndim))
#         for i_t, t in enumerate(temperatures):
#             print("\r%3d / %3d" % (i_t+1, N_TEMPS), end="")
#             temp[ndim][...] = t
#             visc[i_t] = cmp_visc(ndim)
#         subgrp["temperatures"] = temperatures
#         subgrp["visc"] = visc
#     print("\nDONE!\n")
# else:
#     print("SKIPPED!\n")
#
# print("########################################################################\n",
#       "#     Plot harmonized angular viscosities for altered temperatures     #\n",
#       "########################################################################")
# fig, axes = plt.subplots(1, 2, constrained_layout=True,
#                          figsize=(12.75, 6.25))
# fig.suptitle("Viscosity Coefficients of an Adjusted DVM for Altered Temperatures",
#              fontsize=fs_suptitle)
# axes[0].set_title(r"2D Collision Model",
#                   fontsize=fs_title)
# axes[1].set_title(r"3D Collision Model",
#                   fontsize=fs_title)
#
# colors = {3: np.array(["tab:orange",
#                        "tab:green",
#                        "tab:blue"])}
# colors[2] = colors[3][1:]
# labels = np.array([r"$\widetilde{\mu}_1$",
#                    r"$\widetilde{\mu}_2$",
#                    r"$\widetilde{\mu}_3$"])
#
# styles = {3: np.array(["dotted", "dashed", "solid"])}
# styles[2] = styles[3][1:]
#
# axes[0].set_ylabel("Viscosity Coefficients", fontsize=fs_label)
#
# key = "persistence_over_altered_temperature"
# hdf_group = file[key]
# for ax, ndim in enumerate([2, 3]):
#     axes[ax].set_xlabel(r"Temperature", fontsize=fs_label)
#     for i in range(ndim):
#         temperatures = hdf_group[str(ndim)]["temperatures"][()]
#         visc = hdf_group[str(ndim)]["visc"][:, i]
#         axes[ax].plot(temperatures,
#                       visc,
#                       color=colors[ndim][i],
#                       label=labels[i],
#                       ls=styles[ndim][i],
#                       lw=3)
#     axes[ax].set_ylim(0, 1.5 * np.max(visc))
#     axes[ax].tick_params(axis="both", labelsize=fs_ticks)
#     axes[ax].legend(loc="upper right", fontsize=fs_legend)
#     axes[ax].grid(linestyle="dotted")
#
# plt.savefig(bp.SIMULATION_DIR + "/phd_persistence_over_altered_temperature.pdf")
# print("Done!\n")


print("##################################################################\n"
      "#   Multispecies: Apply 3D-Heuristic Species-Wise to 3-Mixture   #\n"
      "##################################################################")

atol = 1e-5
rtol = 1e-3
MAX_ITER = 250
key = "multispecies"
if key not in file.keys():
    hdf_group = file.create_group(key)
    # distribution parameters
    nd3 = np.ones(3)
    temp3 = np.array([2.25] * 3)
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
    model_grp = model.group((model.key_species(model.collision_relations)[:, 1:3]))
    print("DONE!\n")

    print("Apply Heuristic specieswise")
    # process intraspecies collisions first,
    # then harmonize interspecies collisions based on harmonized Single Species
    spc_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for (s1, s2) in spc_pairs:
        print("Harmonize (%1d, %1d)-Collisions..." % (s1, s2))
        subgrp = hdf_group.create_group(str((s1, s2)))
        # setuo new initial states
        sm = model.submodel(list({s1, s2}))
        sm_grp = sm.group(sm.key_species(sm.collision_relations)[:, 1:3])
        if s1 == s2:
            cur_spc = (0, 0)
        else:
            cur_spc = (0, 1)
        cur_iter = 0
        weights = [np.ones(sm.ndim)]
        visc = []
        while True:
            # apply new collision weights
            sm.collision_weights[sm_grp[cur_spc]] = DEFAULT_WEIGHT
            adjust_weight_by_angle(sm, weights[-1], sm_grp[cur_spc])

            # compute viscosity
            visc.append(cmp_visc_ext(sm,
                                     nd3[list({s1, s2})],
                                     temp3[list({s1, s2})]))

            # compute weight factor for new weight
            mean_visc = np.sum(visc[-1]) / ndim
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
                print("\nHarmonization of DVM failed!")
                raise ValueError
            # adjust weights, multiply new correction factor
            weights.append(w_factor * weights[-1])
            cur_iter += 1

        # apply weights to model
        adjust_weight_by_angle(model, weights[-1], model_grp[(s1, s2)])

        # store results in h5py
        subgrp["weights"] = np.array(weights)
        subgrp["visc"] = np.array(visc)
        file.flush()
        print("\nFinal Weights = ", weights[-1])
        print("Directional Viscosities = ", visc[-1], "\n")
    print("DONE!\n")

    print("###########################################################\n"
          "#    compute angular dependecies for adjusted 3D model    #\n"
          "###########################################################")
    hdf_group = hdf_group.create_group("model")
    rotations = []
    rotations.append([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    rotations.append([[1, 0, 0],
                      [0, 1, 1],
                      [0, -1, 1]])
    rotations.append([[-1, 1, 1],
                      [-1, -1, 1],
                      [2, 0, 1]])
    rotations.append([[-3, -2, 1],
                      [0, 10, 2],
                      [1, -6, 3]])

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
            ang_visc[n] = model._cmp_viscosity(
                number_densities=nd3,
                temperature=temp3,
                directions=[a, sa],
                dt=dt,
                normalize=False)
            ang_heat[n] = model.cmp_heat_transfer(
                number_densities=nd3,
                temperature=temp3,
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


print("########################################################\n"
      "#   Plot: Dependency in adjusted 3D, 3-Species Model   #\n"
      "########################################################")
plt.close("all")
fig = plt.figure(constrained_layout=True, figsize=(12.75, 8))
ax = fig.add_subplot(projection="polar")
ax.set_title(r"Angular Dependencies of the Viscosity $\lambda$ in the Adjusted 3D Model",
             fontsize=fs_title)

key = "multispecies/model"
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
    print("Relative Differences for ", first_angle)
    print("Viscosity: ", np.max(visc) / np.min(visc) - 1)
    heat = subgrp["angular_heat"][()]
    print("Heat Conductivity: ", np.max(heat) / np.min(heat) - 1)

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
# ax.set_rticks([0, 0.00005, 0.0001, 0.00015])
ax.tick_params(axis="both", labelsize=16)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
          fontsize=fs_legend)
ax.grid(linestyle="dotted")

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_multispecies_dependency.pdf")
print("Done!\n")


print("######################################################\n"
      "#       compute bisection Scheme for 3D models       #\n"
      "######################################################")
rtol = 1e-3
if "bisection_3D" not in file.keys():
    file.create_group("bisection_3D")
    for ndim in [2,3]:
        hdf_group = file["bisection_3D"].create_group(str(ndim))
        for used_weight in range(ndim):
            subgrp = hdf_group.create_group(str(used_weight))
            angular_weights = np.ones(ndim)
            # prepare initial parameters for bisection
            if used_weight == ndim - 1:
                biscetion_weights = [0.1, 2.0]
            else:
                biscetion_weights = [1.0, 6.0]
            # compute initial viscosities
            visc = []
            for w in biscetion_weights:
                m[ndim].collision_weights[:] = DEFAULT_WEIGHT
                angular_weights[used_weight] = w
                adjust_weight_by_angle(m[ndim], angular_weights)
                visc.append(cmp_visc(ndim))
            print("Initial Viscosity Coefficients:")
            print(np.array(visc))
            visc_diff = [visc[i][ndim-2] - visc[i][ndim-1] for i in [0, 1]]
            # the signs must differ, for the bisection scheme
            assert visc_diff[0] * visc_diff[1] < 0
            # the current weights and diffs describe the best results so far
            order = np.argsort(visc_diff)
            # store all weights, divided into upper and lower bounds
            w_lo = [biscetion_weights[np.argmin(visc_diff)]] * 2
            w_hi = [biscetion_weights[np.argmax(visc_diff)]] * 2
            print("Start Bisection Algorithm")
            while True:
                w = (w_lo[-1] + w_hi[-1]) / 2
                biscetion_weights.append(w)
                # compute new viscosity
                m[ndim].collision_weights[:] = DEFAULT_WEIGHT
                angular_weights[used_weight] = w
                adjust_weight_by_angle(m[ndim], angular_weights)
                visc.append(cmp_visc(ndim))
                visc_diff.append(visc[-1][ndim-2] - visc[-1][ndim-1])
                # update hi/lo lists
                if visc_diff[-1] < 0:
                    w_lo.append(w)
                    w_hi.append(w_hi[-1])
                else:
                    w_hi.append(w)
                    w_lo.append(w_lo[-1])
                assert len(w_hi) == len(w_lo)
                assert len(w_hi) == len(visc)
                rdiff = np.max(visc[-1]) / np.min(visc[-1]) - 1
                print("\rWeight = {:10e} - RDiff = {:3e}"
                      " - i = {}".format(w, rdiff, len(visc)),
                      end="")
                if rdiff < rtol:
                    m[ndim].collision_weights[:] = DEFAULT_WEIGHT
                    break
            # store results in h5py
            subgrp["biscetion_weights"] = np.array(biscetion_weights)
            subgrp["w_hi"] = np.array(w_hi)
            subgrp["w_lo"] = np.array(w_lo)
            subgrp["visc"] = np.array(visc)
            file.flush()
            print("\nDONE!\n")
else:
    print("SKIPPED!\n")


print("####################################\n"
      "Plot: bisection Scheme for 3D models\n"
      "####################################")
# fig = plt.figure(constrained_layout=True, figsize=(12.75, 6.25))
# ax = fig.add_subplot()
fig, axes = plt.subplots(1, 3, constrained_layout=True,
                         sharey="all",
                         figsize=(12.75, 5.25))
# set up second ax
twax = [axes[k].twinx() for k in [0, 1, 2]]

fig.suptitle("Bisecting "
             r"$\left\lvert \widetilde{\mu}_2 - \widetilde{\mu}_3\right\rvert$"
             " with a Single Component Simplified Angular Weight Adjustment",
             fontsize=fs_suptitle)

axes[0].set_ylabel(r"Viscosity Coefficients", fontsize=fs_label)
twax[2].set_ylabel(r"Angular Weight Factor $\gamma_\eta$",
                   fontsize=fs_label,
                   rotation=270,
                   labelpad=25)
colors = ["tab:orange", "tab:green", "tab:blue"]
labels = [r"$\widetilde{\mu}_1$",
          r"$\widetilde{\mu}_2$",
          r"$\widetilde{\mu}_3$"]
angles = [r"(1,1,1)^T",
          r"(0,1,1)^T",
          r"(0,0,1)^T"]
ndim = 3

for k in range(ndim):
    print("load stored results ", k)
    subgrp = file["bisection_3D"][str(ndim)][str(k)]
    visc = subgrp["visc"][()]
    weights = subgrp["biscetion_weights"][()]
    print("final weight = ",
          weights[-1])
    pos = np.arange(visc.shape[0]) + 1
    # max_steps = max(max_steps, visc.shape[0])
    max_Steps = pos
    for j in range(ndim):
        axes[k].plot(pos,
                     visc[:, j],
                     color=colors[j],
                     lw=2,
                     ls="solid",
                     marker="o",
                     markersize=4,
                     label=labels[j])
        twax[k].plot(pos,
                     weights[:],
                     color="black",
                     lw=2,
                     ls="dotted",
                     marker="s",
                     markersize=4,
                     label=r"$\gamma_\eta$")
    # add twax to legend
    axes[k].plot([],[],
                 color="black",
                 lw=2,
                 ls="dotted",
                 marker="x",
                 markersize=4,
                 label=r"$\gamma_\eta$")
    pos = np.arange(max_steps) + 1
    # axes[k].set_xticks(pos[4::5])
    axes[k].tick_params(axis="x", labelsize=fs_ticks)
    axes[k].tick_params(axis="y", labelsize=fs_ticks)
    axes[k].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[k].set_title(r"Angle $\eta = " + angles[k]
                      + "$",
                      fontsize=fs_title)
    axes[k].set_xlabel(r"Algorithmic Time Step", fontsize=fs_label)
    axes[k].grid(linestyle="dotted")
# set up aesthetics for twin axes
for i in [0, 1,2]:
    twax[i].set_ylim(0, 10)
    twax[i].tick_params(axis="y", labelsize=fs_ticks)
    if i != 2:
        twax[i].set_yticks([])
axes[0].legend(loc="upper right", fontsize=fs_legend+5)

plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity_bisection_3D.pdf")
print("Done!\n")


print("######################################################\n"
      "# compute bisection angular invariance for 3D models #\n"
      "######################################################")
ndim = 3
N_ANGLES = 101
key = "bisection3D_angular_invariance"
if key not in file.keys():
    hdf_group = file.create_group(key)
    print("apply computed weight")
    weight = file["bisection_3D"]["2"]["biscetion_weights"][-1]
    m[ndim].collision_weights[:] = DEFAULT_WEIGHT
    angular_weights = np.ones(ndim)
    angular_weights[ndim-1] = weight
    adjust_weight_by_angle(m[ndim], angular_weights)

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
            ang_visc[n] = m[ndim]._cmp_viscosity(
                number_densities=nd[ndim],
                temperature=temp[ndim],
                directions=[a, sa],
                dt=dt,
                normalize=False)
            ang_heat[n] = m[3].cmp_heat_transfer(
                number_densities=nd[ndim],
                temperature=temp[ndim],
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


print("######################################################\n"
      "# Plot: Dependency for adjusted 3D Model (bisection) #\n"
      "######################################################")
plt.close("all")
fig, axes = plt.subplots(1, 2,
                         figsize=(12.75, 9),
                         subplot_kw={"projection": "polar"})

fig.suptitle(r"Angular Dependencies of Flow Parameters in the Adjusted 3D Model",
             fontsize=fs_suptitle)

key = "bisection3D_angular_invariance"
styles = ["solid", "solid", "solid", "dashed"]
colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
labels = [
    r"$\eta = \begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}$",
    r"$\eta = \begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}$",
    r"$\eta = \begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}$",
    r"$\eta = \begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}$"
]
titles = [
    r"Viscosity Coefficient $\widetilde{\mu}(\eta, \theta)$",
    r"Heat Conductivity Coefficient $\widetilde{\kappa}(\theta)$"
]

# iterate over this, to have a fixed order!
a_keys = [(1, 1, 1),
          (0, 1, 1),
          (0, 0, 1),
          (1, 2, 3)
          ]
hdf_group = file[key]
for k, first_angle in enumerate(a_keys):
    subgrp = hdf_group[str(first_angle)]
    visc = subgrp["angular_visc"][()]
    print("Relative Differences for ", first_angle)
    print("Viscosity: ", np.max(visc) / np.min(visc) - 1)
    heat = subgrp["angular_heat"][()]
    print("Heat Conductivity: ", np.max(heat) / np.min(heat) - 1)

    if k == 3:
        rad = subgrp["rad_angle"][()] + 0.9
    else:
        rad = subgrp["rad_angle"][()]
    for a, coeff in enumerate(["angular_visc", "angular_heat"]):
        axes[a].plot(rad,
                     subgrp[coeff][()],
                     ls=styles[k],
                     c=colors[k],
                     label=labels[k] if a == 0 else "_nolegend_",
                     lw=4)
        axes[a].set_title(titles[a], fontsize=fs_title)
        max_val = np.max(subgrp[coeff][()])
        axes[a].set_rlim(0, 1.15 * max_val)
        if a == 1:
            axes[a].set_rticks([0, 0.0075, 0.015, 0.0225])
        else:
            axes[a].set_rticks([0, 0.0005, 0.001, 0.0015])
            # boundaries for counter example
            # axes[a].set_rticks([0.00136, 0.001365])
for ax in axes:
    ax.set_rlabel_position(225)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(linestyle="dotted")
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.14), ncol=4,
               fontsize=fs_legend+4)
plt.subplots_adjust(bottom=0.3)
plt.savefig(bp.SIMULATION_DIR + "/phd_angular_invariance_3D_bisection.pdf",
            bbox_inches='tight')
print("Done!\n")


print("#######################################################\n"
      "# compute Prandtl Dependency on Bisection Scheme (3D) #\n"
      "#######################################################")
rtol = 1e-3
N_WEIGHTS = 100
key = "bisection_3D_free_parameter"
if key not in file.keys():
    hdf_group = file.create_group(key)
    ndim = 3
    diagonal_weights = np.linspace(0.25, 10, N_WEIGHTS)
    hdf_group["diagonal_weights"] = diagonal_weights
    hdf_group["biscetion_weights"] = np.full((N_WEIGHTS, MAX_ITER), -1.0)
    hdf_group["w_hi"] = np.full((N_WEIGHTS, MAX_ITER), -1.0)
    hdf_group["w_lo"] = np.full((N_WEIGHTS, MAX_ITER), -1.0)
    hdf_group["visc"] = np.full((N_WEIGHTS, MAX_ITER, ndim), -1.0)
    hdf_group["prandtl"] = np.full(N_WEIGHTS, -1.0)
    for i_dw, dw in enumerate(diagonal_weights):
        print("i_dw = {:3d}/{:3d} - dw = {:3e} ".format(i_dw, N_WEIGHTS, dw))
        angular_weights = np.ones(ndim)
        angular_weights[0] = dw
        # prepare initial parameters for bisection
        biscetion_weights = [0.1, 4]
        print("Compute Initial Viscosity Coefficients:")
        visc = []
        for w in biscetion_weights:
            m[ndim].collision_weights[:] = DEFAULT_WEIGHT
            angular_weights[ndim-1] = w
            adjust_weight_by_angle(m[ndim], angular_weights)
            visc.append(cmp_visc(ndim))
        print(np.array(visc))
        visc_diff = [visc[i][ndim-2] - visc[i][ndim-1] for i in [0, 1]]
        # the signs must differ, for the bisection scheme
        print(visc_diff)
        assert visc_diff[0] * visc_diff[1] < 0
        # the current weights and diffs describe the best results so far
        order = np.argsort(visc_diff)
        # store all weights, divided into upper and lower bounds
        w_lo = [biscetion_weights[np.argmin(visc_diff)]] * 2
        w_hi = [biscetion_weights[np.argmax(visc_diff)]] * 2
        print("Start Bisection Algorithm")
        while True:
            w = (w_lo[-1] + w_hi[-1]) / 2
            biscetion_weights.append(w)
            # compute new viscosity
            m[ndim].collision_weights[:] = DEFAULT_WEIGHT
            angular_weights[ndim-1] = w
            adjust_weight_by_angle(m[ndim], angular_weights)
            visc.append(cmp_visc(ndim))
            visc_diff.append(visc[-1][ndim-2] - visc[-1][ndim-1])
            # update hi/lo lists
            if visc_diff[-1] < 0:
                w_lo.append(w)
                w_hi.append(w_hi[-1])
            else:
                w_hi.append(w)
                w_lo.append(w_lo[-1])
            assert len(w_hi) == len(w_lo)
            assert len(w_hi) == len(visc)
            rdiff = np.max(visc[-1]) / np.min(visc[-1]) - 1
            print("\rWeight = {:10e} - RDiff = {:3e}"
                  " - i = {}".format(w, rdiff, len(visc)),
                  end="")
            if rdiff < rtol:
                break
            if np.allclose(*biscetion_weights[-2:]):
                print("WARNING! Error below specified threshold")
                break
        print("\nstore results in h5py")
        pos = len(visc)
        hdf_group["biscetion_weights"][i_dw, :pos] = np.array(biscetion_weights)
        hdf_group["w_hi"][i_dw, :pos] = np.array(w_hi)
        hdf_group["w_lo"][i_dw, :pos] = np.array(w_lo)
        hdf_group["visc"][i_dw, :pos] = np.array(visc)
        file.flush()
        # compute prandtl number
        visc_coeff = m[ndim]._cmp_viscosity(
            number_densities=nd[ndim],
            temperature=temp[ndim],
            directions=[[1, 1, 1], [0, 1, -1]],
            dt=dt,
            normalize=True)
        heat_coeff = m[ndim].cmp_heat_transfer(
            number_densities=nd[ndim],
            temperature=temp[ndim],
            direction=[1, 1, 1],
            dt=dt,
            normalize=True)
        hdf_group["prandtl"][i_dw] = visc_coeff / heat_coeff
        file.flush()
    print("\nDONE!\n")
    m[ndim].collision_weights[:] = DEFAULT_WEIGHT
else:
    print("SKIPPED!\n")

print("#####################################################\n"
      "# Plot: Prandtl Dependency on Bisection Scheme (3D) #\n"
      "#####################################################")
plt.close("all")
fig = plt.figure(constrained_layout=True, figsize=(12.75, 5.25))
ax = fig.add_subplot()

fig.suptitle(r"Effect of $\gamma_{(1,1,1)}$ on Prandtl Number "
             r"after 3D Bisection Scheme",
             fontsize=fs_suptitle)
prandtl = file["bisection_3D_free_parameter"]["prandtl"][()]
diag_weights = file["bisection_3D_free_parameter"]["diagonal_weights"][()]
print(prandtl.shape, diag_weights.shape)
ax.plot(diag_weights,
        prandtl,
        lw=4,
        label="$Pr_1 = Pr_2 = Pr_3$")
ax.legend(loc="upper left", fontsize=fs_legend)
ax.tick_params(axis="x", labelsize=fs_ticks)
ax.tick_params(axis="y", labelsize=fs_ticks)
ax.set_xlabel(r"$\gamma_{(1,1,1)}$ Component of $\gamma_\mathcal{P}$ ",
              fontsize=fs_label)
ax.set_ylabel(r"Prandtl Number",
              fontsize=fs_label)
ax.grid(linestyle="dotted")
plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity3D_prandtl_effects.pdf")
print("Done!\n")


print("#################################################\n"
      "# compute Prandtl Dependency on Grid Shape (2D) #\n"
      "#################################################")
rtol = 1e-3
SHAPES = np.arange(5, 41)
key = "prandtl_shape_Effect_2D"
if key not in file.keys():
    hdf_group = file.create_group(key)
    ndim = 2
    hdf_group["prandtl"] = np.full(SHAPES.size, -1.0)
    hdf_group["shapes"] = SHAPES
    for i_w, width in enumerate(SHAPES):
        print("i_w = {:2d}/{:3d} - width = {:2d} ".format(i_w, SHAPES.size, width))
        print("compute model...", end="")
        model = bp.CollisionModel(masses=[1],
                                  shapes=[[width, width]],
                                  base_delta=5 / (width - 1)
                                  )
        if width < 10:
            _DEFAULT_WEIGHT = 10 * DEFAULT_WEIGHT
        else:
            _DEFAULT_WEIGHT = DEFAULT_WEIGHT
        print("Done")
        _nd = np.ones(1)
        _temp = np.ones(1) * 2
        angular_weights = np.ones(ndim)
        # prepare initial parameters for bisection
        biscetion_weights = [0.1, 1]
        # compute initial viscosities
        visc = []
        print("Compute Initial Viscosity Coefficients:")
        for w in biscetion_weights:
            model.collision_weights[:] = _DEFAULT_WEIGHT
            angular_weights[ndim-1] = w
            adjust_weight_by_angle(model, angular_weights)
            visc.append([model._cmp_viscosity(
                    number_densities=_nd,
                    temperature=_temp,
                    directions=directions,
                    dt=dt,
                    normalize=True)
                for directions in [[[1, 1], [1, -1]],
                                   [[0, 1], [1, 0]]]])
        print(np.array(visc))
        visc_diff = [visc[i][ndim-2] - visc[i][ndim-1] for i in [0, 1]]
        # the signs must differ, for the bisection scheme
        assert visc_diff[0] * visc_diff[1] < 0
        # the current weights and diffs describe the best results so far
        order = np.argsort(visc_diff)
        # store all weights, divided into upper and lower bounds
        w_lo = [biscetion_weights[np.argmin(visc_diff)]] * 2
        w_hi = [biscetion_weights[np.argmax(visc_diff)]] * 2
        print("Start Bisection Algorithm")
        while True:
            w = (w_lo[-1] + w_hi[-1]) / 2
            biscetion_weights.append(w)
            # compute new viscosity
            model.collision_weights[:] = _DEFAULT_WEIGHT
            angular_weights[ndim-1] = w
            adjust_weight_by_angle(model, angular_weights)
            visc.append([model._cmp_viscosity(
                    number_densities=_nd,
                    temperature=_temp,
                    directions=directions,
                    dt=dt,
                    normalize=True)
                for directions in [[[1, 1], [1, -1]],
                                   [[0, 1], [1, 0]]]])
            visc_diff.append(visc[-1][ndim-2] - visc[-1][ndim-1])
            # update hi/lo lists
            if visc_diff[-1] < 0:
                w_lo.append(w)
                w_hi.append(w_hi[-1])
            else:
                w_hi.append(w)
                w_lo.append(w_lo[-1])
            assert len(w_hi) == len(w_lo)
            assert len(w_hi) == len(visc)
            rdiff = np.max(visc[-1]) / np.min(visc[-1]) - 1
            print("\rWeight = {:10e} - RDiff = {:3e}"
                  " - i = {}".format(w, rdiff, len(visc)),
                  end="")
            if rdiff < rtol:
                break
            if np.allclose(*biscetion_weights[-2:]):
                print("WARNING! Error below specified threshold")
                break
        print("\nstore results in h5py")
        # compute prandtl number
        visc_coeff = model._cmp_viscosity(
            number_densities=_nd,
            temperature=_temp,
            directions=[[0, 1], [1, 0]],
            dt=dt,
            normalize=True)
        heat_coeff = model.cmp_heat_transfer(
            number_densities=_nd,
            temperature=_temp,
            direction=[0, 1],
            dt=dt,
            normalize=True)
        hdf_group["prandtl"][i_w] = visc_coeff / heat_coeff
        file.flush()
    print("\nDONE!\n")
else:
    print("SKIPPED!\n")


print("####################################################\n"
      "#   Plot: Prandtl Dependency on Grid Shape (2D)    #\n"
      "####################################################")
plt.close("all")
fig = plt.figure(constrained_layout=True, figsize=(12.75, 5.25))
ax = fig.add_subplot()

fig.suptitle(r"Effect of Grid Shape on the Prandtl Number after Adjustment",
             fontsize=fs_suptitle)
prandtl = file["prandtl_shape_Effect_2D"]["prandtl"][()]
shapes = file["prandtl_shape_Effect_2D"]["shapes"][()]
ax.plot(shapes,
        prandtl,
        "-o",
        lw=4,
        label="$Pr_1 = Pr_2 = Pr_3$")
ax.legend(loc="upper left", fontsize=fs_legend)
ax.tick_params(axis="x", labelsize=fs_ticks)
ax.tick_params(axis="y", labelsize=fs_ticks)
ax.set_xlabel(r"Grid Width $k$",
              fontsize=fs_label)
ax.set_ylabel(r"Prandtl Number",
              fontsize=fs_label)
ax.grid(linestyle="dotted")
plt.savefig(bp.SIMULATION_DIR + "/phd_viscosity2D_prandtl_shape.pdf")
print("Done!\n")


print("######################################################\n"
      "#   compute Persistence over Altered Temperatures    #\n"
      "######################################################")
N_TEMPS = 50
key = "persistence_over_altered_temperature"
if key not in file.keys():
    hdf_group = file.create_group(key)
    # copy old parameters for viscosity
    temp_orig = temp.copy()
    for ndim in [2,3]:
        print("Setup {}D model...".format(ndim))
        m[ndim].collision_weights[...] = DEFAULT_WEIGHT
        hdf5_adjustment = file["bisection_3D"][str(ndim)][str(ndim - 1)]
        weights = hdf5_adjustment["biscetion_weights"][-1]
        adjust_weight_by_angle(m[ndim], weights)
        temp_range = m[ndim].temperature_range(atol=0.1)
        temperatures = np.linspace(*temp_range, N_TEMPS)

        print("\nSetup hdf groups")
        subgrp = hdf_group.create_group(str(ndim))
        subgrp["viscosities"] = np.zeros((N_TEMPS, ndim))
        subgrp["temperatures"] = temperatures

        visc = np.zeros((N_TEMPS, ndim))
        for i_t, t in enumerate(temperatures):
            print("\r%3d / %3d" % (i_t+1, N_TEMPS), end="")
            temp[ndim][...] = t
            subgrp["viscosities"][i_t] = cmp_visc(ndim)
            file.flush()
    print("\nDONE!\n")
    temp = temp_orig
else:
    print("SKIPPED!\n")

print("#######################################################\n",
      "#     Plot  Persistence over Altered Temperatures     #\n",
      "#######################################################")
plt.close("all")
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25))
fig.suptitle("Viscosity Coefficients for Altered Temperatures",
             fontsize=fs_suptitle)
axes[0].set_title(r"Adjusted 2D Model",
                  fontsize=fs_title)
axes[1].set_title(r"Adjusted 3D Model",
                  fontsize=fs_title)

colors = {3: np.array(["tab:orange",
                       "tab:green",
                       "tab:blue"])}
colors[2] = colors[3][1:]
labels = np.array([r"$\widetilde{\mu}_1$",
                   r"$\widetilde{\mu}_2$",
                   r"$\widetilde{\mu}_3$"])

styles = {3: np.array(["dotted", "dashed", "solid"])}
styles[2] = styles[3][1:]

axes[0].set_ylabel("Viscosity Coefficients", fontsize=fs_label)

key = "persistence_over_altered_temperature"
hdf_group = file[key]
for ax, ndim in enumerate([2, 3]):
    axes[ax].set_xlabel(r"Temperature", fontsize=fs_label)
    visc = hdf_group[str(ndim)]["visc"]
    temperatures = hdf_group[str(ndim)]["temperatures"][()]
    for i in range(ndim):
        axes[ax].plot(temperatures,
                      visc[:, i],
                      color=colors[ndim][i],
                      label=labels[i],
                      ls=styles[ndim][i],
                      lw=3)
    axes[ax].set_ylim(0, 1.5 * np.max(visc))
    axes[ax].tick_params(axis="both", labelsize=fs_ticks)
    axes[ax].legend(loc="upper left", fontsize=fs_legend+2)
    axes[ax].grid(linestyle="dotted")

plt.savefig(bp.SIMULATION_DIR + "/phd_persistence_over_altered_temperature.pdf")
print("Done!\n")


print("#########################################################\n"
      "# compute Persistence over Altered Number Densities (1) #\n"
      "#########################################################")
N_ND = 11
NUMBER_DENSITIES = 2 * np.ones((N_ND, 2))
NUMBER_DENSITIES[:, 1] = np.linspace(0, 1, N_ND)
NUMBER_DENSITIES[:, 0] -= NUMBER_DENSITIES[:, 1]
key = "persistence_over_altered_number_densities_1"
if key not in file.keys():
    hdf_group = file.create_group(key)
    hdf_group["number_densities"] = NUMBER_DENSITIES
    # store nd parameters and reset after computations
    original_nd = nd.copy()
    for ndim in [2,3]:
        print("\nSetup hdf groups")
        subgrp = hdf_group.create_group(str(ndim))
        subgrp["viscosities"] = np.zeros((NUMBER_DENSITIES.shape[0], ndim))
        print("Setup {}D model...".format(ndim))
        m[ndim].collision_weights[...] = DEFAULT_WEIGHT
        weights = np.ones(ndim)
        hdf5_adjustment = file["bisection_3D"][str(ndim)][str(ndim-1)]
        weights[ndim-1] = hdf5_adjustment["biscetion_weights"][-1]
        adjust_weight_by_angle(m[ndim], weights)
        print("Compute Viscosites for Altered Number Densities")
        for i_nd, _nd in enumerate(NUMBER_DENSITIES):
            print("\r{:3d} / {:3d}"
                  "".format(i_nd, NUMBER_DENSITIES.shape[0]),
                  end="")
            nd[ndim] = _nd
            subgrp["viscosities"][i_nd] = cmp_visc(ndim)
            file.flush()
    print("\nDONE!\n")
    nd = original_nd
else:
    print("SKIPPED!\n")

print("##########################################################\n",
      "#     Plot  Persistence over Altered Number Densities    #\n",
      "##########################################################")
plt.close("all")
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25))
fig.suptitle("Viscosity Coefficients in Adjusted DVM "
             + r"for Altered $\nu \in \mathbb{R}^2$",
             fontsize=fs_suptitle)
axes[0].set_title(r"Adjusted 2D Model",
                  fontsize=fs_title)
axes[1].set_title(r"Adjusted 3D Model",
                  fontsize=fs_title)

colors = {3: np.array(["tab:orange", "tab:blue", "tab:green"])}
colors[2] = colors[3][1:]
labels = np.array([r"$\widetilde{\mu}_1$",
                   r"$\widetilde{\mu}_2$",
                   r"$\widetilde{\mu}_3$"])

styles = {3: np.array(["dotted", "dashed", "solid"])}
styles[2] = styles[3][1:]

axes[0].set_ylabel("Viscosity Coefficients", fontsize=fs_label)

key = "persistence_over_altered_number_densities_1"
hdf_group = file[key]
for ax, ndim in enumerate([2, 3]):
    axes[ax].set_xlabel(""
                        r"Specific Number Densities "
                        r"$\nu \in \mathbb{R}^2$",
                        # + r"with $\nu^r = 1$ for $r \neq s$",
                        fontsize=fs_label)
    number_densities = hdf_group["number_densities"][()]
    viscosities = hdf_group[str(ndim)]["viscosities"][()]
    for i in range(ndim):
        visc = viscosities[:, i]
        axes[ax].plot(np.linspace(0, 1, number_densities.shape[0]),
                      visc,
                      color=colors[ndim][i],
                      label=labels[i],
                      ls=styles[ndim][i],
                      lw=3,
                      zorder=5-i)
    axes[ax].set_ylim(0, 1.5 * np.max(viscosities))
    axes[ax].tick_params(axis="both", labelsize=fs_ticks)
    axes[ax].legend(loc="upper center", fontsize=fs_legend+2, ncol=ndim)
    axes[ax].grid(linestyle="dotted")
    axes[ax].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[ax].set_xticklabels(["$(2, 0)$",
                              r"$(1.5, 0.5)$",
                              "$(1, 1)$",
                              "$(0.5, 1.5)$",
                              "$(0, 2)$"])

plt.savefig(bp.SIMULATION_DIR + "/phd_persistence_over_altered_number_densities_1.pdf")
print("Done!\n")


print("########################################################\n"
      "#   Compute: Enforce Invariance for each species pair  #\n"
      "########################################################")
rtol = 1e-3
key = "bisection_3d_pairwise"
if key not in file.keys():
    file.create_group(key)
    for ndim in [2, 3]:
        hdf_group = file[key].create_group(str(ndim))
        print("Apply Bisection to {:1d}D model".format(ndim))
        spc_keys = m[ndim].key_species(m[ndim].collision_relations)[:, 1:3]
        spc_grp = m[ndim].group(spc_keys)
        # reset any remaining weight adjustments
        m[ndim].collision_weights[...] = DEFAULT_WEIGHT
        # process intraspecies collisions first,
        # then harmonize interspecies collisions based on harmonized Single Species
        spc_pairs = [(0, 0), (1, 1), (0, 1)]
        for sp in spc_pairs:
            cur_nd = np.zeros(nd[ndim].shape)
            for i in sp:
                cur_nd[i] = 1
            cur_nd = cur_nd * 2 / np.sum(cur_nd)    # this should have no effect
            subgrp = hdf_group.create_group(str(sp))
            m[ndim].enforce_angular_invariance(
                number_densities=cur_nd,
                temperatures=temp[ndim],
                dt=dt,
                affected_collisions=spc_grp[sp],
                hdf_log=subgrp,
                initial_weights=[0.1, 4.0],
                verbose=True)
    print("\nDone!")
    for ndim in [2, 3]:
        m[ndim].collision_weights[...] = DEFAULT_WEIGHT
else:
    print("SKIPPED!\n")


print("#########################################################\n"
      "# compute Persistence over Altered Number Densities (2) #\n"
      "#########################################################")
N_ND = 11
NUMBER_DENSITIES = 2 * np.ones((N_ND, 2))
NUMBER_DENSITIES[:, 1] = np.linspace(0, 1, N_ND)
NUMBER_DENSITIES[:, 0] -= NUMBER_DENSITIES[:, 1]
print(NUMBER_DENSITIES)
key = "persistence_over_altered_number_densities_2"
if key not in file.keys():
    hdf_group = file.create_group(key)
    hdf_group["number_densities"] = NUMBER_DENSITIES
    # store nd parameters and reset after computations
    for ndim in [2,3]:
        print("\nSetup hdf groups")
        subgrp = hdf_group.create_group(str(ndim))
        subgrp["viscosities"] = np.zeros((NUMBER_DENSITIES.shape[0], ndim))

        # print("Load {}D model...".format(ndim))
        # model_group = file["bisection_3d_pairwise"][str(ndim)]["model"]
        # model = bp.CollisionModel.load(model_group)
        spc_keys = m[ndim].key_species(m[ndim].collision_relations)[:, 1:3]
        spc_grp = m[ndim].group(spc_keys)
        spc_pairs = [(0, 0), (1, 1), (0,1)]
        for pair in spc_pairs:
            affected_collisions = spc_grp[pair]
            weight_group = file["bisection_3d_pairwise"][str(ndim)][str(pair)]
            angular_weights = np.ones(ndim)
            angular_weights[ndim-1] = weight_group["bisection_weights"][-1]
            weight_factors = m[ndim].simplified_angular_weight_adjustment(
                angular_weights,
                m[ndim].collision_relations[affected_collisions])
            adjusted_weights = (weight_factors * DEFAULT_WEIGHT)
            m[ndim].collision_weights[affected_collisions] = adjusted_weights
        m[ndim].update_collisions(m[ndim].collision_relations,
                                  m[ndim].collision_weights)
        print("Compute Viscosites for Altered Number Densities")
        for i_nd, _nd in enumerate(NUMBER_DENSITIES):
            print("\r%3d / %3d"
                  % (i_nd, NUMBER_DENSITIES.shape[0]),
                  end="")
            subgrp["viscosities"][i_nd] = cmp_visc_ext(
                m[ndim],
                _nd,
                temp[ndim]
            )
            file.flush()
    print("\nDONE!\n")
    for ndim in [2, 3]:
        m[ndim].collision_weights[...] = DEFAULT_WEIGHT
else:
    print("SKIPPED!\n")

print("#############################################################\n",
      "#     Plot  Persistence over Altered Number Densities (2)   #\n",
      "#############################################################")
plt.close("all")
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25))
fig.suptitle("Viscosity Coefficients "
             + r"for Altered $\nu \in \mathbb{R}^2$",
             fontsize=fs_suptitle)
axes[0].set_title(r"Adjusted 2D Model",
                  fontsize=fs_title)
axes[1].set_title(r"Adjusted 3D Model",
                  fontsize=fs_title)

colors = {3: np.array(["tab:orange", "tab:blue", "tab:green"])}
colors[2] = colors[3][1:]
labels = np.array([r"$\widetilde{\mu}_1$",
                   r"$\widetilde{\mu}_2$",
                   r"$\widetilde{\mu}_3$"])

styles = {3: np.array(["dotted", "dashed", "solid"])}
styles[2] = styles[3][1:]

axes[0].set_ylabel("Viscosity Coefficients", fontsize=fs_label)

key = "persistence_over_altered_number_densities_2"
hdf_group = file[key]
for ax, ndim in enumerate([2, 3]):
    axes[ax].set_xlabel(r"Temperature", fontsize=fs_label)
    number_densities = hdf_group["number_densities"][()]
    viscosities = hdf_group[str(ndim)]["viscosities"][()]
    for i in range(ndim):
        visc = viscosities[:, i]
        axes[ax].plot(np.linspace(0, 1, number_densities.shape[0]),
                      visc[:number_densities.shape[0]],
                      color=colors[ndim][i],
                      label=labels[i],
                      ls=styles[ndim][i],
                      lw=3,
                      zorder=5-i)
    axes[ax].set_ylim(0, 1.5 * np.max(visc))
    axes[ax].tick_params(axis="both", labelsize=fs_ticks)
    axes[ax].legend(loc="upper center", fontsize=fs_legend+2, ncol=ndim)
    axes[ax].grid(linestyle="dotted")
    axes[ax].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[ax].set_xticklabels(["$(2, 0)$",
                              r"$(1.5, 0.5)$",
                              "$(1, 1)$",
                              "$(0.5, 1.5)$",
                              "$(0, 2)$"])

plt.savefig(bp.SIMULATION_DIR + "/phd_persistence_over_altered_number_densities_2.pdf")
print("Done!\n")

#
# print("########################################################\n"
#       "#   Compute: Enforce Invariance for each species pair  #\n"
#       "########################################################")
# rtol = 1e-3
# key = "bisection_3d_multispecies_pairwise"
# if key not in file.keys():
#     file.create_group(key)
#     # distribution parameters
#     nd3 = np.ones(3)
#     temp3 = np.array([2.25] * 3)
#     print("Generate Collision Model")
#     masses = [2, 3, 4]
#     ndim = 3
#     model = bp.CollisionModel(masses,
#                               [[5] * ndim, [7] * ndim, [7] * ndim],
#                               0.125,
#                               [12, 8, 6],
#                               np.full((len(masses), len(masses)), DEFAULT_WEIGHT),
#                               )
#     print(model.ncols, " total collisions")
#     hdf_group = file[key].create_group(str(ndim))
#     hdf_group.create_group("model")
#     print("Apply Bisection pairwise onto the species")
#     spc_keys = m[ndim].key_species(m[ndim].collision_relations)[:, 1:3]
#     spc_grp = m[ndim].group(spc_keys)
#     # process intraspecies collisions first,
#     # then harmonize interspecies collisions based on harmonized Single Species
#     spc_pairs = [(0, 0), (1, 1), (0, 1)]
#     for (s1, s2) in spc_pairs:
#         subgrp = hdf_group.create_group(str((s1, s2)))
#
#         # Note: the intraspecies collisions are already adjusted
#         # and used when we are adjusting the interspecies collisions
#         sm = m[ndim].submodel(list({s1, s2}))
#         sm_grp = sm.group((sm.key_species(sm.collision_relations)[:, 1:3]))
#         if s1 == s2:
#             cur_spc = (0, 0)
#         else:
#             cur_spc = (0, 1)
#
#         # prepare initial parameters for bisection
#         angular_weights = np.ones(ndim)
#         biscetion_weights = [0.1, 4]
#         print("Compute Initial Viscosity Coefficients:")
#         visc = []
#         for w in biscetion_weights:
#             sm.collision_weights[:] = DEFAULT_WEIGHT
#             angular_weights[ndim-1] = w
#             adjust_weight_by_angle(sm, angular_weights)
#             visc.append(cmp_visc_ext(sm,
#                                      nd[ndim][list({s1, s2})],
#                                      temp[ndim][list({s1, s2})]
#                                      )
#                         )
#         print(np.array(visc))
#         visc_diff = [visc[i][ndim-2] - visc[i][ndim-1] for i in [0, 1]]
#         # the signs must differ, for the bisection scheme
#         print(visc_diff)
#         assert visc_diff[0] * visc_diff[1] < 0
#         # the current weights and diffs describe the best results so far
#         order = np.argsort(visc_diff)
#         # store all weights, divided into upper and lower bounds
#         w_lo = [biscetion_weights[np.argmin(visc_diff)]] * 2
#         w_hi = [biscetion_weights[np.argmax(visc_diff)]] * 2
#         print("Start Bisection Algorithm")
#         while True:
#             w = (w_lo[-1] + w_hi[-1]) / 2
#             biscetion_weights.append(w)
#             angular_weights[ndim - 1] = w
#             # apply new collision weights
#             sm.collision_weights[sm_grp[cur_spc]] = DEFAULT_WEIGHT
#             adjust_weight_by_angle(sm, angular_weights, sm_grp[cur_spc])
#
#             # compute new viscosity
#             visc.append(cmp_visc_ext(sm,
#                                      nd[ndim][list({s1, s2})],
#                                      temp[ndim][list({s1, s2})]
#                                      )
#                         )
#             visc_diff.append(visc[-1][ndim-2] - visc[-1][ndim-1])
#             # update hi/lo lists
#             if visc_diff[-1] < 0:
#                 w_lo.append(w)
#                 w_hi.append(w_hi[-1])
#             else:
#                 w_hi.append(w)
#                 w_lo.append(w_lo[-1])
#             assert len(w_hi) == len(w_lo)
#             assert len(w_hi) == len(visc)
#             rdiff = np.max(visc[-1]) / np.min(visc[-1]) - 1
#             print("\rWeight = {:10e} - RDiff = {:3e}"
#                   " - i = {}".format(w, rdiff, len(visc)),
#                   end="")
#             if rdiff < rtol:
#                 break
#             if np.allclose(*biscetion_weights[-2:]):
#                 print("WARNING! Error below specified threshold")
#                 break
#         print("\nstore results in h5py")
#         pos = len(visc)
#         subgrp["biscetion_weights"] = np.array(biscetion_weights)
#         subgrp["w_hi"] = np.array(w_hi)
#         subgrp["w_lo"] = np.array(w_lo)
#         subgrp["visc"]= np.array(visc)
#         file.flush()
#         print("apply weights to model")
#         adjust_weight_by_angle(m[ndim], angular_weights, spc_grp[cur_spc])
#
#     print("store adjusted model")
#     m[ndim].save(hdf5_group=hdf_group["model"])
#     file.flush()
#
#     print("###########################################################\n"
#           "#    compute angular dependecies for adjusted 3D model    #\n"
#           "###########################################################")
#     rotations = []
#     rotations.append([[1, 0, 0],
#                       [0, 1, 0],
#                       [0, 0, 1]])
#     rotations.append([[1, 0, 0],
#                       [0, 1, 1],
#                       [0, -1, 1]])
#     rotations.append([[-1, 1, 1],
#                       [-1, -1, 1],
#                       [2, 0, 1]])
#     rotations.append([[-3, -2, 1],
#                       [0, 10, 2],
#                       [1, -6, 3]])
#
#     rotations = np.array(rotations, dtype=int)
#     # store integer first angles, to use as keys
#     first_angles = np.copy(rotations[:, :, -1])
#     # change data type to float and normalize each column
#     rotations = np.array(rotations, dtype=float)
#     for a in range(rotations.shape[0]):
#         for col in range(rotations.shape[-1]):
#             rotations[a, :, col] /= np.linalg.norm(rotations[a, :, col])
#     # base angles in xy plane
#     xy_angles = np.zeros((N_ANGLES, 3))
#     ls = np.linspace(0, 2 * np.pi, N_ANGLES)
#     xy_angles[:, 0] = np.cos(ls)
#     xy_angles[:, 1] = np.sin(ls)
#     # compute second angles from rotating base_angles
#     second_angles = np.einsum("abc, dc -> adb", rotations, xy_angles)
#
#     for i_a, a in enumerate(first_angles):
#         subgrp = hdf_group.create_group(str(tuple(a)))
#         ang_visc = np.empty(N_ANGLES, dtype=float)
#         ang_heat = np.empty(N_ANGLES, dtype=float)
#         for n, sa in enumerate(second_angles[i_a]):
#             print("\rangle_1 = %1d / %1d,     angle_2 = %3d / %3d"
#                   % (i_a + 1, first_angles.shape[0], n + 1, N_ANGLES),
#                   end="")
#             ang_visc[n] = model.cmp_viscosity(
#                 number_densities=nd3,
#                 temperature=temp3,
#                 directions=[a, sa],
#                 dt=dt,
#                 normalize=False)
#             ang_heat[n] = model.cmp_heat_transfer(
#                 number_densities=nd3,
#                 temperature=temp3,
#                 direction=sa,
#                 dt=dt,
#                 normalize=False)
#         subgrp["rad_angle"] = ls
#         subgrp["first_angle"] = a
#         subgrp["rotations"] = rotations[i_a]
#         subgrp["angular_visc"] = ang_visc
#         subgrp["angular_heat"] = ang_heat
#         file.flush()
#     print("\nDONE!\n")
# else:
#     print("SKIPPED!\n")
#
#
# print("########################################################\n"
#       "#   Plot: Dependency in adjusted 3D, 3-Species Model   #\n"
#       "########################################################")
# plt.close("all")
# fig = plt.figure(constrained_layout=True, figsize=(12.75, 8))
# ax = fig.add_subplot(projection="polar")
# ax.set_title(r"Angular Dependencies of the Viscosity $\lambda$ in the Adjusted 3D Model",
#              fontsize=fs_title)
#
# key = "bisection_3d_multispecies_pairwise/model"
# styles = ["solid", "solid", "solid", "dashed"]
# colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
# labels = [
#     r"$\lambda\left(\begin{pmatrix}1 \\ 1 \\ 1 \end{pmatrix}, \psi\right)$",
#     r"$\lambda\left(\begin{pmatrix}0 \\ 1 \\ 1 \end{pmatrix}, \psi\right)$",
#     r"$\lambda\left(\begin{pmatrix}0 \\ 0 \\ 1 \end{pmatrix}, \psi\right)$",
#     r"$\lambda\left(\begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}, \psi\right)$",
# ]
# # iterate over this, to have a fixed order!
# a_keys = [(1,1,1), (0,1,1), (0,0,1), (1,2,3)]
# hdf_group = file[key]
# for k, first_angle in enumerate(a_keys):
#     subgrp = hdf_group[str(first_angle)]
#     visc = subgrp["angular_visc"][()]
#     print("Relative Differences for ", first_angle)
#     print("Viscosity: ", np.max(visc) / np.min(visc) - 1)
#     heat = subgrp["angular_heat"][()]
#     print("Heat Conductivity: ", np.max(heat) / np.min(heat) - 1)
#
#     if k == 3:
#         rad = subgrp["rad_angle"][()] + 0.9
#     else:
#         rad = subgrp["rad_angle"][()]
#     ax.plot(rad,
#             visc,
#             ls=styles[k],
#             c=colors[k],
#             label=labels[k],
#             lw=4)
# ax.set_rlabel_position(225)
# # axes[1].set_rlim(0, 1.4)
# # ax.set_rticks([0, 0.00005, 0.0001, 0.00015])
# ax.tick_params(axis="both", labelsize=16)
# ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
#           fontsize=fs_legend)
# ax.grid(linestyle="dotted")
#
# plt.savefig(bp.SIMULATION_DIR + "/phd_bisection3D_multispecies_invariance.pdf")
# print("Done!\n")
#

print("#################################\n"
      "Test Range of Values of Viscosity\n"
      "#################################")
NUMBER_OF_TESTS = 100000000
if COMPUTE["range_test"]:
    print("Compute Minimum and Maximum Values")
    MIN = m[3]._cmp_viscosity(nd[3], temp[3], dt,
                              directions=[[0,0,1], [0,1,0]])
    MAX = m[3]._cmp_viscosity(nd[3], temp[3], dt,
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
        result = m[3]._cmp_viscosity(nd[3], temp[3], dt,
                                     directions=angles)
        CONJECTURE = MIN <= result <= MAX
        if not CONJECTURE:
            print(orig_angles)
            print(angles)
            print(MIN, "\t", result, "\t", MAX, "\n\n")
    print("DONE!\n")
else:
    print("SKIPPED!\n")
