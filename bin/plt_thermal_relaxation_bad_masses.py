
# Desired Command / Road Map
import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
import copy
import h5py

###########################
#   Control Parameters    #
###########################
EXP_NAME = "/thermal_relaxation"
FILE_ADDRESS = bp.SIMULATION_DIR + EXP_NAME + ".hdf5"
FILE = h5py.File(FILE_ADDRESS, mode="a")

COMPUTE = {"unedited": False,
           "ET": False,
           "All but ET": False,
           "gain": False,
           "gain + ET": False,
           "shape": False,
           "shape + gain + ET": False,
           "masses": False,
           "masses + gain": False
           }

dt = 0.05
linestyles = ["-", ":", "--"]
lw = {"lw": 3}
# setup collision model
masses = [6, 13]
shapes = np.array([(7, 7), (11, 11)])
dv = 0.1
col_factors = np.array([[1, 1], [1, 1]])
spacings = [26, 12]

models = [
        bp.CollisionModel(masses,
                          shapes,
                          dv,
                          spacings,
                          col_factors),
        bp.CollisionModel(masses[:1],
                          shapes[:1],
                          dv,
                          spacings[:1],
                          col_factors[:1, :1]),
        bp.CollisionModel(masses[1:],
                          shapes[1:],
                          dv,
                          spacings[1:],
                          col_factors[1:, 1:])
          ]

##################################################
#       plot energy transferring collisions      #
##################################################
plot_models = [models[0],
               bp.CollisionModel(masses,
                                 shapes + 2,
                                 dv,
                                 spacings,
                                 col_factors)
               ]

print("Create Plot of Collisions")
fig, ax = plt.subplots(1, 2, constrained_layout=True,
                       sharey=True,sharex=True,
                       figsize=(12.75, 6.25))
for i, m in enumerate(plot_models):
    grp = m.group((m.key_species(m.collision_relations)[:, 1:3],
                   m.key_energy_transfer(m.collision_relations)),
                  m.collision_relations)
    model_name = "Small model" if i == 0 else "Larger model"
    print(model_name + ":\t", m.collision_relations.shape[0])
    print("Number of Collisions, by species")
    for key, val in grp.items():
        print(key, val.shape[0])

    # plot only energy transferring Colls
    e_cols = grp[(0, 1, 1)]
    m.plot_collisions(e_cols, plot_object=ax[i])

    # compute number of orbits
    key_orb = m.key_orbit(e_cols)
    print("Number of orbits = ", np.unique(key_orb, axis=0).shape[0])

    # remove ticks
    ax[i].set_aspect('equal')
    ax[i].set_xticks([])
    # ax[i].set_yticks([])
plt.savefig(bp.SIMULATION_DIR + "/thermal_relaxation_bad_masses_grids.pdf")
plt.cla()
print("Done!\n")


#######################################
#       Set up Simulation Cases       #
#######################################
v0 = np.array([0, 0], dtype=float)
v1 = np.array([0, 0], dtype=float)
T0 = 30
T1 = 60

# compute relaxation simulation of annealing temperatures
rules = [bp.HomogeneousRule([1, 1],
                            np.array([v0, v1]),
                            [T0, T1],
                            **models[0].__dict__),
         bp.HomogeneousRule([1],
                            np.array([v0]),
                            [T0],
                            **models[1].__dict__),
         bp.HomogeneousRule([1],
                            np.array([v0]),
                            [T0],
                            **models[2].__dict__),
         ]
# set superposition of maxwellians for monospecies cases
# Note: this gives a very rough comparison, results should NOT match precisely
for i in [1, 2]:
    rules[i].initial_state += models[i].cmp_initial_state([1], np.array([v1]), [T1])
del i

print("Compute Relaxation with uniform weights")
key = "unedited"
if COMPUTE[key] or (key not in FILE):
    # remove old results
    if key in FILE:
        del FILE[key]
    # create new results
    FILE.create_group(key)
    for i, r in enumerate(rules):
        FILE[key][str(i)] = r.compute(dt, 1000000, atol=1e-10, rtol=1e-10)
    print("Done!\n")
else:
    print("Skipped\n")
del key

##########################################################
#       Edit Collision Weights of Energy Collisions      #
##########################################################
# so far all models should have equal weights 1.0
for r in rules:
    assert np.allclose(r.collision_weights, 1)
# copy model/rule for mixture, do adjust weights
r = copy.deepcopy(rules[0])
assert isinstance(r, bp.HomogeneousRule)
ENERGY_FACTORS = [1, 10, 100, 1000]
print("Computing different weights for Energy Transferring Collisions")
key = "ET"
if COMPUTE[key] or (key not in FILE):
    # remove old results
    if key in FILE:
        del FILE[key]
    # create new results
    FILE.create_group(key)
    for ENERGY_FACTOR in ENERGY_FACTORS:
        key_ef = str(ENERGY_FACTOR)
        # no change to previous result, thus reuse it
        if ENERGY_FACTOR == 1:
            FILE[key][key_ef] = FILE["unedited"]["0"]
        else:
            print("Weight factor: ", ENERGY_FACTOR)
            grp = r.group((r.key_species(r.collision_relations)[:, 1:3],
                           r.key_energy_transfer(r.collision_relations)))
            rels = grp[(0, 1, 1)]
            r.collision_weights[rels] = ENERGY_FACTOR
            r.update_collisions(r.collision_relations, r.collision_weights)
            FILE[key][key_ef] = r.compute(dt, 1000000, atol=1e-10, rtol=1e-10)
            print("Time Steps = ", FILE["edit_ET"][key_ef].shape[0])
    print("Done!\n")
    del ENERGY_FACTOR, key_ef
else:
    print("Skipped\n")
del r, key


############################################
#       Create First Relaxation Plot       #
############################################
# determine maximum time range
n_time_steps = [res.shape[0] for res in FILE["unedited"].values()]
print("Number of time steps:\n", n_time_steps, "\n")
max_t = max(n_time_steps) * 2
# multiply by 2, for a little margin in log scale plot
longest_relaxation_time = max_t * dt
time = (np.arange(max_t) + 1) * dt

print("create relaxation plots")
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True,
                               figsize=(12.75, 6.25), sharey=True, sharex=True)
axes = [ax1, ax2]

fig.suptitle("Relaxation of Perturbed Temperatures "
             "for a Mixture with Few Energy Transferring Collisions",
             fontsize=14)
ax1.set_title("Uniform Collision Weights", fontsize=16)
ax2.set_title(r"Increased Weights $\gamma_E$ for Energy Transferring Collisions", fontsize=14)
ax1.set_ylabel(r"$\mathcal{L}^2$-Distance to Equilibrium", fontsize=14)

for ax in [ax1, ax2]:
    ax.set_xlabel("Time", fontsize=14)
    ax.set_xscale("log")
    ax.set_xlim(right=longest_relaxation_time)
    # ax.set_ylim(bottom=0, top=0.03)

# plot left figure (original weights, with references
label_left = [r"Mixture $\mathfrak{S} = \{1,2\}$",
              r"$\mathfrak{S}=\{1\}$, $n^1= (7, 7)$",
              r"$\mathfrak{S}=\{2\}$, $n^2= (11, 11)$"]
for i_r in range(len(rules)):
    raw = FILE["unedited"][str(i_r)][()]
    res = np.sum((raw - raw[-1])**2, axis=-1)
    ax1.plot(time[:len(res)],
             res,
             linestyle=linestyles[i_r],
             label=label_left[i_r],
             **lw)
ax1.legend(fontsize=12, loc="upper right")

# plot right plot (increased energy transfer rates)
label_right = [r"$\gamma_{E} = $" + str(etf)
               for etf in ENERGY_FACTORS]
colors = ["tab:blue", "gold", "tab:purple", "tab:red"]
for i_ef, ef in enumerate(ENERGY_FACTORS):
    raw = FILE["edit_ET"][str(ef)][()]
    res = np.sum((raw - raw[-1])**2, axis=-1)
    ax2.plot(time[:len(res)],
             res,
             color=colors[i_ef],
             label=label_right[i_ef],
             **lw)
ax2.legend(fontsize=12, loc="upper right")

# add greyed out reference plots
for i_r in range(len(rules)):
    raw = FILE["unedited"][str(i_r)][()]
    res = np.sum((raw - raw[-1])**2, axis=-1)
    ax1.plot(time[:len(res)],
             res,
             linestyle=linestyles[i_r],
             color="darkgray",
             label="_nolegend_",
             **lw)
plt.savefig(bp.SIMULATION_DIR
            + "/thermal_relaxation_bad_masses_1.pdf")
print("Done!")
plt.cla()
del i_r, i_ef, raw, res


######################################################
#      Next Plot: Adjusting All Weights, but ET      #
######################################################
# Edit Collision Weights for left model
# all initial weights must be 1, as in the first relaxation
assert all(np.allclose(r.collision_weights, 1)
           for r in rules)
print("Balance weights by gain terms, based on number density.....")
# use energy gains for weight adjustments
# mom_func = [r.cmp_energy_density for r in rules]
mom_func = [r.cmp_number_density for r in rules]
# base weight is set to gains of Interspecies Collisions
r = rules[0]
grp = r.group(r.key_species(r.collision_relations)[:, 1:3])
is_rels = grp[(0, 1)]
BASE_WEIGHT = mom_func[0](r.gain_term(is_rels))
print("Compute new gain based weights by species")
for i_r, r in enumerate(rules):
    print("Rule ", i_r, ":")
    grp = r.group(r.key_species(r.collision_relations)[:, 1:3])
    for g, rels in grp.items():
        # compute gain term
        gains = mom_func[i_r](r.gain_term(rels))
        new_weight = BASE_WEIGHT / gains
        print(g, new_weight)
        r.collision_weights[rels] = new_weight
    r.update_collisions(r.collision_relations, r.collision_weights)
assert np.allclose(rules[0].collision_weights[is_rels], 1)
print("Done!\n")

print("Compute Relaxation with gain adjusted weights")
key = "gain"
if COMPUTE[key] or (key not in FILE):
    # remove old results
    if key in FILE:
        del FILE[key]
    # create new results
    FILE.create_group(key)
    for i_r, r in enumerate(rules):
        FILE[key][str(i_r)] = r.compute(dt, 1000000, atol=1e-10, rtol=1e-10)
    print("Done!\n")
else:
    print("Skipped\n")

print("Increase Collision weights for all Collisions, except ET")
for r in rules:
    r.collision_weights *= 10
    r.update_collisions(r.collision_relations, r.collision_weights)
# do not increase ET weights
r = rules[0]
grp = r.group((r.key_species(r.collision_relations)[:, 1:3],
               r.key_energy_transfer(r.collision_relations)))
et_rels = grp[(0, 1, 1)]
r.collision_weights[et_rels] = 1
r.update_collisions(r.collision_relations, r.collision_weights)
print("Done!\n")


print("Compute Relaxation with increased weights")
key = "gain + ET"
if COMPUTE[key] or (key not in FILE):
    # remove old results
    if key in FILE:
        del FILE[key]
    # create new results
    FILE.create_group(key)
    for i_r, r in enumerate(rules):
        FILE[key][str(i_r)] = r.compute(dt, 1000000, atol=1e-10, rtol=1e-10)
    del i_r
    print("Done!\n")
else:
    print("Skipped\n")
del r, key, grp

#############################################
#       Create second Relaxation Plot       #
#############################################
# create relaxation plots
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True,
                               figsize=(12.75, 6.25), sharey=True, sharex=True)
axes = [ax1, ax2]

ax1.set_title("Gain-Adjusted Collision Weights", fontsize=14)
ax2.set_title("Increased Weights For All Non-Energy-Transferring Collisions", fontsize=14)
ax1.set_ylabel(r"$\mathcal{L}^2$-Distance to Equilibrium", fontsize=14)

for ax in [ax1, ax2]:
    ax.set_xlabel("Time", fontsize=14)
    ax.set_xscale("log")
    ax.set_xlim(right=longest_relaxation_time)
    # ax.set_ylim(bottom=0, top=0.03)


# plot left figure (gain adjusted weights)
label_left = [r"Mixture $\mathfrak{S} = \{1,2\}$",
              r"$\mathfrak{S}=\{1\}$, $n^1= (7, 7)$",
              r"$\mathfrak{S}=\{2\}$, $n^2= (11, 11)$"]
for i_r in range(len(rules)):
    raw = FILE["gain"][str(i_r)][()]
    res = np.sum((raw - raw[-1])**2, axis=-1)
    ax1.plot(time[:len(res)],
             res,
             linestyle=linestyles[i_r],
             label=label_left[i_r],
             **lw)
ax1.legend(fontsize=12, loc="upper right")

# plot right plot (all weights but ET increased)
for i_r in range(len(rules)):
    raw = FILE["gain + ET"][str(i_r)][()]
    res = np.sum((raw - raw[-1])**2, axis=-1)
    ax2.plot(time[:len(res)],
             res,
             linestyle=linestyles[i_r],
             **lw)

plt.savefig(bp.SIMULATION_DIR
            + "/thermal_relaxation_bad_masses_2.pdf")
print("Done!\n")
plt.cla()
del i_r, raw, res


#################################
#       Use larger models       #
#################################
print("Generate new models, with larger shaped grids")
# compute key_center_of_gravity old ecols
# remove new ET Collisions in rules[3]
m = models[0]
grp = m.group((m.key_species(m.collision_relations)[:, 1:3],
               m.key_energy_transfer(m.collision_relations)),
              as_dict=True)
# get an id for the old ET Collisions
e_cols = m.collision_relations[grp[(0,1,1)]]
old_cog = m.key_center_of_gravity(e_cols)[0]
old_colshape = m.key_shape(e_cols)[0]
old_col_id = tuple(old_cog) + tuple(old_colshape)
del m, old_cog, old_colshape

# use bigger shaped grids for new models
new_shapes = shapes + 2
# to have (roughly) the same domain, the dv is reduced
# since the two grids grow at a different percentage, this is not perfetly equal!
new_dv = 7.8 / 10.4 * dv
models = [
        bp.CollisionModel(masses,
                          new_shapes,
                          dv,
                          spacings,
                          col_factors),
        bp.CollisionModel(masses,
                          new_shapes,
                          new_dv,
                          spacings,
                          col_factors),
        bp.CollisionModel(masses[:1],
                          new_shapes[:1],
                          new_dv,
                          spacings[:1],
                          col_factors[:1, :1]),
        bp.CollisionModel(masses[1:],
                          new_shapes[1:],
                          new_dv,
                          spacings[1:],
                          col_factors[1:, 1:])
          ]

print("Done!\n")

print("Set up Homogeneous Rules for new Models")
rules = [bp.HomogeneousRule([1, 1],
                            np.array([v0, v1]),
                            [T0, T1],
                            **models[0].__dict__),
         bp.HomogeneousRule([1, 1],
                            np.array([v0, v1]),
                            [T0, T1],
                            **models[1].__dict__),
         bp.HomogeneousRule([1, 1],
                            np.array([v0, v1]),
                            [T0, T1],
                            **models[1].__dict__),
         bp.HomogeneousRule([1],
                            np.array([v0]),
                            [T0],
                            **models[2].__dict__),
         bp.HomogeneousRule([1],
                            np.array([v0]),
                            [T0],
                            **models[3].__dict__),
         ]

# set superposition of maxwellians for monospecies cases
# Note: this gives a very rough comparison, results should NOT match precisely
for r in rules[-2:]:
    r.initial_state += r.cmp_initial_state([1], np.array([v1]), [T1])
del r
print("Done!\n")

print("Remove new ET COllisions in Rules[3]")
# find old ET Collisions in rules[3]
r = rules[2]
grp = r.group((r.key_species(r.collision_relations)[:, 1:3],
               r.key_energy_transfer(r.collision_relations),
               r.key_center_of_gravity(r.collision_relations),
               r.key_shape(r.collision_relations)),
              as_dict=True)
old_et_pos = grp[(0,1,1) + old_col_id]
# find all new ET Collisions in rules[3]
grp = r.group((r.key_species(r.collision_relations)[:, 1:3],
               r.key_energy_transfer(r.collision_relations)),
              as_dict=True)
# get all new et_rolls
new_et_pos = grp[(0,1,1)]
# set weight to 0 for all new ones, keep old ones
r.collision_weights[new_et_pos] = 0
r.collision_weights[old_et_pos] = 1
r.update_collisions(r.collision_relations, r.collision_weights)
del r, grp, old_et_pos, new_et_pos
print("Done!\n")

print("Compute Relaxation with larger models")
results_3 = [[], []]
results_3[0] = [r.compute(dt, 1000000, atol=1e-10, rtol=1e-10)
                for r in rules]
print("Done!\n")

#####################################
#       Use better mass ratio       #
#####################################
masses = [6, 12]
spacings = [24, 12]
print("Generate new models, with better mass ratio m = ", masses)
models = [
        bp.CollisionModel(masses,
                          shapes,
                          dv,
                          spacings,
                          col_factors),
        bp.CollisionModel(masses[:1],
                          shapes[:1],
                          dv,
                          spacings[:1],
                          col_factors[:1, :1]),
        bp.CollisionModel(masses[1:],
                          shapes[1:],
                          dv,
                          spacings[1:],
                          col_factors[1:, 1:])
          ]
print("Done!\n")

print("Set up Homogeneous Rules for new Models")
rules = [bp.HomogeneousRule([1, 1],
                            np.array([v0, v1]),
                            [T0, T1],
                            **models[0].__dict__),
         bp.HomogeneousRule([1],
                            np.array([v0]),
                            [T0],
                            **models[1].__dict__),
         bp.HomogeneousRule([1],
                            np.array([v0]),
                            [T0],
                            **models[2].__dict__),
         ]
# set superposition of maxwellians for monospecies cases
# Note: this gives a very rough comparison, results should NOT match precisely
for i in [1, 2]:
    rules[i].initial_state += models[i].cmp_initial_state([1], np.array([v1]), [T1])
print("Done!\n")

print("Get number of Collisions, by species")
m = models[0]
grp = m.group(m.key_species(m.collision_relations)[:, 1:3])
for g, v in grp.items():
    print(g, v.shape[0])

grp = m.group((m.key_species(m.collision_relations)[:, 1:3],
               m.key_energy_transfer(m.collision_relations)),
              m.collision_relations)
print("Number of Energy transferring Collisions = ",
      grp[(0,1,1)].shape[0])
key_orb = m.key_orbit(grp[(0,1,1)])
print("Number of orbits = ", np.unique(key_orb, axis=0).shape[0])
print("Done!\n")

print("Compute Relaxation with adjusted masses")
results_3[1] = [r.compute(dt, 1000000, atol=1e-10, rtol=1e-10)
                for r in rules]
print("Done!\n")


############################################
#       Create third Relaxation Plot       #
############################################
# determine maximum time range
n_time_steps = [res.shape[0]
                for run in results_3
                for res in run]
print("Number of time steps:\n", n_time_steps)
max_t = max(n_time_steps)
time = (np.arange(max_t) + 1)

# create relaxation plots
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True,
                               figsize=(12.75, 6.25), sharey=True, sharex=True)
axes = [ax1, ax2]

ax1.set_title("DVM With Increased Grid Shapes $(9,9)$ and $(13,13)$", fontsize=16)
ax2.set_title(r"DVM With Adjusted Masses $m=[6, 12]$", fontsize=14)
ax1.set_ylabel(r"$\mathcal{L}^2$-Distance to Equilibrium", fontsize=14)

for ax in [ax1, ax2]:
    ax.set_xlabel("Time", fontsize=14)
    ax.set_xscale("log")
    ax.set_xlim(right=longest_relaxation_time)
    # ax.set_ylim(bottom=0, top=0.03)


# plot left figure (gain adjusted weights)
label_left = [r"Mixture $\mathfrak{S} = \{1,2\}$"
              + "\nwith original spacings",
              r"Mixture, reduced $\Delta_\mathbb{R}$",
              r"Mixture, reduced $\Delta_\mathbb{R}$,"
              + "\nonly old ET collisions",
              r"$\mathfrak{S}=\{1\}$, $n^1= (9, 9)$"
              + "\nwith reduced " + r"$\Delta_\mathbb{R}$",
              r"$\mathfrak{S}=\{2\}$, $n^2= (13, 13)$"
              + "\nwith reduced " + r"$\Delta_\mathbb{R}$"]
# keep linestyle for both mixtures
new_linestyle = 2*linestyles[:1] + linestyles
colors = ["tab:blue", "tab:red", "tab:purple", "tab:orange", "tab:green"]
for i, res in enumerate(results_3[0]):
    res = np.sum((res - res[-1])**2, axis=-1)
    ax1.plot(time[:len(res)] * dt,
             res,
             linestyle=new_linestyle[i],
             color=colors[i],
             label=label_left[i],
             **lw)
ax1.legend(fontsize=12, loc="upper right")

label_right = [r"Mixture $\mathfrak{S} = \{1,2\}$",
              r"$\mathfrak{S}=\{1\}$, $n^1= (7, 7)$",
              r"$\mathfrak{S}=\{2\}$, $n^2= (11, 11)$"]
# plot right plot (all weights but ET increased)
for i, res in enumerate(results_3[1]):
    res = np.sum((res - res[-1])**2, axis=-1)
    ax2.plot(time[:len(res)] * dt,
             res,
             linestyle=linestyles[i],
             label=label_right[i],
             **lw)
ax2.legend(fontsize=12, loc="upper right")

plt.savefig(bp.SIMULATION_DIR
            + "/thermal_relaxation_bad_masses_3.pdf")
print("Done!\n")
plt.cla()
del results_3
