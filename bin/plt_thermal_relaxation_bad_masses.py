
# Desired Command / Road Map
import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt


dt = 0.5
linestyles = ["-", ":", "--"]
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

# compute relaxations,
print("Compute Relaxation with uniform weights")
results = [[], []]
results[0] = [r.compute(dt, 1000000, atol=1e-10, rtol=1e-10)
              for r in rules]
print("Done!\n")

######################################
#       Edit Collision Weights       #
######################################
print("Balance weights by gain terms, based on number density.....")
# base weight is set by hand, to roughly math the previous relaxation times
BASE_WEIGHT = 7.6   # == gain term of interspecies collisions
# energy factor is multiplied onto all enrgy transferring collisions
# here, it is set by hand as well
ENERGY_FACTOR = 10
print("Computed gain terms (by species) are:")
for i_r, r in enumerate(rules):
    print("Rule ", i_r, ":")
    # group by species
    grp = r.group(r.key_species(r.collision_relations)[:, 1:3])
    for g, rels in grp.items():
        # compute gain term
        gain_terms = r.gain_term(rels)
        nd_gains = r.cmp_energy_density(gain_terms)
        print(g, nd_gains)
        r.collision_weights[rels] = BASE_WEIGHT / nd_gains
    r.update_collisions(r.collision_relations, r.collision_weights)
print("Done!\n")

# # increase energy transfer
# print("Drastically increase weight of energy transferring collisions")
# for r in [rules[0]]:
#     grp = r.group((r.key_species(r.collision_relations)[:, 1:3],
#                    r.key_energy_transfer(r.collision_relations)))
#     rels = grp[(0,1,1)]
#     # gain_terms = np.sum(rules[0].gain_term("number_density", rels))
#     # print(g, gain_terms)
#     # r.collision_weights[rels] = 1.75 / gain_terms
#     r.collision_weights[rels] = ENERGY_FACTOR
#     r.update_collisions(r.collision_relations, r.collision_weights)
# print("Done!\n")

#####################################
#       Recompute Relaxations       #
#####################################
# compute relaxations
print("Compute Relaxation with adjusted weights")
results[1] = [r.compute(dt, 1000000, atol=1e-10, rtol=1e-10)
              for r in rules]
print("Done!\n")

################################
#       Plot Relaxations       #
################################
# determine maximum time range
n_time_steps = [res.shape[0]
                for run in results
                for res in run]
print("Number of time steps:\n", n_time_steps)
max_t = max(n_time_steps)
time = (np.arange(max_t) + 1) * dt

# create relaxation plots
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True,
                               figsize=(12.75, 6.25), sharey=True, sharex=True)
axes = [ax1, ax2]
label = ["Mixture $\mathfrak{S} = \{1,2\}$",
         "$\mathfrak{S}=\{1\}$, $n^1= (7, 7)$",
         "$\mathfrak{S}=\{2\}$, $n^2= (11, 11)$"]
# for l, length in enumerate([300, -1]):
#     for i, res in enumerate(results):
#         model = models[i]
#         axes[l].plot(time[:length]+1, np.max(np.abs(res[:length] - res[-1]), axis=-1),
#                      linestyle=linestyles[i],
#                      label=label[i])

fig.suptitle("Relaxation of Perturbed Temperatures for  a Mixture with Few Energy Transferring Collisions", fontsize=14)
ax1.set_title("Uniform Collision Weights")
ax2.set_title("Gain-Term--Adjusted Collision Weights")
ax1.set_ylabel("$\mathcal{L}^2$-Distance to Equilibrium", fontsize=14)
for ax in [ax1, ax2]:
    ax.set_xlabel("Time", fontsize=14)
    ax.set_xscale("log")
    # ax.set_xlim(left=1, right=max_t+1)
    # ax.set_ylim(bottom=0, top=0.03)

DRAW_ADDITIONAL_ZEROS = False
for k, run in enumerate(results):
    for i, res in enumerate(run):
        if DRAW_ADDITIONAL_ZEROS:
            _time = time
            _results = np.zeros(max_t, dtype=float)
            _results[:len(res)] = np.sum((res - res[-1]) ** 2, axis=-1)
        else:
            _time = time[:len(res)]
            _results = np.sum((res - res[-1])**2, axis=-1)
        print(k, i, np.max(_results), np.min(_results))
        axes[k].plot(_time,
                     _results,
                     linestyle=linestyles[i],
                     label=label[i])
ax2.legend(fontsize=12)
plt.savefig(bp.SIMULATION_DIR
            + "/thermal_relaxation_bad_masses.pdf")
print("Done!")
plt.show()
plt.cla()

