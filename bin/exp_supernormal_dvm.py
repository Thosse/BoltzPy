import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

masses = [1, 1]
shapes = ((3, 3), (3, 3))

# Todo Buggy!
def n_collision_invariants(masses, vels0, vels1=None):
    # construct a matrix with a collision invariant in each line
    matrix = np.zeros((3 + vels0.shape[1], vels0.shape[0] + vels1.shape[0]),
                      dtype=int)
    # number density 0
    matrix[0, 0: vels0.shape[0]] = 1
    # number density 1
    matrix[1, vels0.shape[0]:] = 1
    # total energy
    matrix[2, 0: vels0.shape[0]] = masses[0] * np.sum(vels0**2, axis=1)
    matrix[2, vels0.shape[0]:] = masses[1] * np.sum(vels1**2, axis=1)
    # each components of total momentum is a separate line
    for i in range(vels0.shape[1]):
        matrix[3+i, 0: vels0.shape[0]] = masses[0] * vels0[:, i]
        matrix[3+i, vels0.shape[0]:] = masses[1] * vels1[:, i]
    invariants = np.linalg.matrix_rank(matrix)
    return invariants


model = bp.CollisionModel(masses=masses,
                          shapes=shapes,
                          base_delta=0.5)

print("2d, normal, complete model")
print("number of collision invariants   = ",
      model.collision_invariants)
print("maximal physical Collision Invariants = ",
        n_collision_invariants(
            masses,
            model.vels[model.idx_range(0)],
            model.vels[model.idx_range(1)]))

# group collisions by species
grp_colls = model.group(model.key_species(model.collision_relations),
                        model.collision_relations,)
# model.plot_collisions(relations=grp_colls[(0,0,0,0)],
#                       save_as="normal_25+25_dvm_C00.png",
#                       )
# model.plot_collisions(relations=grp_colls[(1,1,1,1)],
#                       save_as="normal_25+25_dvm_C11.png",
#                       )
# model.plot_collisions(relations=grp_colls[(0,0,1,1)],
#                       save_as="normal_25+25_dvm_C01.png",
#                       )


# delete intraspecies diamond collisions
for k in [(0,0,0,0), (1,1,1,1)]:
    grp_shp = model.group(model.key_shape(grp_colls[k]))
    print(grp_shp)
    removed_colls = np.copy(grp_colls[k][grp_shp[(8,8)]])
    grp_colls[k] = np.delete(grp_colls[k], grp_shp[(8,8)], axis=0)
    # print("Plotting Reduced Model")
    # model.plot_collisions(grp_colls[k])

for k in grp_colls.keys():
    print("species = ", k)
    spc = np.unique(k)

    # reuse grouped collisions
    colls = grp_colls[k]
    assert model.nspc <= 2
    if len(spc) == 1:
        s = spc[0]
        choice = np.where(np.all(colls >= model._idx_offset[s], axis=1)
                          & np.all(colls < model._idx_offset[s+1], axis=1))
        colls = colls[choice] - model._idx_offset[s]
    # construct new model
    new_model = bp.CollisionModel(masses=model.masses[spc],
                                  shapes=model.shapes[spc],
                                  base_delta=model.base_delta,
                                  spacings=model.spacings[spc],
                                  collision_relations=colls)
    print("number of collision invariants   = ",
          new_model.collision_invariants)
    # print("maximal physical Collision Invariants = ",
    #         n_collision_invariants(
    #             masses,
    #             new_model.vels[new_model.idx_range(0)],
    #             new_model.vels[new_model.idx_range(1)]))


# create plot of reduced collisions
# setup plot
fig, ax = plt.subplots(1, 2, constrained_layout=True,  sharey="all",
                       figsize=(12.75, 6.375))
# plot_styles = [{"marker": 'o', "alpha": 0.5, "s": 50},
#                {"marker": 'x', "alpha": 0.9, "s": 50},
#                {"marker": 's', "alpha": 0.5, "s": 50},
#                {"marker": 'D', "alpha": 0.5, "s": 50}]
model.plot_styles[0]["s"] = 250
model.plot_styles[1]["s"] = 250
# plot removed collisions
for col in removed_colls:
    colvel = model.vels[np.tile(col, 2)]
    ax[0].plot(colvel[:, 0], colvel[:, 1], "--", c="red",
               dashes=(8, 18), lw=0.5)

# plot remaining collisions
for i, k in enumerate([(0, 0, 0, 0), (0, 0, 1, 1)]):
    model.plot_collisions(grp_colls[k], plot_object=ax[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
ax[0].set_title("Reduced Intraspecies Collisions $\mathfrak{C}_{\gamma > 0}^{s,s}$ ",
                # "for $s \in \{1,2\}$",
                fontsize=fs_title)
ax[1].set_title("Complete Interspecies Collisions $\mathfrak{C}_{\gamma > 0}^{1,2}$",
                fontsize=fs_title)

plt.savefig(bp.SIMULATION_DIR + "/plot_normal_dvm_2.pdf")
plt.show()

#######################################################################
#       Plot extreme numral/not semi supernormal DVM
#######################################################################
# create plot of reduced collisions
# setup plot
fig, ax = plt.subplots(1, 2, constrained_layout=True,  sharey="all",
                       figsize=(12.75, 6.375))
# plot_styles = [{"marker": 'o', "alpha": 0.5, "s": 50},
#                {"marker": 'x', "alpha": 0.9, "s": 50},
#                {"marker": 's', "alpha": 0.5, "s": 50},
#                {"marker": 'D', "alpha": 0.5, "s": 50}]
model.plot_styles[0]["s"] = 250
model.plot_styles[1]["s"] = 250

# plot remaining collisions
model.plot_collisions([[0,0,0,0]], plot_object=ax[0])
model.plot_collisions(grp_colls[(0,0,1,1)], plot_object=ax[1])
for i in [0,1]:
    ax[i].set_xticks([])
    ax[i].set_yticks([])
ax[0].set_title("Removed Intraspecies Collisions $\mathfrak{C}_{\gamma > 0}^{s,s}$ ",
                # "for $s \in \{1,2\}$",
                fontsize=fs_title)
ax[1].set_title("Complete Interspecies Collisions $\mathfrak{C}_{\gamma > 0}^{1,2}$",
                fontsize=fs_title)

plt.savefig(bp.SIMULATION_DIR + "/plot_normal_dvm_1.pdf")
plt.show()