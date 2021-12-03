import numpy as np
import boltzpy as bp

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks, fs_legend_title
import h5py


##############################
#   Generation Parameters    #
##############################
FILENAME = "/phd_number_of_collisions_for_spacing.hdf5"
# parameters: fixed masses
FIX_MASSES = np.array([25, 30])
MAX_SHAPE = 31
SHAPES = np.zeros((MAX_SHAPE - 2, 2), dtype=int)
SHAPES[...] = np.arange(3, MAX_SHAPE + 1)[:, None]

# parameters: fixed shape
FIX_SHAPE = 31
MAX_MASS = 30
MASSES = np.full((MAX_MASS, 2), MAX_MASS, dtype=int)
MASSES[:, 0] = np.arange(1, MAX_MASS + 1)
FORCE_COMPUTE = False


###########################
#   Compute Fix Masses    #
###########################
FILE = h5py.File(bp.SIMULATION_DIR + FILENAME, mode="a")

MASS_STR = "Masses = " + str(tuple(FIX_MASSES))
if MASS_STR in FILE.keys() and not FORCE_COMPUTE:
    h5py_group = FILE[MASS_STR]
else:
    # remove existing group
    if MASS_STR in FILE.keys():
        del FILE[MASS_STR]
    # set up new groups
    FILE.create_group(MASS_STR)

    for spacings in [None, [2, 2]]:
        if spacings is None:
            h5py_group = FILE[MASS_STR].create_group("Adjusted Spacings")
        else:
            h5py_group = FILE[MASS_STR].create_group("Equal Spacings")
        # set up data sets
        h5py_group["ncols"] = np.full((SHAPES.shape[0], 1, 2),
                                      -1,
                                      dtype=int)
        h5py_group["is_supernormal"] = np.full(SHAPES.shape[0],
                                                -1,
                                                dtype=int)
        # Compute number of Collisions (and if supernormal)
        for i, cur_shape in enumerate(SHAPES):
            print("\rComputing shape ", cur_shape, end="")
            model = bp.CollisionModel(FIX_MASSES,
                                      np.full((2, 2), cur_shape),
                                      spacings=spacings,
                                      setup_collision_matrix=False)
            # get collision numbers, grouped by species
            grp = model. group(model.key_species(model.collision_relations)[:, 1:3],
                               as_dict=True)
            for (s1, s2) in [(0, 0), (0, 1)]:
                try:
                    h5py_group["ncols"][i, s1, s2] = grp[(s1, s2)].shape[0]
                except KeyError:
                    h5py_group["ncols"][i, s1, s2] = 0
            del grp
            # check if model is supernormal (exists an energy transferring collision)
            transfers_energy = model.key_energy_transfer(model.collision_relations,
                                                         as_bool=True)
            is_supernormal = np.any(transfers_energy)
            h5py_group["is_supernormal"][i] = is_supernormal
            FILE.flush()
        print("")


###########################
#   Compute Fix Shapes    #
###########################
SHAPE_STR = "Shape = " + str(FIX_SHAPE)
if SHAPE_STR in FILE.keys() and not FORCE_COMPUTE:
    h5py_group = FILE[SHAPE_STR]
else:
    # remove existing group
    if SHAPE_STR in FILE.keys():
        del FILE[SHAPE_STR]
    # set up new groups
    FILE.create_group(SHAPE_STR)

    for spacings in [None, [2, 2]]:
        if spacings is None:
            h5py_group = FILE[SHAPE_STR].create_group("Adjusted Spacings")
        else:
            h5py_group = FILE[SHAPE_STR].create_group("Equal Spacings")
        # set up data sets
        h5py_group["ncols"] = np.full((MASSES.shape[0], 1, 2),
                                      -1,
                                      dtype=int)
        h5py_group["is_supernormal"] = np.full(MASSES.shape[0],
                                                -1,
                                                dtype=int)
        # Compute number of Collisions (and if supernormal)
        for i, cur_masses in enumerate(MASSES):
            print("\rComputing masses ", cur_masses, end="")
            model = bp.CollisionModel(cur_masses,
                                      np.full((2, 2), FIX_SHAPE),
                                      spacings=spacings,
                                      setup_collision_matrix=False)
            # get collision numbers, grouped by species
            grp = model. group(model.key_species(model.collision_relations)[:, 1:3],
                               as_dict=True)
            for (s1, s2) in [(0, 0), (0, 1)]:
                try:
                    h5py_group["ncols"][i, s1, s2] = grp[(s1, s2)].shape[0]
                except KeyError:
                    h5py_group["ncols"][i, s1, s2] = 0
            del grp
            # check if model is supernormal (exists an energy transferring collision)
            transfers_energy = model.key_energy_transfer(model.collision_relations,
                                                         as_bool=True)
            is_supernormal = np.any(transfers_energy)
            h5py_group["is_supernormal"][i] = is_supernormal
            FILE.flush()
        print("")
FILE.close()

############################
#       create  plots      #
############################
# read results
FILE = h5py.File(bp.SIMULATION_DIR + FILENAME, mode="r")
# setup plot
fig, ax = plt.subplots(1, 2,
                       figsize=(12.75, 6.25))
# plot number of collision numbers, for fixed masses
res = {0: FILE[MASS_STR],
       1: FILE[SHAPE_STR]}
x_vals = {0: SHAPES,
          1: MASSES}
x_label = {0: r"Grid Widths $n^1_i = n^2_i$",
           1: r"Mass $m^1$ for $m^2 = " + str(MAX_MASS) + "$"}
color = ["tab:blue", "tab:green"]
for a in [0, 1]:
    x = x_vals[a][:, 0]
    ax[a].set_yscale("log")
    ax[a].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                     linewidth=0.4)
    ax[a].xaxis.grid(color='darkgray', linestyle='dashed', which="major",
                     linewidth=0.4)
    ax[a].set_xlabel(x_label[a], fontsize=fs_label)
    ax[0].set_ylabel(r"Number of Collisions", fontsize=fs_label)
    ax[a].tick_params(axis="both", labelsize=fs_ticks)
    ax[a].set_xticks(range(5, 31, 5))
    # ax[d].set_xticklabels(SHAPES[d + 2][:max_idx[d], 0][plt_beg[c][d]::plt_spacing[c][d]])

    # plot number of interspecies collisions
    for k, key in enumerate(["Adjusted Spacings", "Equal Spacings"]):
        y = res[a][key]["ncols"][:, 0, 1]
        y = np.where(y == 0, None, y)
        label = key if a == 0 else None
        ax[a].plot(x, y,  c=color[k], label=label)

    # plot intraspecies collisions, does not mather which spacing
    y = res[a]["Adjusted Spacings"]["ncols"][:, 0, 0]
    y = np.where(y == 0, None, y)
    label = "Intraspecies Collisions" if a == 0 else None
    ax[a].plot(x, y, c="tab:red", label=label)

    # add scatter plot where model is supernormal
    for k, key in enumerate(["Adjusted Spacings", "Equal Spacings"]):
        y = res[a][key]["ncols"][:, 0, 1]
        choice = res[a][key]["is_supernormal"][()]
        y = np.where(choice, y, None)
        x = np.where(choice, x, None)
        ax[a].scatter(x, y,  c=color[k])
ax[0].scatter([], [],  c="gray", label="Supernormal DVM")
# fig.legend(loc="lower center", cols=3, fontsize=fs_legend)
lg = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4,
                fontsize=fs_legend+1
                )
ax[0].set_title(r"Fixed Masses $m = (25,30)$",
                fontsize=fs_title)
ax[1].set_title(r"Fixed Shapes $n^1 = n^2 = (31, 31)$",
                fontsize=fs_title)
st = fig.suptitle("Number of Collisions and Supernormality for Equal and Mass Adjusted Spacings",
                  fontsize=fs_suptitle)
plt.subplots_adjust(top=0.85)
plt.savefig(bp.SIMULATION_DIR + "/phd_number_of_collisions_for_spacing.pdf",
            bbox_extra_artists=(lg, st ),
            bbox_inches='tight')

##################################################################################
#           Local Symmetries in mixtures with unequal spacings                   #
##################################################################################
fig = plt.figure(figsize=(12.75, 6.25))
ax = fig.add_subplot()

model = bp.CollisionModel([2, 3],
                          [(4, 4), (6, 6)],
                          2.0)

ax = model.plot_collisions(plot_object=ax)
col_black = np.array(
    [[6, -6],
     [4,-4],
     [-4, -4],
     [-6, -6],
     [6, -6],
     [6, 6],
     [4, 4],
     [4, -4],
     [6, -6]],
    dtype=float
)
ax.plot(col_black[:, 0],
        col_black[:, 1],
        color="black",
        label="Existing Collisions,\nAll Velocities On Grid")
red_shift = col_black + np.array([[-12, 12]])
ax.plot(red_shift[:, 0], red_shift[:, 1],
        color="red",
        linewidth=2,
        linestyle="dotted",
        # dashes=(20, 15)
        )
red_rotate = np.array(
    [[6, -6],
     [-6, -6],
     [-4, -8],
     [4, -8],
     [6, -6],
     [18, -6],
     [16, -8],
     [8, -8],
     [6, -6],
     [18, -6],
     [16, -4],
     [8, -4],
     [6, -6],
     [6, 6],
     [8, 4],
     [8, -4],
     [6, -6],
     [8, -8],
     [8, -16],
     [6, -18],
     [6, -6],
     [4, -8],
     [4, -16],
     [6, -18],
     [6, -6],
     ],
    dtype=float
)
ax.plot(red_rotate[:, 0], red_rotate[:, 1],
        color="red",
        linewidth=2,
        linestyle="dotted",
        label="Missing Collisions,\nSome Velocities Off Grid"
        )
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_title(r"Missing Local Symmetries in a Mixture",
             fontsize=fs_title)
ax.legend(loc="upper right",
          fontsize=fs_legend)

plt.tight_layout()
plt.savefig(bp.SIMULATION_DIR + "/phd_local_symmetries_mixture.pdf")
# plt.show()
