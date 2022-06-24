import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
import h5py
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks, fs_legend_title

EXP_NAME = bp.SIMULATION_DIR + "/number_of_energy_transferring_collisions"
FILE_ADDRESS = EXP_NAME + ".hdf5"
FILE = h5py.File(FILE_ADDRESS, mode="a")
FORCE_COMPUTATION = False

MAX_MASS = 30
USED_SHAPE = MAX_MASS + (MAX_MASS + 1) % 2
MASSES = np.array([[i, MAX_MASS] for i in range(1, 2*MAX_MASS+1)])
SHAPE = np.full((2, 2), USED_SHAPE, dtype=int)
print("Grid Shapes = ", SHAPE)
print("Masses = ", MASSES[:, 0])
BAR_WIDTH = 0.90

key = str(MAX_MASS) + "/" + str(SHAPE[0])
# store all keys of FILE, to not recompute results
all_keys = []
FILE.visit(all_keys.append)


print("compute number of ET and NET Collisions")
if FORCE_COMPUTATION and key in all_keys:
    del FILE[key]

if (key not in all_keys) or FORCE_COMPUTATION:
    group = FILE.create_group(key)
    for spacings, mode in [([2,2], "_eq"), (None, "_neq")]:
        et = group.create_dataset("ET" + mode, shape=MASSES.shape[0], dtype=int)
        net = group.create_dataset("NET" + mode, shape=MASSES.shape[0], dtype=int)
        for i_m, masses in enumerate(MASSES):
            print("\rmode = %s: %2d / %2d" % (mode, masses[0], MASSES[-1, 0]), end="")
            m = bp.CollisionModel(masses=masses,
                                  shapes=SHAPE,
                                  spacings=spacings,
                                  setup_collision_matrix=False)
            # get Energy transferring Collisions
            grp = m.group((m.key_species(m.collision_relations)[:, 1:3],
                           m.key_energy_transfer(m.collision_relations,
                                                 as_bool=True)),)
            try:
                et[i_m] = grp[(0, 1, 1)].shape[0]
            except KeyError:
                et[i_m] = 0
            try:
                net[i_m] = grp[(0, 1, 0)].shape[0]
            except KeyError:
                net[i_m] = 0
            del m

        print("\n")

print("setup figure and axes")
fig, axes = plt.subplots(1, 2, constrained_layout=True,
                         figsize=(12.75, 6.25),
                         sharey=True, sharex=True)
fig.suptitle(r"Interspecies Collisions for Different Masses $(m^0, m^1)$ in $(31, 31)$ shaped Grids",
             fontsize=fs_suptitle)
axes[0].set_title(r"$(\Delta_\mathbb{N}^0, \Delta_\mathbb{N}^1) = (2m^1, 2m^0)$",
                  fontsize=fs_title)
axes[1].set_title(r"$(\Delta_\mathbb{N}^0, \Delta_\mathbb{N}^1) = (2, 2)$",
                  fontsize=fs_title)
axes[0].set_ylabel(r"Number of Interspecies Collisions $\left\lvert \mathfrak{C}^{0,1} \right\rvert $",
                   fontsize=fs_label)
for a, ax in enumerate(axes):
    mode = ["_neq", "_eq"][a]
    et = FILE[key]["ET" + mode][()]
    net = FILE[key]["NET" + mode][()]
    print("ET = ", et)
    print("NET = ", net)
    masses = MASSES[:, 0]
    ax.bar(x=masses,
           height=net,
           width=BAR_WIDTH,
           bottom=0.0,
           color="tab:orange",
           label="NET Collisions")
    ax.bar(x=masses,
           bottom=net,
           width=BAR_WIDTH,
           height=et,
           color="tab:green",
           label="ET Collisions")
    # # create twin axis, for second plot
    # # plot percentage et / net here
    # twax = ax.twinx()
    # percentage = np.where((et + net) == 0, 0.0, et / (et + net))
    # twax.plot(masses,
    #           percentage,
    #           label="Energy Transferring Collisions")
    # twax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlabel("Mass $m^0$, for fixed $m^1=30$", fontsize=fs_label)
    ax.set_xlim(left=0.5, right=masses.max() + 0.5)
    ax.set_xticks(masses[4::5])
    ax.tick_params(axis="both", labelsize=fs_ticks)
axes[1].legend(fontsize=fs_legend + 3,
               title_fontsize=fs_legend_title)
t = axes[0].yaxis.get_offset_text()
t.set_size(fs_label)
plt.savefig(EXP_NAME + ".pdf")
