import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks, fs_legend_title

MAX_MASS = 30
# MASSES = np.array([[1, 30],
#                    [5, 30],
#                    [10, 30],
#                    [13, 30],
#                    [15, 30],
#                    [29, 30],
#                    [30, 30]
#                    ])
MASSES = np.array([[i, MAX_MASS] for i in range(1, 2*MAX_MASS+1)])
N = MASSES.shape[0]
SHAPES = [MAX_MASS + 1, 2*MAX_MASS + 1]
result = {shape: {dim: {key: np.full(MASSES.shape[0], -1, dtype=int)
                        for key in ["dist", "sdist", "nasd"]}
                  for dim in [2, 3]}
          for shape in SHAPES}

for shape in SHAPES:
    for dim in [2, 3]:
        for m, masses in enumerate(MASSES):
            print("\rMasses = ", masses, end="")
            model = bp.CollisionModel(masses,
                                      np.full((2, dim), shape, dtype=int),
                                      0.25,
                                      collision_relations=[],
                                      collision_weights=[],
                                      setup_collision_matrix=False)
            grid = [bp.Grid(model.shapes[s] % 2 + 4,
                            0.25,
                            model.spacings[s],
                            True)
                    for s in model.species]
            vels = [model.i_vels[model.idx_range(s)]
                    for s in model.species]
            dist = grid[1].key_distance(vels[0])
            result[shape][dim]["dist"][m] = model.filter(dist).shape[0]
            del dist

            sdist = grid[1].key_sorted_distance(vels[0])[..., :-1]
            result[shape][dim]["sdist"][m] = model.filter(sdist).shape[0]

            sort_vel = np.sort(np.abs(vels[0]), axis=-1)
            result[shape][dim]["nasd"][m] = model.filter(sort_vel).shape[0]
            del sdist, sort_vel

for shape in SHAPES:
    for dim in [2,3]:
        print(result[shape][dim]["nasd"])
COLORS = ["tab:purple", "tab:green", "tab:blue"]
STYLE = ["-o", "-x", "-"]
WIDTH = [0.5, 0.5, 3]
# setup plot
fig, ax = plt.subplots(1, 3,
                       # constrained_layout=True,
                       figsize=(12.75, 6.25),
                       sharex="all", sharey="all")

labels = {"sdist": r"\textsc{sadi}",
          "dist":  r"\textsc{dist}",
          "nasd":  r"\textsc{sabv}"
          }
shape = SHAPES[0]
for d, dim in enumerate([2, 3]):
    max_val = 1
    print(np.max(result[shape][dim]["dist"] / result[shape][dim]["sdist"]))
    for k, key in enumerate(["sdist", "dist", "nasd"]):
        res = result[shape][dim][key]
        ax[d].plot(np.arange(N) + 1, res, STYLE[k], color=COLORS[k],
                   linewidth=WIDTH[k], label=labels[key])
        # max_val = max(max_val, np.partition(res, -5)[-5])
        max_val = max(max_val, np.max(res))
    ax[d].set_yscale("log")
    ax[d].set_ylim(1, max_val ** 1.05)
    # ax[d].set_ylim(0, max_val * 1.05)
    ax[d].set_xlim(1, MASSES[:, 0].max())
    ax[d].set_axisbelow(True)
    ax[d].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                     linewidth=0.15)
    # ax[d].set_ylabel("Number of Equivalence Classes")
    ax[d].set_xlabel(r"Mass $m^1$",
                     fontsize=fs_label)
    ax[d].set_title("Grids  Shape {}".format(tuple([shape] * dim)),
                    fontsize=fs_title)

# plot another ax with larger shaped
max_val = 1
shape = SHAPES[1]
dim = 3
d = 2
print(np.max(result[shape][2]["dist"] / result[shape][2]["sdist"]))
print(np.max(result[shape][dim]["dist"] / result[shape][dim]["sdist"]))
for k, key in enumerate(["sdist", "dist", "nasd"]):
    res = result[shape][dim][key]
    ax[d].plot(np.arange(N) + 1, res, STYLE[k], color=COLORS[k],
               linewidth=WIDTH[k], label=key)
    # max_val = max(max_val, np.partition(res, -5)[-5])
    max_val = max(max_val, np.max(res))
ax[d].set_yscale("log")
ax[d].set_ylim(0.8, max_val ** 1.05)
# ax[d].set_ylim(0, max_val * 1.05)
ax[d].set_xlim(0, MASSES[:, 0].max() + 1)
ax[d].set_axisbelow(True)
ax[d].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                 linewidth=0.15)
ax[d].set_xlabel("Mass  $m^1$",
                 fontsize=fs_label)
ax[d].set_title("Grid Shape {}".format(tuple([shape] * dim)),
                fontsize=fs_title)

fig.suptitle("Partition Sizes for Masses $m=(m^1, 30)$ and Different Grid Shapes",
             fontsize=fs_suptitle)
ax[0].set_ylabel(r"Partition Size of $\mathfrak{V}^1$",
                 fontsize=fs_label)
ax[0].legend(loc="upper center", title="Key Functions",
             fontsize=fs_legend,
             title_fontsize=fs_legend_title)
plt.subplots_adjust(top=0.85)
for i in [0,1,2]:
    ax[i].tick_params(axis="both", labelsize=fs_ticks)
plt.savefig(bp.SIMULATION_DIR + "/plt_partition_size.pdf",
            bbox_inches='tight',)
plt.show()
