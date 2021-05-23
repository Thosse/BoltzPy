import boltzpy as bp
import numpy as np
import matplotlib.pyplot as plt

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
SHAPES = {dim: np.array([[MAX_MASS+1] * dim] * 2, dtype=int)
          for dim in [2, 3]}
result = {dim: {key: np.full(MASSES.shape[0], -1, dtype=int)
                for key in ["dist", "sdist", "nasd"]}
          for dim in [2, 3]}


for dim in [2, 3]:
    for m, masses in enumerate(MASSES):
        # print("Masses = ", masses)
        model = bp.CollisionModel(masses,
                                  SHAPES[dim],
                                  0.25,
                                  collision_relations=[],
                                  collision_weights=[],
                                  setup_collision_matrix=False)
        grid = [bp.Grid(SHAPES[dim][s] % 2 + 4,
                        0.25,
                        model.spacings[s],
                        True)
                for s in model.species]
        vels = [model.i_vels[model.idx_range(s)]
                for s in model.species]
        dist = grid[1].key_distance(vels[0])
        result[dim]["dist"][m] = model.filter(dist).shape[0]
        del dist

        sdist = grid[1].key_sorted_distance(vels[0])[..., :-1]
        result[dim]["sdist"][m] = model.filter(sdist).shape[0]

        sort_vel = np.sort(np.abs(vels[0]), axis=-1)
        nasd = np.concatenate((sort_vel, sdist), axis=-1)
        result[dim]["nasd"][m] = model.filter(nasd).shape[0]


COLORS = ["tab:purple", "tab:green", "tab:blue"]
STYLE = ["-o", "-x", "-"]
WIDTH = [0.7, 0.7, 1]
# setup plot
fig, ax = plt.subplots(1, 2, constrained_layout=True, sharex="all")
for d, dim in enumerate([2, 3]):
    max_val = 1
    print(result[dim]["dist"] / result[dim]["sdist"])
    for k, key in enumerate(["sdist", "dist", "nasd"]):
        res = result[dim][key]
        ax[d].plot(np.arange(N) + 1, res, STYLE[k], color=COLORS[k],
                   linewidth=WIDTH[k], label=key)
        # max_val = max(max_val, np.partition(res, -5)[-5])
        max_val = max(max_val, np.max(res))
    ax[d].set_yscale("log")
    ax[d].set_ylim(1, max_val ** 1.05)
    # ax[d].set_ylim(0, max_val * 1.05)
    ax[d].set_xlim(1, MASSES[:, 0].max())
    ax[d].set_axisbelow(True)
    ax[d].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                     linewidth=0.2)
    ax[d].set_ylabel("Number of Equivalence Classes")
    ax[d].set_xlabel("Mass of Specimen_1")
    ax[d].set_title("Partition Sizes in {}D Grids of Shape {}"
                    "".format(dim, tuple(SHAPES[d+2][0])))
ax[0].legend(loc="upper left", title="key function")
plt.show()
