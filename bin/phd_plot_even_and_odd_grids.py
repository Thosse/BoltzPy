# Desired Command / Road Map
import boltzpy
import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
matplotlib.rcParams['legend.title_fontsize'] = 14
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

file_name = boltzpy.SIMULATION_DIR + "/plot_even_and_odd_grids.pdf"

ndim = 2
masses = np.array([1], dtype=int)
spacing = np.array([2], dtype=int)


models = [bp.CollisionModel(masses,
                            ((4, 4),),
                            1,
                            spacing,
                            np.full((len(masses), len(masses)), 1),),
          bp.CollisionModel(masses,
                            ((5, 5),),
                            1,
                            spacing,
                            np.full((len(masses), len(masses)), 1), ),
          bp.CollisionModel(masses,
                            ((5, 4),),
                            1,
                            spacing,
                            np.full((len(masses), len(masses)), 1),)
          ]

fig, ax = plt.subplots(1, 3, constrained_layout=True,
                       sharex="all",
                       sharey="all",
                       figsize=(12.75, 4.8))
titles = [str(tuple(models[i].shapes[0])) + " Shaped Grid"
          for i in [0, 1, 2]]
for i in [0, 1, 2]:
    # Plot Grids as scatter plot
    grid_points = models[i].vels.transpose()
    ax[i].scatter(*grid_points, zorder=5)

    # set tick values on axes, None = auto choice of matplotlib
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    # keep equal aspect ratio of the axes
    ax[i].set_aspect("equal")
    ax[i].set_title(titles[i], fontsize=fs_title)
    # add coordinate axes
    # ax[i].spines["top"].set_color("none")
    # ax[i].spines["bottom"].set_position("zero")
    # ax[i].spines["left"].set_position("zero")
    # ax[i].spines["right"].set_color("none")
    max = models[i].max_vel + 0.5
    ax[i].arrow(-max, 0, 2*max, 0, head_width=0.15, zorder=0, color="black")
    ax[i].arrow(0,-max, 0, 2*max, head_width=0.25, zorder=0, color="black")
plt.savefig(file_name)
del fig, ax
plt.cla()


#################################################
#           More Complex Velocity Spaces        #
#################################################
file_name = boltzpy.SIMULATION_DIR + "/plot_complex_velocity_spaces.pdf"
models = [bp.BaseModel(masses,
                       ((9, 9),),
                       1,
                       spacing),
          bp.BaseModel(masses[[0, 0]],
                       ((5, 5), (5, 5)),
                       1,
                       [2, 4]),
          ]

fig, ax = plt.subplots(1, 2, constrained_layout=True,
                       sharex="all",
                       sharey="all",
                       figsize=(12.75, 4.8))
titles = ["Reduced Velocity Space", "Merged Velocity Spaces"]

# circular space
grid_points = models[0].vels.transpose()
ax[0].scatter(*grid_points, c="lightgray", s= 50)
norms = np.linalg.norm(grid_points, axis=0)
grid_points = grid_points[:, norms <= 8]
ax[0].scatter(*grid_points, s=125)

# merged space
colors = ["tab:blue", "tab:green"]
for s in [0,1]:
    grid_points = models[1].subgrids(s).pG.transpose()
    ax[1].scatter(*grid_points, c=colors[s], zorder=5, s=125)

for i in [0, 1]:
    # set tick values on axes, None = auto choice of matplotlib
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    # keep equal aspect ratio of the axes
    ax[i].set_aspect("equal")
    ax[i].set_title(titles[i], fontsize=fs_title)
    # add coordinate axes
    # ax[i].spines["top"].set_color("none")
    # ax[i].spines["bottom"].set_position("zero")
    # ax[i].spines["left"].set_position("zero")
    # ax[i].spines["right"].set_color("none")
    max = models[i].max_vel + 0.5
    ax[i].arrow(-max, 0, 2*max, 0, head_width=0.15, zorder=0, color="black")
    ax[i].arrow(0,-max, 0, 2*max, head_width=0.25, zorder=0, color="black")

plt.savefig(file_name)
