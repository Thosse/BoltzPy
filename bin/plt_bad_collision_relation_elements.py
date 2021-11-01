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

file_name = boltzpy.SIMULATION_DIR + "/plot_bad_collision_relation_elements.pdf"

ndim = 2
masses = np.array([3, 2], dtype=int)
spacings = np.array([4, 6], dtype=int)


model = bp.CollisionModel(masses,
                          ((3, 3), (3, 3)),
                          1,
                          spacings,
                          np.full((len(masses), len(masses)), 1),)

fig, ax = plt.subplots(1, 2, constrained_layout=True,
                       sharex="all",
                       sharey="all",
                       figsize=(12.75, 6.25))
colls = [[[1, 7, 8, 2], [0, 6, 15, 9]],
         [[1, 7, 2, 8], [0, 9, 6, 15]]
         ]
for i in [0, 1]:
    model.plot_collisions(colls[i], plot_object=ax[i], lw=1)
    # plot momentum vectors
    v = np.array([model.vels[colls[i][0][k]] for k in [0, 1, 2, 3]])
    ax[i].arrow(*v[0], *((v[1] - v[0]) * 0.925),
                head_width=0.25, lw=3, zorder=0, color="black")
    pos = (v[0] + v[1]) / 2 - np.array([1.2, 0.6])
    ax[i].text(*pos,
               r"$\mathfrak{v}_{\beta_2} - \mathfrak{v}_{\beta_1}$",
               fontsize=fs_label)
    ax[i].arrow(*v[2], *((v[3] - v[2]) * 0.925),
                head_width=0.25, lw=3, zorder=0, color="black")
    pos = (v[3] + v[2]) / 2 - np.array([1.2, -0.4])
    ax[i].text(*pos,
               r"$\mathfrak{v}_{\beta_4} - \mathfrak{v}_{\beta_3}$",
               fontsize=fs_label)
    # plot species comparison
    v = np.array([model.vels[colls[i][1][k]] for k in [0, 1, 2, 3]])
    if i == 0:
        pos = (v[0] + v[1]) / 2 - np.array([1.2, -0.4])
        ax[i].text(*pos,
                   r"$\mathfrak{s}_{\beta_4} = \mathfrak{s}_{\beta_3}$",
                   fontsize=fs_label)
        pos = (v[3] + v[2]) / 2 - np.array([1.2, +0.6])
        ax[i].text(*pos,
                   r"$\mathfrak{s}_{\beta_1} = \mathfrak{s}_{\beta_2}$",
                   fontsize=fs_label)
    elif i == 1:
        pos = (v[0] + v[1]) / 2 - np.array([1.1 + 0.45, 1.1 - 0.45])
        ax[i].text(*pos,
                   r"$\mathfrak{s}_{\beta_1} = \mathfrak{s}_{\beta_2}$",
                   rotation=45,
                   fontsize=fs_label)
        pos = v[3] + np.array([-1.8, 0.4])
        ax[i].text(*pos,
                   r"$\mathfrak{s}_{\beta_4} = \mathfrak{s}_{\beta_3}$",
                   rotation=-45,
                   fontsize=fs_label)

ax[0].set_title(r"Collision Trapezoids of $\beta \in [\alpha]_\sim$",
                fontsize=fs_title)
ax[1].set_title(r"Deformed Collision Trapezoids of $\beta \notin [\alpha]_\sim$",
                fontsize=fs_title)
for i in [0, 1]:
    # set tick values on axes, None = auto choice of matplotlib
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    # keep equal aspect ratio of the axes
    ax[i].set_aspect("equal")
    max = model.max_vel * 1.2
    ax[i].set_xlim(-max, max )
    ax[i].set_ylim(-max, max)

plt.savefig(file_name)
