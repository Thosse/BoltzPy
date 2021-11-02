import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, faktor}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                    figsize=(12.75, 5.),
                                    sharey="all", sharex="all")

model_1 = bp.CollisionModel([2],
                            [(4, 4)],
                            1.)

model_2 = bp.CollisionModel([2, 1],
                            [(4, 4), (3, 3)],
                            1.)

# Left Plot
ax1 = model_1.plot_collisions(plot_object=ax1)
base_mono = -np.array([[-1, 1], [1, 3], [3, 1], [1, -1], [-1, 1]])
cols_mone = [base_mono + c for c in [[0, 2], [2, 0], [2, 2]]]
for c in cols_mone:
    ax1.plot(c[:, 0], c[:, 1], color="darkgray")
ax1.plot(base_mono[:, 0], base_mono[:, 1], color="black")
# ax1.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])


# Middle Plot
ax2 = model_2.plot_collisions(plot_object=ax2)
base_mix = np.array([[-4,0], [0, 0], [-1, -1], [-3, -1], [-4, 0]])
cols_mix = [base_mix + c for c in [[0, 4], [4, 0], [4, 4]]]
cols_bad = [base_mix + c for c in [[0,  2], [2,  2], [4,  2], [2, 4],
                                   [0, -2], [2, -2], [4, -2], [2, 0]]]
for c in cols_bad:
    ax2.plot(c[:, 0], c[:, 1],
             color="red",
             linewidth=2,
             linestyle="dotted",
             # dashes=(20, 15)
             )
for c in cols_mix:
    ax2.plot(c[:, 0], c[:, 1],
             color="darkgray",)
ax2.plot(base_mix[:, 0], base_mix[:, 1], color="black")


# Right Plot
offset = np.full(2, .1)
key_distance = model_2.subgrids(1).key_distance(model_2.subgrids(0).iG)
partitions = model_2.group(key_distance,
                           model_2.subgrids(0).iG,
                           as_dict=False)

for point, dist in zip(model_2.subgrids(0).iG, key_distance):
    ax3.annotate(s='', xy=point - 0.15*dist, xytext=point - 0.85*dist,
                 arrowprops=dict(arrowstyle='<->', linewidth=2, color="gray"))

for point in model_2.subgrids(1).iG:
    point = point - offset
    ax3.text(point[0], point[1], "1",
             bbox={"boxstyle": "circle", "color": "orange"},
             fontsize=13)

for p, prt in enumerate(partitions):
    for point in prt:
        point = point - offset
        ax3.text(point[0], point[1], str(p + 1),
                 bbox={"boxstyle": "circle", "color": "lightsteelblue"},
                 fontsize=13)
#
#     ax3.scatter(*p.transpose(), marker='o', alpha=0.5, s=50, c=colors[i])
ax3.set_aspect('equal')
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
ax1.set_title(r"Shifted Collisions in $\mathfrak{C}^{s,s}$",
              fontsize=fs_title)
ax2.set_title(r"Shifting Collisions in a Mixture",
              fontsize=fs_title)
ax3.set_title(# r"Equivalence Classes of "
              r"$\faktor{\mathfrak{V}^1}{dist_2}$"
              r" and $\faktor{\mathfrak{V}^2}{dist_1}$",
              fontsize=fs_title)

plt.subplots_adjust(top=0.7)
plt.tight_layout()
plt.savefig(bp.SIMULATION_DIR + "/plt_translation_of_collisions.pdf")
# plt.show()
