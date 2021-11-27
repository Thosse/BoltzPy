import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, faktor}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(12.75, 5.8),
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
ax1.plot(base_mono[:, 0], base_mono[:, 1], color="black",)
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
             color="darkgray")
ax2.plot(base_mix[:, 0], base_mix[:, 1], color="black")
ax2.plot([], [],
         color="black",
         label="Original Collision")
ax2.plot([], [],
         color="darkgray",
         label="Shifted Collisions")
ax2.plot([],[],
         color="red",
         linewidth=2,
         linestyle="dotted",
         label="Failed, Off-Grid Shifts"
         )
lg = fig.legend(loc="lower center",
           bbox_to_anchor=(0.50, -0.025),
           ncol=3,
           fontsize=fs_legend + 2)

for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])
ax1.set_title(r"Shifting Intraspecies Collisions",
              fontsize=fs_title)
ax2.set_title(r"Shifting Interspecies Collisions",
              fontsize=fs_title)

# plt.subplots_adjust(bottom=0.0)
# plt.tight_layout()
plt.savefig(bp.SIMULATION_DIR + "/plt_translation_of_collisions.pdf",
                bbox_extra_artists=(lg,),
                bbox_inches='tight')
# plt.show()
plt.clf()

#################################################################################
#               Equivalence Classes plot                                        #
#################################################################################
fig = plt.Figure(figsize=(6, 6.5))
ax = fig.add_subplot()
# Right Plot
offset = np.full(2, .1)
key_distance = model_2.subgrids(1).key_distance(model_2.subgrids(0).iG)
partitions = model_2.group(key_distance,
                           model_2.subgrids(0).iG,
                           as_dict=False)

for point, dist in zip(model_2.subgrids(0).iG, key_distance):
    ax.annotate(s='', xy=point - 0.15*dist, xytext=point - 0.85*dist,
                 arrowprops=dict(arrowstyle='<->', linewidth=2, color="gray"))

for point in model_2.subgrids(1).iG:
    point = point - offset
    ax.text(point[0], point[1], "1",
             bbox={"boxstyle": "circle", "color": "orange"},
             fontsize=13)

for p, prt in enumerate(partitions):
    for point in prt:
        point = point - offset
        ax.text(point[0], point[1], str(p + 1),
                 bbox={"boxstyle": "circle", "color": "lightsteelblue"},
                 fontsize=13)
#
#     ax3.scatter(*p.transpose(), marker='o', alpha=0.5, s=50, c=colors[i])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title(r"Equivalence Classes $\faktor{\mathfrak{V}^s}{dist^r}$",
          # r"$\faktor{\mathfrak{V}^1}{dist_2}$"
          # r" and $\faktor{\mathfrak{V}^2}{dist_1}$",
          fontsize=fs_title,
          pad=15)
ax.set_aspect('equal')
ax.scatter([], [],
            color="lightsteelblue",
            s=300,
            label=r"$\left\lvert \faktor{\mathfrak{V}^1}{dist^2} \right\rvert = 4$")
ax.scatter([], [],
            color="orange",
            s=300,
            label=r"$\left\lvert \faktor{\mathfrak{V}^2}{dist^1} \right\rvert = 1$")
fig.legend(loc="lower center",
           ncol=2,
           fontsize=fs_legend)
plt.subplots_adjust(bottom=0.0)
plt.tight_layout()
fig.savefig(bp.SIMULATION_DIR + "/plt_velocity_equivalence_classes.pdf")