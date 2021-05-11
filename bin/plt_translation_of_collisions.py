import boltzpy as bp
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, sharey="all", sharex="all")

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
             color="tomato",
             linewidth=0.7,
             linestyle="dashed")
for c in cols_mix:
    ax2.plot(c[:, 0], c[:, 1],
             color="darkgray",)
ax2.plot(base_mix[:, 0], base_mix[:, 1], color="black")


# Right Plot
model_2.subgrids(1).plot(plot_object=ax3,
                         **{"marker": 'x', "alpha": 0.9, "s": 100, "c": "orange"})
key_distance = model_2.subgrids(1).key_distance(model_2.subgrids(0).iG)
partitions = model_2.group(key_distance,
                           model_2.subgrids(0).iG,
                           as_dict=False)

for p, prt in enumerate(partitions):
    for point in prt:
        ax3.text(point[0], point[1], str(p + 1),
                 bbox={"boxstyle" : "circle", "color":"lightsteelblue"})
#
#     ax3.scatter(*p.transpose(), marker='o', alpha=0.5, s=50, c=colors[i])
ax3.set_aspect('equal')

plt.tight_layout()
plt.savefig(bp.SIMULATION_DIR + "/plt_translation_of_collisions.eps")
plt.show()
