
# Desired Command / Road Map
import boltzpy as bp
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt

# for c in range(1000000):
#    res = []
#    for a in range(1, c):
#        b = int(np.sqrt(c**2 - a**2))
#        if a**2 + b**2 == c**2:
#            res.append([a, b])
#    res = np.array(res, ndmin=2)
#    res.sort(axis=1)
#    uniques = np.unique(res, axis=0)
#    if len(uniques) > 1:
#        print("c = ", c)
#        print(uniques)
#        input()

# plot trapezoid
fig, ax = plt.subplots(1, 1, constrained_layout=True,
                       figsize=(12.75, 6.25))
fs = 50
x1 = 3
x2 = 5
h = 2
vels = np.array([[-x1,0], [x1,0], [x2, h], [-x2,h], [-x1, 0]]) / np.array([[2, 1]])
ax.plot(vels[:,0], vels[:, 1], "-",  c="black",
        zorder=2)
ax.scatter(vels[:,0], vels[:, 1], s=150,  c="black",
           zorder=2)
# annotate velocities
rot90 = np.array([[0, -1], [1, 0]])
cur_rot = np.eye(2)
offset = np.array([[-0.2, -0.2],
                   [-0.15, -0.2],
                   [-0.2, 0.1],
                   [-0.2, 0.1]])
for i, v in enumerate(vels[:4]):
    pos = v + offset[i]
    ax.text(*pos, r"$\mathfrak{v}_{\alpha_" + str(i+1) + "}$",
            fontsize=fs,
            c="black",
            zorder=2)

# plot height
h = 0.5 * np.array([vels[0] + vels[1], vels[2] + vels[3]])
c = "tab:blue"
# plt.plot(*h, "-->", lw=3, c="red")
plt.arrow(*h[0], h[1, 0], h[1,1] - 0.175,
          lw=4,
          color=c,
          width=0.03,
          zorder=1)
ax.text(0.05, 1.0, r"$\vec{h}$",
        fontsize=fs,
        c=c)

# plot width
w = vels[:2]
c = "tab:green"
plt.arrow(*w[0], 2*w[1, 0] - 0.175, 2*w[1,1],
          lw=4,
          color=c,
          width=0.03)
ax.text(1.0, 0.1, r"$\Delta M$",
        fontsize=fs,
        c=c,
        zorder=1)

# remove axes
fig.patch.set_visible(False)
ax.axis("off")
# set equal aspect ratio
ax.set_aspect('equal')
plt.savefig(bp.SIMULATION_DIR + "/plt_key_angle.pdf")


# model = bp.CollisionModel([2, 3],
#                           [(5, 5), (7, 7)],
#                           1,
#                           [6, 4],
#                           np.array([[50, 50], [50, 50]]))
# grp = model.group(model.key_shape(model.collision_relations),
#                   model.collision_relations)
# # create plots
# for k,v in grp.items():
#     print("key = ", k)
#     fig, axes = plt.subplots(1, 1)
#     model.plot_collisions(v, plot_object=axes)
#     rvels = model.vels[v[0]]
#     axes.plot(2 * tuple(rvels[:, 0]), 2 * tuple(rvels[:, 1]), color="black", lw=2)
#     plt.show()
#     # if input() == "1":
#     #     for c in v:
#     #         model.plot_collisions(c)
