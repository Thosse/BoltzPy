import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

masses = np.array([2, 3])
model = bp.CollisionModel(masses, [[5, 5], [5, 5]])

grp = model.group(model.key_species(model.collision_relations),
                  model.collision_relations)
model.collision_relations = grp[(0, 0, 1, 1)]

grp = model.group(model.collision_relations[:, :2],
                  model.collision_relations)
# for col in grp.values():
#     print(col)
#     model.plot_collisions(col)

cols = grp[(2, 10)]
colvels = model.i_vels[cols]
(v0, v1) = colvels[0, :2]
print("v0, v1 = ", v0, v1)
# model.plot_collisions(cols)

dv = v1 - v0
dw = -(dv * masses[0]) // masses[1]
M = v0 + dv // 2
w_p = M - dw // 2
w_p_2 = M + dw // 2
normal = dv[::-1] * [1, -1]
print(dv, normal)


fig, ax = plt.subplots(1, 1, constrained_layout=True,
                       figsize=(3, 3))
# plot collisions
model.plot_collisions(cols, plot_object=ax)
# denote v_i and v_j
offset = np.array([0.8, -1.1])
ax.text(*v0 - offset, "$\mathfrak{v}_i$", fontsize=25)
offset = np.array([-0.9, 0.5])
ax.text(*v1 - offset, "$\mathfrak{v}_j$", fontsize=25)

# plot right hyperplane
h1 = w_p + np.array([2*normal, -2*normal])
ax.plot(h1[:, 0], h1[:, 1],
        linestyle="dashed",
        linewidth=2,
        c="black",
        zorder=-11)
ax.text(4.75, 1.75, "$H^{s,r}_{\mathfrak{v}_i, \mathfrak{v}_j}$",
        # rotation=45,
        fontsize=28,
        c="black")

# # plot left hyperplane
# h2 = w_p_2 + np.array([2*normal, -2*normal])
# ax.plot(h2[:, 0], h2[:, 1],
#         linestyle="dashed",
#         linewidth=2,
#         c="gray",
#         zorder=-11)

# plot distance to middle
ax.annotate(text="", xy=v0, xytext=M,
            arrowprops=dict(arrowstyle='<->',
                            linewidth=1.5,
                            color="black"))
# offset = np.array([1.25, 1.25])
# ax.text(*M - offset, "M", fontsize=18)
# ax.scatter(*M, c="black", marker="*", s=150,
#            zorder=11)

# plot distance to projected w
ax.annotate(text="", xy=M, xytext=w_p,
            arrowprops=dict(arrowstyle='<->',
                            linewidth=1.5,
                            color="black"))
offset = np.array([3.0, 0.5])
ax.text(*w_p - offset, "$p$", fontsize=25)
ax.scatter(*w_p, c="black", marker="+", s=250,
           zorder=11)



lim = model.max_i_vels[0, 0] * 1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig(bp.SIMULATION_DIR + "/plt_hyperplane.pdf")

