import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']
import matplotlib.pyplot as plt

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


fig, ax = plt.subplots(1, 1)
# plot collisions
model.plot_collisions(cols, plot_object=ax)
# denote v_i and v_j
offset = np.array([0.8, -0.7])
ax.text(*v0 - offset, "$\mathfrak{v}_i$", fontsize=20)
offset = np.array([-0.8, 0.5])
ax.text(*v1 - offset, "$\mathfrak{v}_j$", fontsize=20)

# plot right hyperplane
h1 = w_p + np.array([2*normal, -2*normal])
ax.plot(h1[:, 0], h1[:, 1],
        linestyle="dashed",
        linewidth=2,
        c="tab:red",
        zorder=-11)
ax.text(9, 4.5, "$H^{s,r}_{\mathfrak{v}_i, \mathfrak{v}_j}$",
        fontsize=18,
        c="tab:red")

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
offset = np.array([1.5, 0.5])
ax.text(*w_p - offset, "$p$", fontsize=20)
ax.scatter(*w_p, c="black", marker="+", s=150,
           zorder=11)



lim = model.max_i_vels[0, 0] * 1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

