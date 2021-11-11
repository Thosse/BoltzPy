import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, faktor}'
import matplotlib.pyplot as plt

'''
NOTE: These plots often require the right display resolution 
to look nice. I used 2560x1440, 16:9 displays.
If you want to reconstruct them you will have to tweak a bit around
'''

# setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True,  sharey="all", sharex="all")

# setup model
model = bp.CollisionModel([3,  5],
                          [[5, 5], [9, 9]])

# # choose a representative
repr = np.array([-10, 0])
# choose some collisions, attached to repr
cols = np.array([[[-10, 0], [-20, 0], [-18, -6], [-12, -6], [-10, 0]],
                 [[-10, 0], [0, 20], [-6, 18], [-12, 6], [-10, 0]],
                 [[-10, 0], [0, -20], [-6, -18], [-12, -6], [-10, 0]]
                 ])
# choose a representative
# repr = np.array([-10, 10])
# # choose some collisions, attached to repr
# # cols = np.array([[[-10, 10], [-20, 10], [-18, 12], [-12, 12], [-10, 10]],
# #                  [[-10, 10], [-20, 10], [-18, 6], [-12, 6], [-10, 10]]])
# cols = np.array([[[-10, 10], [-20, 10], [-18, 12], [-12, 12], [-10, 10]],
#                  [[-10, 10], [-20, 10], [-18, 6], [-12, 6], [-10, 10]],
#                  [[-10, 10], [-10, 20], [-12, 18], [-12, 12], [-10, 10]]])

# plot second species as X's
grid = [model.subgrids(0), model.subgrids(1)]
ax1.scatter(grid[1].iG[:, 0], grid[1].iG[:, 1],
            **{"marker": 'x', "alpha": 0.9, "s": 300, "c": "tab:orange"})

# compute equivalence classes by distance
key_distance = grid[1].key_distance(grid[0].iG)
partitions = model.group(key_distance,
                         model.subgrids(0).iG,
                         as_dict=False)

# find group that contains repr
repr_grp = None
for p, prt in enumerate(partitions):
    for point in prt:
        if np.all(point == repr):
            repr_grp = prt
assert repr_grp is not None

# # plot shifted collision in grey (only one exists)
for p, p_vel in enumerate(repr_grp):
    new_cols = cols + p_vel[None, None] - cols[:, :1]
    for c in new_cols:
        if np.all(model.get_idx([0,0,1,1], c[0:4]) != -1):
            ax1.plot(c[:, 0], c[:, 1], c="darkgray", linewidth=2)

# plot collisions
for c in cols:
    ax1.plot(c[:, 0], c[:, 1], c="black", linewidth=4)

# plot first species as equivalence classes
offset = 0.45      # offset text, needs tweaking by hand to look nice
for p, prt in enumerate(partitions):
    for point in prt:
        point = point - offset
        ax1.text(point[0], point[1], str(p + 1),
                 size=25,
                 bbox={"boxstyle": "circle", "color": "lightsteelblue"})

#####################
# Create right plot #
#####################

# plot second species as X's
ax2.scatter(grid[1].iG[:, 0], grid[1].iG[:, 1],
            **{"marker": 'x', "alpha": 0.9, "s": 300, "c": "tab:orange"})

# compute equivalence classes by distance
key_pd = grid[1].key_sorted_distance(grid[0].iG)
partitions, rotations = model.group(key_pd[:, :-1],
                                    (grid[0].iG, key_pd[:, -1]),
                                    as_dict=False)

# plot first species as equivalence classes
for p, prt in enumerate(partitions):
    for point in prt:
        point = point - offset
        ax2.text(point[0], point[1], str(p + 1),
                 size=25,
                 bbox={"boxstyle": "circle", "color": "lightsteelblue"})

# find group that contains repr
repr_grp = None
repr_grp_idx = None
for p, prt in enumerate(partitions):
    for point in prt:
        if np.all(point == repr):
            repr_grp = prt
            repr_grp_idx = p
assert repr_grp is not None

# normalize repr_cols
sym_mat = model.symmetry_matrices
repr_rot = grid[1].key_sorted_distance([repr])[0, -1]
repr_cols = np.einsum("ji, nkj->nki",
                      sym_mat[repr_rot],
                      cols - repr[None])

# # plot shifted collision in grey (only one exists)

for p, p_vel in enumerate(repr_grp):
    new_cols = np.einsum("ij, nkj->nki",
                         sym_mat[rotations[repr_grp_idx][p]],
                         repr_cols)
    new_cols += p_vel
    for c in new_cols:
        if np.all(model.get_idx([0,0,1,1], c[0:4]) != -1):
            ax2.plot(c[:, 0], c[:, 1], c="darkgray", linewidth=2)
# plot collisions
for c in cols:
    ax2.plot(c[:, 0], c[:, 1], c="black", linewidth=4)



ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title(r"Collision Shifts in $\faktor{\mathfrak{C}^{s,r}}{dist^r}$",
              fontsize=45, pad=20)
ax2.set_title(r"Symmetric Shifts in $\faktor{\mathfrak{C}^{s,r}}{sadi^r}$",
              fontsize=45, pad=20)

plt.show()
