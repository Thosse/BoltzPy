import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, faktor}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

'''
NOTE: These plots often require the right display resolution 
to look nice. I used 2560x1440, 16:9 displays.
If you want to reconstruct them you will have to tweak a bit around
'''

# setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True,  sharey="all", sharex="all")

# setup model
model = bp.CollisionModel([3,  13],
                          [[4, 4], [6, 6]],
                          collision_relations=[],
                          collision_weights=[],
                          setup_collision_matrix=False)
# contruct groups
key_dist = model.subgrids(1).key_distance(model.subgrids(0).iG)
norm = np.sum(model.subgrids(0).iG**2, axis=-1)
grp = model.group(key_dist, model.subgrids(0).iG, sort_key=norm)

# setup extended grids as a model, for easier plotting
shapes = [G.shape for G in model._get_extended_grids((0, 1), grp)]
ext = bp.CollisionModel(model.masses,
                        shapes,
                        collision_relations=[],
                        collision_weights=[],
                        setup_collision_matrix=False)

# stretch plot of model, to properly show the classes
stretch = 2.25
offset = 2.4      # offset text, needs tweaking by hand to look nice

# plot a large collision, to show necessity of extended grids
cols = np.array([[-39, -39], [39, -39], [9, 15], [-9, 15], [-39, -39]])
ax1.plot(stretch * cols[:, 0], stretch * cols[:, 1], c="black", linewidth=2)

# plot second species as X's
grid = model.subgrids(1)
points = stretch * grid.iG
ax1.scatter(points[:, 0], points[:, 1],
            **{"marker": 'x', "alpha": 0.9, "s": 300, "c": "tab:orange"})

# compute equivalence classes
key_distance = model.subgrids(1).key_distance(model.subgrids(0).iG)
partitions = model.group(key_distance,
                         model.subgrids(0).iG,
                         as_dict=False)

# plot first species as equivalence classes
for p, prt in enumerate(partitions):
    for point in prt:
        point = stretch*point - offset
        ax1.text(point[0], point[1], str(p + 1),
                 size=30,
                 bbox={"boxstyle": "circle", "color": "lightsteelblue"})

# plot x and y axis
ax1.annotate(s='', xy=(-117, 0), xytext=(117, 0),
             arrowprops=dict(arrowstyle='<->', linewidth=0.5, color="gray"))
ax1.annotate(s='', xy=(0, -117), xytext=(0, 117),
             arrowprops=dict(arrowstyle='<->', linewidth=0.5, color="gray"))


# plot line segments to show the distances
ax1.annotate(s='', xy=(0, 0), xytext=(-39*stretch, 0),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:blue"))
ax1.annotate(s='', xy=(0, 0), xytext=(15*stretch, 0),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:orange"))
ax1.annotate(s='', xy=(0, 0), xytext=(0, -39*stretch),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:blue"))
ax1.annotate(s='', xy=(0, 0), xytext=(0, 15*stretch),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:orange"))


# plot extended grid
ext.plot_collisions(plot_object=ax2)

# draw used grid parts as frames
ax2.plot([117, -39, -39, 117, 117], [117, 117, -39, -39, 117], c="Tab:blue")
ax2.plot([93, -15, -15, 93, 93], [93, 93, -15, -15, 93], c="Tab:orange")

# plot shifted collision
repr = np.array([39, 39])
dvels = cols - cols[None, 0] + repr[None, :]
ax2.plot(dvels[:, 0], dvels[:, 1], c="black", linewidth=2)

# plot x and y axis
ax2.annotate(s='', xy=(-117, 0), xytext=(117, 0),
             arrowprops=dict(arrowstyle='<->', linewidth=0.5, color="gray"))
ax2.annotate(s='', xy=(0, -117), xytext=(0, 117),
             arrowprops=dict(arrowstyle='<->', linewidth=0.5, color="gray"))


# plot distances as line segments
# ax2.annotate(s='', xy=(39,39), xytext=(0,0),
#              arrowprops=dict(arrowstyle='<->', linewidth=4))
ax2.annotate(s='', xy=(78, 39), xytext=(39, 39),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:blue"))
ax2.annotate(s='', xy=(93, 39), xytext=(78, 39),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:orange"))
ax2.annotate(s='', xy=(39, 78), xytext=(39, 39),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:blue"))
ax2.annotate(s='', xy=(39, 117), xytext=(39, 78),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:blue"))

ax2.annotate(s='', xy=(0, 39), xytext=(39, 39),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:blue"))
ax2.annotate(s='', xy=(-15, 39), xytext=(0, 39),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:orange"))
ax2.annotate(s='', xy=(39, 0), xytext=(39, 39),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:blue"))
ax2.annotate(s='', xy=(39, -39), xytext=(39, 0),
             arrowprops=dict(arrowstyle='<->', linewidth=4, color="tab:blue"))
ax2.scatter(repr[0], repr[1],
            **{"marker": 'o', "s": 100, "c": "black"})
ax1.set_aspect('equal')
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title(r"Velocity Grids $\mathfrak{V}^s$ and $\mathfrak{V}^r$",
              fontsize=65)
ax2.set_title(r"Extended Grids $\mathfrak{V}^s_{ext}$ and $\mathfrak{V}^r_{ext}$",
              fontsize=65)
plt.show()
