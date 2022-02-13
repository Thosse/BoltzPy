
# Desired Command / Road Map
import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

MASSES = [2, 3]
SHAPES = [[5, 5],
          [7, 7]]

m = bp.CollisionModel(MASSES, SHAPES)

# get keys: energy transfer and species
key_spc = m.key_species(m.collision_relations)[:, 1:3]
key_E = m.key_energy_transfer(m.collision_relations)

grp = m.group([key_spc, key_E], m.collision_relations)

ET = grp[(0, 1, 1)]
NET = grp[(0, 1, 0)]

# angles = m.key_angle(NET)
# print(np.unique(angles, axis=0))
# m.plot_collisions(m.filter(m.key_angle(NET), NET))

fig, axes = plt.subplots(nrows=1, ncols=2,
                         figsize=(10.75, 6.0),
                         constrained_layout=True)
# fix limits for each plot
for ax in axes:
    ax.set_xlim(xmin=-1.1 * m.max_vel, xmax=1.1 * m.max_vel, auto=False)
    ax.set_ylim(ymin=-1.1 * m.max_vel, ymax=1.1 * m.max_vel, auto=False)

# plot all collisions in background
axes[0].set_title("Energy Transferring Collisions",
                  fontsize=fs_title)
axes[1].set_title("Non-Energy Transferring Collisions",
                  fontsize=fs_title)
m.plot_collisions(ET, plot_object=axes[0], lw=0.3,
                  color=0.85 * np.ones(3), zorder=-5)
m.plot_collisions(NET, plot_object=axes[1], lw=0.3,
                  color=0.85 * np.ones(3), zorder=-5)

# plot highlighted collisions for each angle
plane_style = {"color": "black",
               "ls": "dotted",
               "lw": 2.5}
axes[0].plot([-12, 6, 4, -8, -12], [-6, 0, -4, -8, -6], lw=1, c="black",
             label=r"Highlighted Collisions $\alpha$")
axes[0].plot([-3-100, -3+100], [-3+300, -3-300], **plane_style,
             label=r"Hyperplanes $H(\alpha)$")

axes[0].plot([8, 6, 12, 12, 8], [0, 0, 6, 4, 0], lw=1, c="black")
axes[0].plot([0, 4, 12, 12, 0], [0, 0, 8, 12, 0], lw=1, c="black")
axes[0].plot([12-100, 12+100], [+100, -100], **plane_style)

axes[1].plot([-12, -12, -8, -8, -12], [-6, 6, 4, -4, -6], lw=1, c="black")
# axes[1].plot([-12, -12, -4, -4, -12], [-12, 12, 8, -8, -12], lw=1, c="black")
axes[1].plot([-100, 100], [0, 0], **plane_style)

axes[1].plot([6, 8, 12, 12, 6], [12, 12, 8, 6, 12], lw=1, c="black")
axes[1].plot([-100, +100], [-100, +100], **plane_style)

axes[1].plot([12, 8, -4, -6, 12], [-6, -4, -8, -12, -6], lw=1, c="black")
axes[1].plot([-100, +100], [+300, -300], **plane_style)

# set up legend
axes[0].plot([], [], lw=1, color=0.8 * np.ones(3), label="All NET/ET Collisions")
fig.legend(fontsize=fs_legend + 2,
           loc="lower center",
           ncol=3)
plt.subplots_adjust(left=0.05,
                    bottom=0.1,
                    right=0.95,
                    top=1,
                    wspace=0.2,
                    hspace=0)
plt.savefig(bp.SIMULATION_DIR + "/phd_plot_energy_transferring_collisions.pdf")
