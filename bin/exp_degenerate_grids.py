
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as mpl_ani
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks, fs_legend_title
import boltzpy as bp

def plot(masses, vels0, vels1, ax=None):
    """plot 2 given velocity arrays"""
    dim = vels0.shape[1]
    # Just in case, I wnad to male a plot of multiple axes
    if ax is None:
        projection = "3d" if dim == 3 else None
        ax = plt.figure().add_subplot(projection=projection)
    # print number of independent collision invariants to terminal
    # These must be added manually to the tex file
    print("Linear independent collision invariants = ",
          n_collision_invariants(masses, vels0, vels1))
    # plot the velocities
    ax.scatter(*(vels0.transpose()),
               **{"marker": 'o', "alpha": 0.5, "s": 100},
               label=r"$m^0=" + str(masses[0]) + r"$"
    )
    ax.scatter(*(vels1.transpose()), **{"marker": 'x', "alpha": 0.9, "s": 150},
               label=r"$m^1=" + str(masses[1]) + r"$"
    )
    # add a legend that states the masses
    ax.legend(title="Masses:",
              fontsize=fs_legend,
              title_fontsize=fs_legend_title)
    # classic view of coordinate axes
    if dim == 2:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color(None)
        ax.spines['top'].set_color(None)
    elif dim == 3:
        # spines dont work in 3d
        max_val = np.max([np.max(vels0), np.max(vels1)])
        arrows = np.eye(3) * max_val
        ax.quiver(*(-arrows), *(2*arrows), color='k', linewidth=.7, arrow_length_ratio=0.08)
    # no axis values
    ax.tick_params(labelbottom=False, labelleft=False)
    # use equal aspect ratio for x and y axis
    if dim == 2:
        plt.gca().set_aspect('equal', adjustable='box')
    elif dim == 3:
        plt.gca().set_aspect('auto', adjustable='box')
    return


def n_collision_invariants(masses, vels0, vels1=None):
    # construct a matrix with a collision invariant in each line
    matrix = np.zeros((3 + vels0.shape[1], vels0.shape[0] + vels1.shape[0]),
                      dtype=int)
    # number density 0
    matrix[0, 0: vels0.shape[0]] = 1
    # number density 1
    matrix[1, vels0.shape[0]:] = 1
    # total energy
    matrix[2, 0: vels0.shape[0]] = masses[0] * np.sum(vels0**2, axis=1)
    matrix[2, vels0.shape[0]:] = masses[1] * np.sum(vels1**2, axis=1)
    # each components of total momentum is a separate line
    for i in range(vels0.shape[1]):
        matrix[3+i, 0: vels0.shape[0]] = masses[0] * vels0[:, i]
        matrix[3+i, vels0.shape[0]:] = masses[1] * vels1[:, i]
    invariants = np.linalg.matrix_rank(matrix)
    return invariants


# mass ratio for all models
masses = [2, 3]

# # 2D-Broadwell Mixture
# # unitless velocities, are mutliplied with a factor for each specimen
# base_vels = np.array([[ 1,  0],
#                       [-1,  0],
#                       [ 0,  1],
#                       [ 0, -1]],
#                      dtype=int)
# # velocities of the concatenated grid
# vels0 = base_vels * masses[1]
# vels1 = base_vels * masses[0]
# plot(masses, vels0, vels1)

fig = plt.figure(figsize=(12.75, 6.25),
                 constrained_layout=True)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# 3D-Broadwell Mixture
# unitless velocities, are mutliplied with a factor for each specimen
base_vels = np.array([[ 1,  0,  0],
                      [-1,  0,  0],
                      [ 0,  1,  0],
                      [ 0, -1,  0],
                      [ 0,  0,  1],
                      [ 0,  0, -1]],
                     dtype=int)
# velocities of the concatenated grid
vels0 = base_vels * masses[1]
vels1 = base_vels * masses[0]
plot(masses, vels0, vels1, ax=ax1)

# 3D-(2,2,2) model
# unitless velocities, are mutliplied with a factor for each specimen
base_vels = np.array([[ 1,  1,  1],
                      [ 1,  1, -1],
                      [ 1, -1,  1],
                      [ 1, -1, -1],
                      [-1,  1,  1],
                      [-1,  1, -1],
                      [-1, -1,  1],
                      [-1, -1, -1]],
                     dtype=int)
# velocities of the concatenated grid
vels0 = base_vels * masses[1]
vels1 = base_vels * masses[0]
plot(masses, vels0, vels1, ax=ax2)

for ax in [ax1, ax2]:
    ax.view_init(elev=15.,
                 azim=15 # 22.5
                 )
    plt.tight_layout()
plt.savefig(bp.SIMULATION_DIR + "/degenerated_models.pdf",
            bbox_inches='tight',
            )
