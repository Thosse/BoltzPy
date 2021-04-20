
import numpy as np
import matplotlib.pyplot as plt


def plot(masses, vels0, vels1, ax=None, save=None):
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
    line1 = ax.scatter(*(vels0.transpose()), **{"marker": 'o', "alpha": 0.5, "s": 100})
    line2 = ax.scatter(*(vels1.transpose()), **{"marker": 'x', "alpha": 0.9, "s": 150})
    # add a legend that states the masses
    ax.legend((masses[0], masses[1]), title="masses:")
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
    if save is None:
        plt.show()
    else:
        assert type(save) is str
        plt.savefig(save)
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

# 2D-Broadwell Mixture
# unitless velocities, are mutliplied with a factor for each specimen
base_vels = np.array([[ 1,  0],
                      [-1,  0],
                      [ 0,  1],
                      [ 0, -1]],
                     dtype=int)
# velocities of the concatenated grid
vels0 = base_vels * masses[1]
vels1 = base_vels * masses[0]
plot(masses, vels0, vels1)

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
plot(masses, vels0, vels1)

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
plot(masses, vels0, vels1)


