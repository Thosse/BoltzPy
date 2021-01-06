import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import boltzpy as bp


def plot_continuous_maxwellian(particle_number,
                               mean_velocity,
                               temperature,
                               mass,
                               minimum_velocity,
                               maximum_velocity,
                               num=100,
                               plot_object=None,
                               **plot_style):
    """Plot the discrete distribution of a single specimen using matplotlib 3D.

    Parameters
    ----------
    plot_object : TODO Figure? matplotlib.pyplot?
    """
    if "color" not in plot_style.keys():
        plot_style["color"] = "blue"
    if "alhpa" not in plot_style.keys():
        plot_style["alpha"] = 0.2
    # show plot directly, if no object to store in is specified
    show_plot_directly = plot_object is None

    # Construct default plot object if None was given
    if plot_object is None:
        # Choose standard pyplot
        plot_object = plt

    # Set plot_object to be a 3d plot
    ax = plot_object.gca(projection="3d")

    # set up X and Y axes
    steps = np.linspace(minimum_velocity, maximum_velocity, num)
    X, Y = np.meshgrid(steps, steps)
    # Compute continuous maxwellian as Z axis
    velocities = np.vstack((X.flatten(), Y.flatten())).T
    Z = bp.CollisionModel.maxwellian(velocities,
                                     mass,
                                     particle_number,
                                     mean_velocity,
                                     temperature).reshape((num, num))

    # Create Surface plot
    ax.plot_surface(X, Y, Z, **plot_style)

    if show_plot_directly:
        plot_object.show()
    return plot_object
