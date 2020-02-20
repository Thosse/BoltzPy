import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import boltzpy.initialization as bp_i


def plot_discrete_distribution(discrete_distribution,
                               velocities,
                               physcial_spacing,
                               plot_object=None,
                               **plot_style):
    """Plot the discrete distribution of a single specimen using matplotlib 3D.

    Parameters
    ----------
    plot_object : TODO Figure? matplotlib.pyplot?
    """
    if "color" not in plot_style.keys():
        plot_style["color"] = "green"
    if "alhpa" not in plot_style.keys():
        plot_style["alpha"] = 0.4

    # show plot directly, if no object to store in is specified
    show_plot_directly = plot_object is None

    # Construct default plot object if None was given
    if plot_object is None:
        # Choose standard pyplot
        plot_object = plt

    # Set plot_object to be a 3d plot
    ax = plot_object.gca(projection="3d")

    # plot discrete distribution as a transparent 3D bar plot
    #   subtract physical_spacing/2 to place the bars centered on the velocities
    X = velocities[..., 0] - physcial_spacing / 2
    Y = velocities[..., 1] - physcial_spacing / 2
    Z = discrete_distribution

    # Create 3d bar plot
    ax.bar3d(X, Y, np.zeros(X.size),
             physcial_spacing, physcial_spacing, Z,
             **plot_style)

    if show_plot_directly:
        plot_object.show()
    return plot_object


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
    momenta = np.array([particle_number, *mean_velocity, temperature])
    Z = bp_i.maxwellian(velocities, mass, momenta).reshape((num, num))

    # Create Surface plot
    ax.plot_surface(X, Y, Z, **plot_style)

    if show_plot_directly:
        plot_object.show()
    return plot_object
