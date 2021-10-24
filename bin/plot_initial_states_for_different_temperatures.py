import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

"""
Create a plot for the discretization error and cutoff error 
for a given basemodel (concatenated grid).
We compute the (continuous) maxwellian on the grid 
for a fixed temperature and number density, 
and a fine range of mean velocities (x component)  v_x.
We plot the difference of the theoretical and discrete moments 
against v_x. 

The resulting error plot is a superposition of 
the discretization error and cutoff error.
"""

# number of discretization points
N = 201
# plot line styles
line_style = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

#  Used Velocity Model
mass = 1
# setup velocity models
model = bp.BaseModel([mass],
                     [(5, 5)],
                     1,
                     [2])

# initialization parameters for continuous maxwellian
number_density = 1.0
temperatures = np.array([1.0, 2.0, 4.0])


# compute maxwellian, without numerically normalizing the number density
def maxwellian(velocities,
               number_density,
               mean_velocity,
               temperature,
               mass):
    dim = velocities.shape[-1]
    # compute exponential with matching mean velocity and temperature
    distance = np.sum((velocities - mean_velocity) ** 2, axis=-1)
    exponential = np.exp(-0.5 * mass / temperature * distance)
    divisor = (2 * np.pi * temperature / mass) ** (dim / 2)
    result = number_density * exponential / divisor
    return result


""" PLOT: initial states for different temperatures on same model
    This is to illustrate the importance of mathcing temperatures.
"""
fig = bp.Plot.AnimatedFigure(figsize=(12.75, 4.25), dpi=None)
for i_t, temperature in enumerate(temperatures):
    # plot state (mean_v = 0)
    ax = fig.add_subplot((1, 3, i_t + 1), dim=3)
    state = maxwellian(model.vels,
                       number_density,
                       0,
                       temperature,
                       mass)
    s = 0
    vels = model.vels
    ax.plot(vels[..., 0], vels[..., 1], state)
    ax.mpl_axes.view_init(elev=20., azim=45)
    ax.mpl_axes.tick_params(axis="both", labelsize=fs_ticks)
    ax.mpl_axes.set_title(r"Parameter $\vartheta = " + str(temperature) + "$",
                          fontsize=fs_title)
fig._plt.tight_layout(pad=4)
plt.savefig(bp.SIMULATION_DIR + "/plot_initial_states.pdf")
