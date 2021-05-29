import boltzpy as bp
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
from plot_grid_error import maxwellian

"""
Create plots for the discretization error 
for a given basemodel.
"""


#  Used Velocity Model
MASS = 1
NUMBER_DENSITY = 1.0
# setup velocity models
MAX_VEL = 6
MAX_T = 7
MODEL = bp.BaseModel([MASS],
                     [(31, 31)],
                     1,
                     [2])

# number of discretization points
N = 6 * 31 + 1
ORIGINAL_DIRECTIONS = np.array([[1, 0], [3, 1], [6, 1]])
DIRECTIONS = ORIGINAL_DIRECTIONS / np.max(ORIGINAL_DIRECTIONS, axis=-1)[:, None]

STYLE = ["-", "--", ":", "-.", ":"]
if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2, constrained_layout=True,
                           figsize=(12.75, 6.25))

    ##############################################################
    # LEFT PLOT: Heatmap of DISCRETIZATION ERROR for each v (2D) #
    ##############################################################
    T = 1
    # Velocities in XY plane -> 2D values
    VELS = np.empty((N, N, 2), dtype=float)
    VELS[..., 0] = np.linspace(0, MAX_VEL, N)[:, None]
    VELS[..., 1] = np.linspace(0, MAX_VEL, N)[None]
    VELS = VELS.reshape((-1, 2))
    res1 = np.full(N**2, np.nan, dtype=float)

    for v, vel in enumerate(VELS):
        distr = maxwellian(MODEL.vels,
                           NUMBER_DENSITY,
                           vel,
                           T,
                           MASS)
        # correct value is 1
        res1[v] = T - MODEL.cmp_temperature(distr)
    print("Error range: ", res1.min(), res1.max())

    # plot error heatmap
    res1 = res1.reshape((N,N))
    hm = ax[0].imshow(res1, cmap=plt.cm.RdBu, interpolation='spline16',
                      extent=(0, MAX_VEL, 0, MAX_VEL)
                      )
    # add colorbar (requires a separate ax, that is added by relative positions)
    # cax = fig.add_axes([0.04, 0.1, 0.27, 0.025])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(hm, cax=cax, orientation='vertical')
    # plt.colorbar(hm, ax=ax[0], orientation='vertical')

    # plot grid points
    choice = np.where(np.all((MODEL.vels <= MAX_VEL) & (MODEL.vels >= 0),
                             axis=-1))
    points = MODEL.vels[choice]
    ax[0].scatter(points[:, 0], points[:, 1], c="tab:orange", marker="x", s=100)
    ax[0].set_title(r"Discretization error for all $\mathfrak{v} \in \mathfrak{V}^s$ and arbitrary T")

    ax[0].set_xlabel(r"Mean velocity $\overline{v}_x$", fontsize=18)
    ax[0].set_ylabel(r"Mean velocity $\overline{v}_y$", fontsize=18)


    # ##############################################################
    # # MIDDLE PLOT: Discretization Error for different directions #
    # ##############################################################
    # T = 1
    # res2 = np.full((len(DIRECTIONS), N), np.nan)
    # for d, direction in enumerate(DIRECTIONS):
    #     for f, factor in enumerate(np.linspace(0, MAX_VEL, N)):
    #         mean_v = factor * direction
    #         distr = maxwellian(MODEL.vels,
    #                            NUMBER_DENSITY,
    #                            mean_v,
    #                            T,
    #                            MASS)
    #         res2[d, f] = T - MODEL.cmp_temperature(distr)
    #
    # # plot errors
    # for d, dir in enumerate(DIRECTIONS):
    #     ax[1].plot(np.linspace(0, MAX_VEL, N),
    #                res2[d],
    #                label=tuple(ORIGINAL_DIRECTIONS[d]),
    #                linewidth=2,
    #                linestyle=STYLE[d])
    #
    # ax[1].set_axisbelow(True)
    # ax[1].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
    #                  linewidth=0.4)
    # ax[1].set_xlabel(r"Mean velocity $\overline{v}_x$", fontsize=18)
    #
    # ax[1].legend(title="Directions", loc="lower left")
    # ax[1].set_ylabel(
    #     r"$\mathcal{E}_{\mathfrak{V}^s_\infty}(\overline{v}, T)$",
    #     fontsize=20
    # )
    # ax[1].set_title("Discretization error along different directions")

    ##############################################################
    # RIGHT PLOT: Discretization Error Amplitude vs. Temperature #
    ##############################################################
    VELS = np.array([[0, 0], [1, 1]])
    # use quadratic scale for smaller temperatures
    TEMPERATURES = np.concatenate(([T**2 for T in np.linspace(0, 1, N)[1:]],
                                   np.linspace(1, 7, N)[1:]))

    # add first axis in case I want to add more Velocities or Specimen
    res3 = np.full((len(VELS), len(TEMPERATURES)), np.nan)
    for v, vel in enumerate(VELS):
        for t, T in enumerate(TEMPERATURES):
            distr = maxwellian(MODEL.vels,
                               NUMBER_DENSITY,
                               vel,
                               T,
                               MASS)
            try:
                res3[v, t] = np.abs(T - MODEL.cmp_temperature(distr))
            except ValueError:
                print("Error at T = ", T)
                res3[v, t] = 1

    # plot discretization error
    for v, vel in enumerate(VELS):
        ax[1].plot(TEMPERATURES,
                   res3[v],
                   label=tuple(vel),
                   linestyle=STYLE[v],
                   linewidth=2)
    ax[1].set_axisbelow(True)
    ax[1].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                     linewidth=0.4)
    ax[1].xaxis.grid(color='darkgray', linestyle='dashed', which="both",
                     linewidth=0.4)
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r" Temperature $T$", fontsize=18)
    ax[1].set_ylabel(
        r"$\left\lvert\mathcal{E}_{\mathfrak{V}^s_\infty}(\overline{v}, T) \right\rvert$",
        fontsize=20)

    ax[1].legend(title="Mean velocity $\overline{v}$", loc="lower left")
    ax[1].set_title("Amplitude of discretization error")
    fig.tight_layout()
    plt.show()
