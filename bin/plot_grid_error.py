import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']
import matplotlib.pyplot as plt

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
line_style = ["-", "--", ":", "-.", "-", "--", "-.", ":"]
STYLE = {55: "-",
         77: "-.",
         66: "--",
         88: ":"}
COLOR = {55: "tab:blue",
         77: "tab:orange",
         66: "tab:green",
         88: "tab:red"}
MOMENTS = ["Temperature",
           "Number Density",
           "Momentum",
           "Mean Velocity",
           "Stress",
           "Heat Flow"]
SYMBOL = [r"T",
          r"\nu",
          r"M_x",
          r"\overline{v}_x",
          r"P_{xy}",
          r"q_x"]
#  Used Velocity Model
MASS = 1
NUMBER_DENSITY = 1.0
# setup velocity models
MODELS = dict()
MODELS[55] = bp.BaseModel([MASS],
                          [(5, 5)],
                          1,
                          [2])
MODELS[66] = bp.BaseModel([MASS],
                          [(6, 6)],
                          1,
                          [2])
MODELS[77] = bp.BaseModel([MASS],
                          [(7, 7)],
                          1,
                          [2])
MODELS[88] = bp.BaseModel([MASS],
                          [(8, 8)],
                          1,
                          [2])
MODELS[3131] = bp.BaseModel([MASS],
                            [(31, 31)],
                            1,
                            [2])
MODELS[3030] = bp.BaseModel([MASS],
                            [(30, 30)],
                            1,
                            [2])


# compute maxwellian, without numerically normalizing the number density
def maxwellian(velocities,
               number_density,
               mean_velocity,
               temperature,
               mass,
               force_number_density=True):
    dim = velocities.shape[-1]
    # compute exponential with matching mean velocity and temperature
    distance = np.sum((velocities - mean_velocity) ** 2, axis=-1)
    exponential = np.exp(-0.5 * mass / temperature * distance)
    if force_number_density:
        dv = np.max(np.abs(velocities[0] - velocities[1]))
        divisor = np.sum(exponential * dv**dim)
    else:
        divisor = (2 * np.pi * temperature / mass) ** (dim / 2)
    result = number_density * exponential / divisor
    return result


if __name__ == "__main__":
    # #################################################################################
    """ PLOT: Errors vs Temperature, for fixed shapes and mean velocity = 0
        This shows the discretisation and cutoff error"""
    #################################################################################
    fig, ax = plt.subplots(1, 3, constrained_layout=True,
                           sharex="all", sharey="all",
                           figsize=(12.75, 6.25))
    # PARAMS
    T_cont = np.linspace(0.1, 7, N)
    # store results in dict
    res_T = {m: {key: np.full(T_cont.size, np.nan)
                 for key in MOMENTS}
             for m in MODELS.keys()}

    for m, model in MODELS.items():
        assert isinstance(model, bp.BaseModel)
        mean_v = np.zeros((1, model.ndim))
        res = res_T[m]
        for i_t, t in enumerate(T_cont):
            distr = maxwellian(model.vels,
                               NUMBER_DENSITY,
                               mean_v,
                               t,
                               MASS)
            res["Number Density"][i_t] = model.cmp_number_density(distr)
            res["Momentum"][i_t] = model.cmp_momentum(distr)[0]
            res["Mean Velocity"][i_t] = model.cmp_mean_velocity(distr)[0]
            res["Temperature"][i_t] = model.cmp_temperature(distr)
            res["Stress"][i_t] = model.cmp_stress(distr)
            res["Heat Flow"][i_t] = model.cmp_heat_flow(distr)[0]

    # plot errors (normal and split up)
    for a in [0, 1, 2]:
        ax[a].set_axisbelow(True)
        ax[a].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                         linewidth=0.4)
        ax[a].set_xlabel(r"Temperature $T$", fontsize=18)
        for m, model in MODELS.items():
            if m in [3131, 3030]:
                continue
            if m % 2 == 1:
                offset = (T_cont - res_T[3131]["Temperature"])
            else:
                offset = (T_cont - res_T[3030]["Temperature"])

            if a == 0:
                error = (T_cont - res_T[m]["Temperature"])
            elif a == 1:
                error = offset
            elif a == 2:
                error = (T_cont - res_T[m]["Temperature"]) - offset
            else:
                raise NotImplementedError
            ax[a].plot(T_cont,
                       error,
                       label=model.shapes[0],
                       linestyle=STYLE[m],
                       color=COLOR[m],
                       linewidth=4)

    ax[0].legend(title="Grid Shapes", loc="upper left",
                 ncol=2)
    ax[0].set_ylabel(
        r"$\mathcal{E}_T^{\mathfrak{V}^s}(\overline{v}, T)$",
        fontsize=20
    )
    ax[1].set_ylabel(
        r"$\mathcal{E}_T^{\mathfrak{V}^s_\infty}(\overline{v}, T)$",
        fontsize=20
    )
    ax[2].set_ylabel(
        r"$\mathcal{E}_T^{\mathfrak{V}^s}(\overline{v}, T) - \mathcal{E}_T^{\mathfrak{V}^s_\infty}(\overline{v}, T)$",
        fontsize=20
    )

    ax[0].set_title("Total Error")
    ax[1].set_title("Discretization Error")
    ax[2].set_title("Domain Error")
    fig.suptitle("Isolated Temperature Errors for Different Grid Shapes and $\overline{v} = 0$")
    plt.show()

    ##################################################################################
    """ PLOT: Errors vs Mean Velocity, for fixed Shapes and mean v = 0
        This shows the discretisation and cutoff error"""
    #################################################################################
    fig, ax = plt.subplots(1, 3, constrained_layout=True,
                           sharex="all", sharey="all",
                           figsize=(12.75, 6.25))
    # use a fixed model, and several temperatures instead
    model = MODELS[77]
    # PARAMS
    TEMPERATURES = np.array([1.0, 2.0, 4.0])
    MEAN_VELS = np.zeros((N, 2))
    MEAN_VELS[:, 0] = np.linspace(0, model.max_vel, N)
    # store results in dict
    res_v = {m: {t: {key: np.full(T_cont.size, np.nan)
                     for key in MOMENTS}
                 for t in TEMPERATURES}
             for m in [77, 3131]}
    for m in [77, 3131]:
        model = MODELS[m]
        for t in TEMPERATURES:
            res = res_v[m][t]
            for i_v, v in enumerate(MEAN_VELS):
                distr = maxwellian(model.vels,
                                   NUMBER_DENSITY,
                                   v,
                                   t,
                                   MASS)
                res["Number Density"][i_v] = model.cmp_number_density(distr)
                res["Momentum"][i_v] = model.cmp_momentum(distr)[0]
                res["Mean Velocity"][i_v] = model.cmp_mean_velocity(distr)[0]
                res["Temperature"][i_v] = model.cmp_temperature(distr)
                res["Stress"][i_v] = model.cmp_stress(distr)
                res["Heat Flow"][i_v] = model.cmp_heat_flow(distr)[0]

    # plot errors (normal and split up)

    for a in [0, 1, 2]:
        for i_t, t in enumerate(TEMPERATURES):
            offset = t - res_v[3131][t]["Temperature"]
            error = t - res_v[77][t]["Temperature"]
            if a == 0:
                pass
            elif a == 1:
                error = offset
            elif a == 2:
                error = error - offset
            else:
                raise NotImplementedError

            ax[a].plot(MEAN_VELS[:, 0],
                       error,
                       label=t,
                       linestyle=line_style[i_t],
                       linewidth=4)
            ax[a].set_xlabel(r"Mean Velocity $\overline{v}_x$", fontsize=18)
            ax[a].set_axisbelow(True)
            ax[a].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                             linewidth=0.4)
    ax[0].legend(title="Temperatures", loc="upper left")
    ax[0].set_ylabel(
        r"$\mathcal{E}_T^{\mathfrak{V}^s}(\overline{v}, T)$",
        fontsize=20
    )
    ax[1].set_ylabel(
        r"$\mathcal{E}_T^{\mathfrak{V}^s_\infty}(\overline{v}, T)$",
        fontsize=20
    )
    ax[2].set_ylabel(
        r"$\mathcal{E}_T^{\mathfrak{V}^s}(\overline{v}, T) - \mathcal{E}_T^{\mathfrak{V}^s_\infty}(\overline{v}, T)$",
        fontsize=20
    )

    ax[0].set_title("Total Error")
    ax[1].set_title("Discretization Error")
    ax[2].set_title("Domain Error")
    fig.suptitle("Isolated Temperature Errors of a $(7, 7)$ Grid for Different Temperatures T")
    plt.show()

    ##################################################################################
    """ PLOT: both previous versions in a large, multirow, 2 column plot
    each row for another moment"""
    #################################################################################
    fig, ax = plt.subplots(len(MOMENTS), 2, constrained_layout=True, sharex="col",
                           figsize=(8.27, 11.69))
    for row, moment in enumerate(MOMENTS):
        for i in [0, 1]:
            ax[row, i].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                                  linewidth=0.4)
            ax[row, i].xaxis.grid(color='darkgray', linestyle='dashed', which="both",
                                  linewidth=0.4)
        ax[row, 0].set_ylabel(
            moment + " $" + SYMBOL[row] + "$",
            # + r"\\$\displaystyle\mathcal{E}_{"
            # + SYMBOL[row]
            # + r"}^{\mathfrak{V}^s}(\overline{v}, T)$",
            fontsize=16)
        # plot left side: variable temperature, set of different shapes, fixed v=0
        for m, model in MODELS.items():
            if m in [3131, 3030]:
                continue
            value = res_T[m][moment]
            if moment == "Number Density":
                expected = NUMBER_DENSITY
            elif moment == "Momentum":
                expected = 0
            elif moment == "Mean Velocity":
                expected = 0
            elif moment == "Temperature":
                expected = T_cont
            elif moment == "Stress":
                expected = 0
            elif moment == "Heat Flow":
                expected = 0
            else:
                raise NotImplementedError
            error = expected - value

            ax[row, 0].plot(T_cont,
                            error,
                            label=model.shapes[0],
                            linestyle=STYLE[m],
                            color=COLOR[m],
                            linewidth=4)

            if row == 0:
                ax[row, 0].legend(title="Grid Shapes", loc="upper left", ncol=2,
                                  fontsize=8)
                ax[row, 0].set_title("Total Errors for Different Grid Shapes and $\overline{v} = 0$")
            if row == len(MOMENTS) - 1:
                ax[row, 0].set_xlabel(r"Temperature $T$", fontsize=18)

        # plot right side: variable velocites, set of temperatures, fixed shape
        for i_t, t in enumerate(TEMPERATURES):
            value = res_v[77][t][moment]
            if moment == "Number Density":
                expected = NUMBER_DENSITY
            elif moment == "Momentum":
                expected = NUMBER_DENSITY * MASS * MEAN_VELS[..., 0]
            elif moment == "Mean Velocity":
                expected = MEAN_VELS[..., 0]
            elif moment == "Temperature":
                expected = t
            elif moment == "Stress":
                expected = 0
            elif moment == "Heat Flow":
                expected = 0
            else:
                raise NotImplementedError
            error = expected - value
            ax[row, 1].plot(MEAN_VELS[:, 0],
                            error,
                            label=t,
                            linestyle=line_style[i_t],
                            linewidth=4)

            if row == 0:
                ax[row, 1].legend(title="Temperatures", loc="upper left",
                                  fontsize=8)
                ax[row, 1].set_title("Total Errors of a $(7, 7)$ Grid for Different Temperatures T")
            if row == len(MOMENTS) - 1:
                ax[row, 1].set_xlabel(r"Mean Velocity $\overline{v}$", fontsize=18)

    plt.show()
