import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks, fs_legend_title

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
N = 401
# plot line styles
line_style = ["-", "--", ":", "-.", "-", "--", "-.", ":"]
STYLE = {55: "-",
         77: "-.",
         66: "--",
         88: ":"}
MODEL_COLOR = {55: "tab:pink",
         77: "tab:brown",
         66: "tab:cyan",
         88: "tab:orange"}
TEMP_COLOR = ["tab:blue",
              "tab:green",
              "tab:red", "tab:purple"]
MOMENTS = ["Temperature",
           "Number Density",
           "Mean Velocity",
           "Momentum",
           "Heat Flow",
           "Stress"]
SYMBOL = [r"T",
          r"\nu",
          r"\overline{v}_x",
          r"M_x",
          r"q_x",
          r"P_{xy}"]
#  Used Velocity Model
MASS = 1
NUMBER_DENSITY = 1.0
MAX_T = 8
FORCE_NUMBER_DENSITY = True
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


def cmp_temp_dependent_errors(models, temperatures, velocities, moments,
                              force_number_density=True, mass=1, number_density=1):
    # store results in dict
    res_T = {m: {key: np.full(temperatures.size, np.nan)
                 for key in moments}
             for m in models.keys()}

    for m, model in models.items():
        res = res_T[m]
        for i_t, t in enumerate(temperatures):
            for moment in moments:
                mean_v = velocities[m][moment].reshape((1, model.ndim))
                distr = maxwellian(model.vels,
                                   number_density,
                                   mean_v,
                                   t,
                                   mass,
                                   force_number_density=force_number_density)
                if moment == "Number Density":
                    res[moment][i_t] = model.cmp_number_density(distr)
                elif moment == "Temperature":
                    res[moment][i_t] = model.cmp_temperature(distr)
                elif moment == "Momentum":
                    res[moment][i_t] = model.cmp_momentum(distr)[0]
                elif moment == "Mean Velocity":
                    res[moment][i_t] = model.cmp_mean_velocity(distr)[0]
                elif moment == "Stress":
                    res[moment][i_t] = model.cmp_stress(distr)
                elif moment == "Heat Flow":
                    res[moment][i_t] = model.cmp_heat_flow(distr)[0]
                else:
                    raise NotImplementedError
    return res_T


def cmp_vel_dependent_errors(model_dict, temperatures, velocities, moments,
                             force_number_density, mass=1, number_density=1):
    # store results in dict
    res_v = {m: {t: {key: np.full(velocities.shape[0], np.nan)
                     for key in MOMENTS}
                 for t in temperatures}
             for m in model_dict.keys()}
    for m, model in model_dict.items():
        for t in temperatures:
            res = res_v[m][t]
            for i_v, v in enumerate(velocities):
                distr = maxwellian(model.vels,
                                   number_density,
                                   v,
                                   t,
                                   mass,
                                   force_number_density=force_number_density)
                for moment in moments:
                    if moment == "Number Density":
                        res[moment][i_v] = model.cmp_number_density(distr)
                    elif moment == "Temperature":
                        res[moment][i_v] = model.cmp_temperature(distr)
                    elif moment == "Momentum":
                        res[moment][i_v] = model.cmp_momentum(distr)[0]
                    elif moment == "Mean Velocity":
                        res[moment][i_v] = model.cmp_mean_velocity(distr)[0]
                    elif moment == "Stress":
                        res[moment][i_v] = model.cmp_stress(distr)
                    elif moment == "Heat Flow":
                        res[moment][i_v] = model.cmp_heat_flow(distr)[0]
                    else:
                        raise NotImplementedError
    return res_v


# mean_v[0] = 0.25 * model.spacings[0] * model.base_delta
# distr = maxwellian(model.vels,
#                    NUMBER_DENSITY,
#                    mean_v,
#                    t,
#                    mass,
#                    force_number_density=force_number_density)

if __name__ == "__main__":
    # #################################################################################
    """ PLOT: Errors vs Temperature, for fixed shapes and mean velocity = 0
        This shows the discretisation and cutoff error"""
    #################################################################################
    fig, ax = plt.subplots(1, 3,
                           sharex="all", sharey="all",
                           figsize=(12.75, 6.25))
    T_TEMPS = np.linspace(0.1, MAX_T, N)

    T_VELS = dict()
    for m, model in MODELS.items():
        T_VELS[m] = dict()
        for mom in MOMENTS:
            if mom in ["Temperature", "Number Density"]:
                T_VELS[m][mom] = np.zeros(model.ndim)
            elif mom in ["Mean Velocity", "Momentum",  "Heat Flow"]:
                T_VELS[m][mom] = np.zeros(model.ndim)
                T_VELS[m][mom][0] = 0.25 * model.spacings[0] * model.base_delta
            elif mom in ["Stress"]:
                T_VELS[m][mom] = np.full(model.ndim,
                                         0.25 * model.spacings[0] * model.base_delta)
            else:
                raise NotImplementedError
    res_T = cmp_temp_dependent_errors(MODELS, T_TEMPS, T_VELS, MOMENTS,
                                      force_number_density=False,
                                      mass=1, number_density=1)
    # plot errors (normal and split up)
    fig.suptitle(r"Isolated Grid Errors for Different DVM and $\widetilde{v} = 0$",
                 fontsize=fs_suptitle)
    for a in [0, 1, 2]:
        ax[a].set_axisbelow(True)
        ax[a].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                         linewidth=0.4)
        ax[a].set_xlabel(r"Temperature Parameter $\vartheta$", fontsize=fs_label)
        ax[a].tick_params(axis="both", labelsize=fs_ticks)
        for m, model in MODELS.items():
            if m in [3131, 3030]:
                continue
            if m % 2 == 1:
                offset = (T_TEMPS - res_T[3131]["Temperature"])
            else:
                offset = (T_TEMPS - res_T[3030]["Temperature"])

            if a == 0:
                error = (T_TEMPS - res_T[m]["Temperature"])
            elif a == 1:
                error = offset
            elif a == 2:
                error = (T_TEMPS - res_T[m]["Temperature"]) - offset
            else:
                raise NotImplementedError
            ax[a].plot(T_TEMPS,
                       error,
                       label=model.shapes[0],
                       linestyle=STYLE[m],
                       color=MODEL_COLOR[m],
                       linewidth=3)

    ax[0].legend(title="Grid Shapes", loc="upper left",
                 ncol=2,
                 fontsize=fs_legend,
                 title_fontsize=fs_legend_title)
    ax[0].set_ylabel(
        r"$\mathcal{E}_{\mathfrak{V}^s}(\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    ax[1].set_ylabel(
        r"$\mathcal{E}_{\mathfrak{V}^s_\infty}(\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    ax[2].set_ylabel(
        r"$\mathcal{E}_{\mathfrak{V}^s}(\widetilde{v}, \vartheta) - \mathcal{E}_{\mathfrak{V}^s_\infty}(\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )

    ax[0].set_title("Total Error", fontsize=fs_title)
    ax[1].set_title("Isolated Discretization Error", fontsize=fs_title)
    ax[2].set_title("Isolated Domain Error", fontsize=fs_title)
    # fig.suptitle("Isolated Temperature Based Errors "
    #              "for Different Grid Shapes and $\overline{v} = 0$", fontsize=18)
    plt.tight_layout(pad=2)
    plt.subplots_adjust(top=0.85)
    plt.savefig(bp.SIMULATION_DIR + "/grid_err_T.pdf")
    plt.close(fig)

    ##################################################################################
    """ PLOT: Errors vs Mean Velocity, for fixed Shapes and mean v = 0
        This shows the discretisation and cutoff error"""
    #################################################################################
    fig, ax = plt.subplots(1, 3,
                           sharex="all",
                           figsize=(12.75, 6.25))
    # use a fixed model, and several temperatures instead
    model = MODELS[77]
    V_TEMPS = np.array([1.0, 2.0, 4.0])
    V_VELS = np.zeros((N, 2))
    V_VELS[:, 0] = np.linspace(0, model.max_vel, N)
    V_VELS[:, 1] = np.linspace(0, model.max_vel, N)
    V_MODELS = {m: MODELS[m] for m in [77, 3131]}
    res_v = cmp_vel_dependent_errors(V_MODELS, V_TEMPS, V_VELS, MOMENTS,
                                     False, mass=1, number_density=1)
    # plot errors (normal and split up)
    for a in [0, 1, 2]:
        for i_t, t in enumerate(V_TEMPS):
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

            ax[a].plot(V_VELS[:, 0],
                       error,
                       label=t,
                       color=TEMP_COLOR[i_t],
                       linestyle=line_style[i_t],
                       linewidth=3)
            ax[a].set_xlabel(r"Velocity Parameters $\widetilde{v}_x = \widetilde{v}_y$",
                             fontsize=fs_label)
            ax[a].set_axisbelow(True)
            ax[a].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                             linewidth=0.4)
            ax[a].tick_params(axis="both", labelsize=fs_ticks)
    ax[0].legend(title=r"Parameter $\vartheta$", loc="upper left",
                 fontsize=fs_legend,
                 title_fontsize=fs_legend_title)
    ax[0].set_ylabel(
        r"$\mathcal{E}_{\mathfrak{V}^s}(\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    ax[1].set_ylabel(
        r"$\mathcal{E}_{\mathfrak{V}^s_\infty}(\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    ax[2].set_ylabel(
        r"$\mathcal{E}_{\mathfrak{V}^s}(\widetilde{v}, \vartheta) - \mathcal{E}_{\mathfrak{V}^s_\infty}(\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )

    ax[0].set_title("Total Error", fontsize=fs_title)
    ax[1].set_title("Isolated Discretization Error", fontsize=fs_title)
    ax[2].set_title("Isolated Domain Error", fontsize=fs_title)
    fig.suptitle(r"Isolated Grid Errors for a $(7,7)$ DVM and $\vartheta \in \{1,2,4\}$",
                 fontsize=fs_suptitle)
    plt.tight_layout(pad=2)
    plt.subplots_adjust(top=0.85)
    plt.savefig(bp.SIMULATION_DIR + "/grid_err_v.pdf")
    plt.close()

    ##################################################################################
    """ PLOT: both previous versions moment by moment in a 3 column plot"""
    #################################################################################
    for i_mom, moment in enumerate(MOMENTS):
        fig, ax = plt.subplots(1, 3, sharex="col",
                               figsize=(12.75, 6.25))
        ax = ax.reshape((1, 3))
        row = 0
        # plot left side: variable temperature, set of different shapes, fixed v=0
        for m, model in MODELS.items():
            if m in [3131, 3030]:
                continue
            value = res_T[m][moment]
            if moment == "Number Density":
                expected = NUMBER_DENSITY
            elif moment == "Momentum":
                # Todo just using the x component of velocities
                expected = NUMBER_DENSITY * MASS * T_VELS[m][moment][..., 0]
            elif moment == "Mean Velocity":
                expected = T_VELS[m][moment][..., 0]
            elif moment == "Temperature":
                expected = T_TEMPS
            elif moment == "Stress":
                expected = 0
            elif moment == "Heat Flow":
                expected = 0
            else:
                raise NotImplementedError
            total_error = expected - value
            ax[row, 0].plot(T_TEMPS,
                            total_error,
                            label=model.shapes[0],
                            linestyle=STYLE[m],
                            color=MODEL_COLOR[m],
                            linewidth=3)

        # plot right side: variable velocites, set of temperatures, fixed shape
        for i_t, t in enumerate(V_TEMPS):
            value = res_v[77][t][moment]
            if moment == "Number Density":
                expected = NUMBER_DENSITY
            elif moment == "Momentum":
                expected = NUMBER_DENSITY * MASS * V_VELS[..., 0]
            elif moment == "Mean Velocity":
                expected = V_VELS[..., 0]
            elif moment == "Temperature":
                expected = t
            elif moment == "Stress":
                expected = 0
            elif moment == "Heat Flow":
                expected = 0
            else:
                raise NotImplementedError
            total_error = expected - value
            discr_error = expected - res_v[3131][t][moment]
            dom_error = total_error - discr_error
            ax[row, 1].plot(V_VELS[:, 0],
                            discr_error,
                            label=t,
                            color=TEMP_COLOR[i_t],
                            linestyle=line_style[i_t],
                            linewidth=3)
            ax[row, 2].plot(V_VELS[:, 0],
                            dom_error,
                            label=r"$\vartheta = " + str(t) + "$",
                            color=TEMP_COLOR[i_t],
                            linestyle=line_style[i_t],
                            linewidth=3)

        # add grids
        for i in [0, 1, 2]:
            ax[row, i].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                                  linewidth=0.4)
            ax[row, i].xaxis.grid(color='darkgray', linestyle='dashed', which="both",
                                  linewidth=0.4)

        # add moment on the left
        if moment == "Mean Velocity":
            ax[row, 0].set_ylabel(
                r"$\widetilde{v} - \overline{v}$",
                fontsize=fs_label)
        else:
            ax[row, 0].set_ylabel(
                moment + " Based Errors",
                #" $" + SYMBOL[row] + "$",
                # + r"\\$\displaystyle\mathcal{E}_{"
                # + SYMBOL[row]
                # + r"}^{\mathfrak{V}^s}(\widetilde{v}, \vartheta)$",
                fontsize=fs_label)

        # set legends and title
        # fig.suptitle("Isolated " + moment + " Based Errors"
        #              + r" for different $T$ and $\overline{v}_x$" ,
        #              fontsize=18)
        ax[row, 0].legend(title="Grid Shapes", loc="upper right", ncol=2,
                          fontsize=fs_legend,
                          title_fontsize=fs_legend_title)
        ax[row, 0].set_title("Total Errors for $\widetilde{v} = "
                             + str(tuple(T_VELS[m][moment])) + "$",
                             fontsize=fs_title
                             # "for Different Grid Shapes and $\widetilde{v} = 0$"
                             )
        # ax[row, 1].legend(title="Temperatures", loc="upper left",
        #                   fontsize=8)
        ax[row, 1].set_title("Discretization Errors", fontsize=fs_title
                             # " of a $(7, 7)$ Grid for Different Temperatures T"
                             )
        ax[row, 2].legend(loc="upper left", title="Temperature Parameter",
                          fontsize=fs_legend + 1,
                          title_fontsize=fs_legend_title)
        ax[row, 2].set_title("Domain Errors", fontsize=fs_title
                             # " of a $(7, 7)$ Grid for Different Temperatures T"
                             )

        # add parameter at bottom
        ax[row, 0].set_xlabel(r"Temperature Parameter $\vartheta$", fontsize=fs_label)
        ax[row, 1].set_xlabel(r"Velocity Parameter $\widetilde{v}_x$",
                              fontsize=fs_label)
        ax[row, 2].set_xlabel(r"Velocity Parameter $\widetilde{v}_x$",
                              fontsize=fs_label)
        for c in [0,1,2]:
            ax[row, c].tick_params(axis="both", labelsize=fs_ticks)

        ax[row, 0].set_ylabel(
            r"$" + SYMBOL[i_mom]
            + r"\left[{\mathfrak{V}^s}\right](\widetilde{v}, \vartheta)$",
            fontsize=fs_label
        )
        ax[row, 1].set_ylabel(
            r"$" + SYMBOL[i_mom]
            + r"\left[{\mathfrak{V}^s_\infty}\right](\widetilde{v}, \vartheta)$",
            fontsize=fs_label
        )
        ax[row, 2].set_ylabel(
            r"$" + SYMBOL[i_mom]
            + r"\left[{\mathfrak{V}^s}\right](\widetilde{v}, \vartheta) - "
            + SYMBOL[i_mom]
            + r"\left[{\mathfrak{V}^s_\infty}\right](\widetilde{v}, \vartheta)$",
            fontsize=fs_label
        )

        plt.suptitle("Isolated " + moment + " Based Grid Errors",
                     fontsize=fs_suptitle)
        plt.tight_layout(pad=2)
        plt.subplots_adjust(top=0.85)
        plt.savefig(bp.SIMULATION_DIR + "/mom_err_" + moment + ".pdf")
        plt.close(fig)


    print("################################################################\n"
          "Plot Stress Error, this was formeerly a large multiplot\n"
          "################################################################")
    # compute all again, but force number density
    res_T_normed = cmp_temp_dependent_errors(MODELS, T_TEMPS, T_VELS, MOMENTS,
                                             force_number_density=True,
                                             mass=1, number_density=1)
    res_v_normed = cmp_vel_dependent_errors(V_MODELS, V_TEMPS, V_VELS, MOMENTS,
                                            force_number_density=True,
                                            mass=1, number_density=1)

    # create plot
    ordererd_moments = [
        "Stress"]
    ordered_symbols = [
        r"P_{xy}"]
    fig, ax = plt.subplots(len(ordererd_moments), 3, constrained_layout=True, sharex="col",
                           figsize=(12.75, 6.25))
    ax = np.array(ax, ndmin=2)

    for row, moment in enumerate(ordererd_moments):
        # plot left side: variable temperature, set of different shapes, fixed v=0
        for m, model in MODELS.items():
            if m in [3131, 3030]:
                continue
            value = res_T_normed[m][moment]
            if moment == "Number Density":
                expected = NUMBER_DENSITY
            elif moment == "Momentum":
                expected = NUMBER_DENSITY * MASS * T_VELS[m][moment][..., 0]
            elif moment == "Mean Velocity":
                expected = T_VELS[m][moment][..., 0]
            elif moment == "Temperature":
                expected = T_TEMPS
            elif moment == "Stress":
                expected = 0
            elif moment == "Heat Flow":
                expected = 0
            else:
                raise NotImplementedError
            total_error = expected - value
            ax[row, 0].plot(T_TEMPS,
                            total_error,
                            label=model.shapes[0],
                            linestyle=STYLE[m],
                            color=MODEL_COLOR[m],
                            linewidth=3)

        # plot right side: variable velocites, set of temperatures, fixed shape
        for i_t, t in enumerate(V_TEMPS):
            value = res_v_normed[77][t][moment]
            if moment == "Number Density":
                expected = NUMBER_DENSITY
            elif moment == "Momentum":
                expected = NUMBER_DENSITY * MASS * V_VELS[..., 0]
            elif moment == "Mean Velocity":
                expected = V_VELS[..., 0]
            elif moment == "Temperature":
                expected = t
            elif moment == "Stress":
                expected = 0
            elif moment == "Heat Flow":
                expected = 0
            else:
                raise NotImplementedError
            total_error = expected - value
            discr_error = expected - res_v_normed[3131][t][moment]
            dom_error = total_error - discr_error
            ax[row, 1].plot(V_VELS[..., 0],
                            discr_error,
                            color=TEMP_COLOR[i_t],
                            label=t,
                            linestyle=line_style[i_t],
                            linewidth=3)
            ax[row, 2].plot(V_VELS[..., 0],
                            dom_error,
                            color=TEMP_COLOR[i_t],
                            label=t,
                            linestyle=line_style[i_t],
                            linewidth=3)
        for a in [0,1,2]:
            ylim = ax[row, a].get_ylim()
            if ylim[1] - ylim[0] < 1e-12:
                ax[row, a].set_ylim(-1e-12, 2e-12)


        # add grids
        for i in [0, 1, 2]:
            ax[row, i].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                                  linewidth=0.4)
            ax[row, i].xaxis.grid(color='darkgray', linestyle='dashed', which="both",
                                  linewidth=0.4)

        # add moment on the left
        ax[row, 0].set_ylabel(
            moment + " $" + ordered_symbols[row] + "$",
            # + r"\\$\displaystyle\mathcal{E}_{"
            # + SYMBOL[row]
            # + r"}^{\mathfrak{V}^s}(\widetilde{v}, \vartheta)$",
            fontsize=fs_label)

        # set legends and title
        if row == 0:
            loc = "upper center"
            ax[row, 0].legend(title="Grid Shapes", loc=loc, ncol=2,
                              fontsize=fs_legend,
                              title_fontsize=fs_legend_title)
            ax[row, 0].set_title("Total Error",
                                 fontsize=fs_title
                                 )
            ax[row, 1].legend(title="Temperatures", loc="upper right",
                              fontsize=fs_legend,
                              title_fontsize=fs_legend_title)
            ax[row, 1].set_title("Discretization Error",
                                 fontsize=fs_title
                                 )
            ax[row, 2].legend(title="Temperatures", loc="upper right",
                              fontsize=fs_legend,
                              title_fontsize=fs_legend_title)
            ax[row, 2].set_title("Domain Error",
                                 fontsize=fs_title
                                 )

        # add parameter at bottom
        if row == len(ordererd_moments) - 1:
            ax[row, 0].set_xlabel(r"Temperature Parameter $\vartheta$",
                                  fontsize=fs_label)
            ax[row, 1].set_xlabel(r"Velocity Parameter $\widetilde{v}_x$",
                                  fontsize=fs_label)
            ax[row, 2].set_xlabel(r"Velocity Parameter $\widetilde{v}_x$",
                                  fontsize=fs_label)
        for c in [0,1,2]:
            ax[row, c].tick_params(axis="both", labelsize=fs_ticks)

    plt.savefig(bp.SIMULATION_DIR + "/grid_err_stress_nondiag.pdf")
    plt.close()


    print("################################\n"
          "PLOT: Stress Diagonal Components\n"
          "################################")
    fig, ax = plt.subplots(1, 3,
                           sharex="all",
                           figsize=(12.75, 6.25))
    # use a fixed model, and several temperatures instead
    model = MODELS[77]
    TEMPS = np.array([1, 2, 4])
    VELS = np.zeros((N, 2))
    VELS[:, 0] = np.linspace(0, model.max_vel, N)
    VELS[:, 1] = 0
    MODELS = {m: MODELS[m] for m in [77, 3131]}
    # store results in here
    stress_diag = {m: {t: {i_dir: np.full(VELS.shape[0], np.nan)
                           for i_dir in [0,1]}
                       for t in TEMPS}
                   for m in [77, 3131]}

    for m, model in MODELS.items():
        for t in TEMPS:
            res = stress_diag[m][t]
            for i_v, v in enumerate(VELS):
                distr = maxwellian(model.vels,
                                   1,
                                   v,
                                   t,
                                   1,
                                   force_number_density=True)
                for i_dir in [0, 1]:
                    direction = np.zeros((2, 2))
                    direction[:, i_dir] = 1
                    res[i_dir][i_v] = model.cmp_stress(distr, directions=direction)

    LABEL_P = [r"$P_{%1d, %1d}$" % (i+1, i+1)
               for i in [0,1]]
    LABELS_THETA = [r", $\vartheta = %1d $" % t for t in TEMPS]
    STYLES = ["dotted", "solid", ]
    # plot errors (normal and split up)
    for a in [0, 1, 2]:
        for i_t, t in enumerate(TEMPS):
            for i_dir in [0,1]:
                offset = stress_diag[3131][t][i_dir]
                error = stress_diag[77][t][i_dir]
                if a == 0:
                    pass
                elif a == 1:
                    error = offset
                elif a == 2:
                    error = error - offset
                else:
                    raise NotImplementedError

                ax[a].plot(VELS[:, 0],
                           error,
                           label=LABEL_P[i_dir] + LABELS_THETA[i_t] if a == 0 else "_nolegend_",
                           color=TEMP_COLOR[i_t],
                           linestyle=STYLES[i_dir],
                           linewidth=3)
        ax[a].set_xlabel(r"Mean Velocity Parameter $\widetilde{v}_x$",
                         fontsize=fs_label)
        ax[a].set_axisbelow(True)
        ax[a].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                         linewidth=0.4)

        ax[a].tick_params(axis="both", labelsize=fs_ticks)
    lg = fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15),
                    fontsize=fs_legend + 6,
                    title_fontsize=fs_legend_title)
    ax[0].set_ylabel(
        r"$P_{i, i}\left[{\mathfrak{V}^s}\right](\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    ax[1].set_ylabel(
        r"$P_{i, i}\left[{\mathfrak{V}^s_\infty}\right](\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    ax[2].set_ylabel(
        r"$P_{i, i}\left[{\mathfrak{V}^s}\right](\widetilde{v}, \vartheta) - P_{i, i}\left[{\mathfrak{V}^s_\infty}\right]((\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    #
    ax[0].set_title("Stress Diagonal Entries", fontsize=fs_title)
    ax[1].set_title("Discretization Related Part", fontsize=fs_title)
    ax[2].set_title("Domain Related Part", fontsize=fs_title)
    st = fig.suptitle(r"Diagonal Entries of the Stress Tensor of a $(7, 7)$ DVM"
                      r" for $\widetilde{v}_x \in [0, 6]$ and $\widetilde{v}_y = 0$",
                      fontsize=fs_suptitle)
    plt.tight_layout(pad=2)
    # fig.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(top=0.85)
    plt.savefig(bp.SIMULATION_DIR + "/grid_error_stress_diagonal.pdf",
                bbox_extra_artists=(lg, st),
                bbox_inches='tight')
    plt.close()


    print("################################\n"
          "PLOT: Error for Heat Flow \n"
          "################################")
    fig, ax = plt.subplots(1, 3,
                           sharex="all",
                           figsize=(12.75, 6.25))
    # use a fixed model, and several temperatures instead
    model = MODELS[77]
    TEMPS = np.array([1, 2, 4])
    VELS = np.zeros((N, 2))
    VELS[:, 0] = np.linspace(0, model.max_vel, N)
    VELS[:, 1] = 0
    MODELS = {m: MODELS[m] for m in [77, 3131]}
    # store results in here
    heat_flow = {m: {t: {i_dir: np.full(VELS.shape[0], np.nan)
                         for i_dir in [0, 1]}
                     for t in TEMPS}
                 for m in [77, 3131]}

    for m, model in MODELS.items():
        for t in TEMPS:
            res = stress_diag[m][t]
            for i_v, v in enumerate(VELS):
                distr = maxwellian(model.vels,
                                   1,
                                   v,
                                   t,
                                   1,
                                   force_number_density=True)
                for i_dir in [0, 1]:
                    direction = np.zeros(2)
                    direction[i_dir] = 1
                    res[i_dir][i_v] = model.cmp_heat_flow(distr, direction=direction)

    LABEL_P = [r"$q_{%1d}$" % (i + 1)
               for i in [0, 1]]
    LABELS_THETA = [r", $\vartheta = %1d $" % t for t in TEMPS]
    STYLES = ["dotted", "solid", ]
    # plot errors (normal and split up)
    for a in [0, 1, 2]:
        for i_t, t in enumerate(TEMPS):
            for i_dir in [0, 1]:
                offset = stress_diag[3131][t][i_dir]
                error = stress_diag[77][t][i_dir]
                if a == 0:
                    pass
                elif a == 1:
                    error = offset
                elif a == 2:
                    error = error - offset
                else:
                    raise NotImplementedError

                ax[a].plot(VELS[:, 0],
                           error,
                           label=LABEL_P[i_dir] + LABELS_THETA[i_t] if a == 0 else "_nolegend_",
                           color=TEMP_COLOR[i_t],
                           linestyle=STYLES[i_dir],
                           linewidth=3)
        ax[a].set_xlabel(r"Mean Velocity Parameter $\widetilde{v}_x$",
                         fontsize=fs_label)
        ax[a].set_axisbelow(True)
        ax[a].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                         linewidth=0.4)
        ax[a].tick_params(axis="both", labelsize=fs_ticks)

    lg = fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15),
                    fontsize=fs_legend + 6,
                    title_fontsize=fs_legend_title)
    ax[0].set_ylabel(
        r"$q_{i}\left[{\mathfrak{V}^s}\right](\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    ax[1].set_ylabel(
        r"$q_{i}\left[{\mathfrak{V}^s_\infty}\right](\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    ax[2].set_ylabel(
        r"$	q_{i}\left[{\mathfrak{V}^s}\right](\widetilde{v}, \vartheta) "
        r"- q_{i}\left[{\mathfrak{V}^s_\infty}\right](\widetilde{v}, \vartheta)$",
        fontsize=fs_label
    )
    #
    ax[0].set_title("Heat Flow", fontsize=fs_title)
    ax[1].set_title("Discretization Related Part", fontsize=fs_title)
    ax[2].set_title("Domain Related Part", fontsize=fs_title)
    st = fig.suptitle(r"Heat Flow Components of a $(7, 7)$ DVM"
                      r" for $\widetilde{v}_x \in [0, 6]$ and $\widetilde{v}_y = 0$",
                      fontsize=fs_suptitle)
    plt.tight_layout(pad=2)
    plt.subplots_adjust(top=0.85)
    plt.savefig(bp.SIMULATION_DIR + "/grid_error_heat_flow.pdf",
                bbox_extra_artists=(lg, st),
                bbox_inches='tight')
    plt.close()
