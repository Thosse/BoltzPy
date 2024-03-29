import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from plot_grid_error import maxwellian
from scipy.integrate import dblquad
from scipy.special import erf as sp_erf
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks


def integrate_over_domain(mean_velocity, temperature,
                          min_x, max_x,
                          min_y, max_y,
                          mass=1,
                          number_density=1):

    def f_temp(x,y):
        vel = np.array([x, y])
        maxw = maxwellian(vel,
                          number_density,
                          mean_velocity,
                          temperature,
                          mass,
                          False)
        mom_f = 0.5 * mass / number_density * np.sum((vel - mean_velocity)**2)
        return mom_f * maxw
    return dblquad(f_temp, min_x, max_x, min_y, max_y)[0]


def integrate_theory(mean_velocity, temperature,
                     min_x, max_x,
                     min_y, max_y,
                     mass=1,
                     number_density=1):
    max_dom = np.array([max_x, max_y], dtype=float)
    min_dom = np.array([min_x, min_y], dtype=float)
    temp_coefficient = 1 / np.sqrt(2 * temperature / mass)
    upper = (max_dom - mean_velocity) * temp_coefficient
    lower = (min_dom - mean_velocity) * temp_coefficient
    erf = 0.5 * (sp_erf(upper) - sp_erf(lower))
    factor_const = 0.5 * temperature / np.sqrt(np.pi)
    factor_erf = erf[::-1]
    factor_exp = upper * np.exp(-upper**2) - lower * np.exp(-lower**2)
    err_exp = np.sum(factor_const * factor_erf * factor_exp)
    err_erf = temperature * (1 - np.prod(erf))
    return err_erf + err_exp


# number of discretization points
N_PER_VEL = 20

MOMENTS = ["Temperature",
           # "Number Density",
           # "Momentum",
           # "Mean Velocity",
           # "Stress",
           # "Heat Flow"
           ]
#  Used Velocity Model
MASS = 1
NUMBER_DENSITY = 1.0
# setup velocity models
L0, L1 = 50, 51
MODELS = {s: bp.BaseModel([MASS],
                          [(s, s)],
                          1,
                          [2])
          for s in [7, L1]}


# This requires square / cubic grids!
DOMAINS = {s: {ext: (m.shapes[0, 0] + ext) * m.spacings[0] * m.base_delta * 0.5
               for ext in [-1, 0]}
           for s, m in MODELS.items()}

COLORS = {-1: "tab:red",
          0: "tab:blue",
          0.5: "tab:orange",
          2: "gray"}

# Each model has a different number of discretizations points
# These points must be subsets of each other (up to parity and domain)!
N = {m: (model.shapes[0, 0] - 1) * model.spacings[0] // 2 * N_PER_VEL + 1
     for m, model in MODELS.items()}


if __name__ == "__main__":
    # #################################################################################
    """ PLOT: Errors vs Temperature, for fixed shapes and mean velocity = 0
        This shows the discretisation and cutoff error"""
    #################################################################################
    fig, ax = plt.subplots(1, 2, constrained_layout=True,
                           figsize=(12.75, 6.25))
    # PARAMS
    TEMPERATURE = 2

    # store results in dict
    res_err = {m: {key: np.full(N[m], np.nan)
                   for key in MOMENTS}
               for m in MODELS.keys()}

    res_thr = {m: {key: {ext: np.full(N[m], np.nan)
                         for ext in DOMAINS[m].keys()}
                   for key in MOMENTS}
               for m in MODELS.keys()}

    for m, model in MODELS.items():
        VELS = np.zeros((N[m], 2), dtype=float)
        VELS[:, 0] = np.linspace(0, model.max_vel, N[m])
        for v, vel in enumerate(VELS):
            distr = maxwellian(model.vels,
                               NUMBER_DENSITY,
                               vel,
                               TEMPERATURE,
                               MASS,
                               # force_number_density=True
                               )
            # res1["Number Density"][m][v] = NUMBER_DENSITY - model.cmp_number_density(distr)
            # res1["Momentum"][m][v] = (MASS * NUMBER_DENSITY * vel[0]
            #                        - model.cmp_momentum(distr)[0])
            # res1["Mean Velocity"][m][v] = vel[0] - model.cmp_mean_velocity(distr)[0]
            res_err[m]["Temperature"][v] = TEMPERATURE - model.cmp_temperature(
                distr,
                # mean_velocity=vel
            )
            # res1["Stress"][m][v] = model.cmp_stress(distr)
            # res1["Heat Flow"][m][v] = model.cmp_heat_flow(distr)[0]
            if m in [L0, L1]:
                continue
            # fine_model = FMODS[m]
            # distr = maxwellian(fine_model.vels,
            #                    NUMBER_DENSITY,
            #                    vel,
            #                    temp,
            #                    MASS,
            #                    force_number_density=True
            #                    )
            # err1 = fine_model.cmp_temperature(distr)

            for ext, dom in DOMAINS[m].items():
                error_thr = integrate_theory(vel, TEMPERATURE, -dom, dom, -dom, dom)
                res_thr[m]["Temperature"][ext][v] = error_thr

    # plot errors
    ax[0].set_axisbelow(True)
    ax[0].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                     linewidth=0.4)
    ax[0].xaxis.grid(color='darkgray', linestyle='dashed', which="both",
                     linewidth=0.4)
    ax[0].set_xlabel(r"Mean Velocity Parameter $\widetilde{v}_x$",
                     fontsize=fs_label)
    ax[0].set_ylabel("Domain Error",
                     fontsize=fs_label)
    ax[0].tick_params(axis="both", labelsize=fs_ticks)
    ax[0].set_yscale("log")
    moment = "Temperature"
    for m, model in MODELS.items():
        if m in [L0, L1]:
            continue
        if m % 2 == 1:
            error = (res_err[m]["Temperature"] - res_err[L1]["Temperature"][: N[m]])
        else:
            error = (res_err[m]["Temperature"] - res_err[L0]["Temperature"][: N[m]])
        VELS = np.linspace(0, model.max_vel, N[m])

        # error = error / TEMPERATURE
        ax[0].plot(VELS,
                   error,
                   label="Measured Error",
                   color="black",
                   linewidth=3)

        for ext, dom in DOMAINS[m].items():
            error_thr = res_thr[m]["Temperature"][ext]
            # error_thr = error_thr / TEMPERATURE
            ax[0].plot(VELS,
                       error_thr,
                       label=r"$\mathfrak{v}_{max} = " + str(dom) + "$",
                       linestyle="--",
                       color=COLORS[ext],
                       linewidth=3)
    ax[0].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                     linewidth=0.4)
    ax[0].legend(loc="upper left",
                 fontsize=fs_legend,
                 ncol=1)
    ax[0].set_title("Measured and predicted domain errors",
                    # + r"for $\mathfrak{v}_{max} \in \{6, 7\}$ ",
                    fontsize=fs_title)

    #########################################
    #   RIGHT PLOT: Show DOMAIN CHOICES     #
    #########################################
    model = MODELS[7]
    # color background of the grid, do show the integration areas
    max_v = DOMAINS[7][0]
    ax[1].fill_between([-max_v, max_v], [-max_v, -max_v], max_v,
                       color="lightsteelblue",
                       zorder=-3)
    # plot lines to show integration areas of each point
    POS = np.unique(model.vels[...,0])
    POS = np.append(POS - 1, POS[-1] + 1)
    for p in POS:
        ax[1].plot([p, p], [max_v, -max_v], color="gray", linewidth=0.75)
        ax[1].plot([max_v, -max_v], [p, p], color="gray", linewidth=0.75)

    # plot grid points
    GRID = bp.Grid(model.shapes[0], model.base_delta, model.spacings[0], True)
    GRID.plot(plot_object=ax[1], alpha=0.9, color="tab:blue", s=75)

    # plot domains
    for ext, dom in DOMAINS[7].items():
        ax[1].plot([dom, dom, -dom, -dom, dom],
                   [dom, -dom, -dom, dom, dom],
                   color=COLORS[ext],
                   linestyle="--",
                   linewidth=3.0,
                   zorder=-2)

    ax[1].set_xlabel(r"Mean Velocity Parameter $\widetilde{v}_x$",
                     fontsize=fs_label)
    ax[1].set_ylabel(r"Mean Velocity Parameter $\widetilde{v}_y$",
                     fontsize=fs_label)
    ax[1].set_title(r"Grid domain $D\left(\mathfrak{V^s}\right)$ and integration domain",
                    fontsize=fs_title)
    ax[1].tick_params(axis="both", labelsize=fs_ticks)
    plt.tight_layout(pad=2)
    plt.savefig(bp.SIMULATION_DIR + "/domain_error.pdf")

    #############################################
    #   Assert the upper boundary conjecture    #
    #############################################
    model = MODELS[7]
    ext_model = MODELS[L1]
    dom = DOMAINS[7][0]
    test_temps = np.linspace(0.05, 10, 500)
    test_vels = np.zeros((N[7], 2), dtype=float)
    test_vels[:, 0] = np.linspace(0, model.max_vel, N[7])
    # test_vels[:, 1] = np.linspace(0, 2*model.max_vel, N[7])
    for T in test_temps:
        diff = np.zeros(test_vels.shape[0], dtype=float)
        for v, vel in enumerate(test_vels):
            distr = maxwellian(model.vels,
                               NUMBER_DENSITY,
                               vel,
                               T,
                               MASS,
                               )
            ext_distr = maxwellian(ext_model.vels,
                                   NUMBER_DENSITY,
                                   vel,
                                   T,
                                   MASS,
                                   )
            measured = model.cmp_temperature(distr) - ext_model.cmp_temperature(ext_distr)
            predicted = integrate_theory(vel, T, -dom, dom, -dom, dom)
            assert measured < 1e-15 or predicted < measured
            diff[v] = ((1e-15 + predicted) - measured) / T
        plt.plot(test_vels, diff, color="gray")
    plt.yscale("log")
    plt.xlabel("Mean Velocity", fontsize=12)
    plt.ylabel("Predicted Error - Measuered Error", fontsize=12)
    plt.title("Predicted domain error is an upper bound "
              "for the measured domain error", fontsize=14)
    plt.show()
    print("Upper bound confirmed for a chosen set of parameters")

