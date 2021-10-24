import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
matplotlib.rcParams['legend.title_fontsize'] = 14
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks, fs_legend_title

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

#  Used Velocity Model
MASSES = [2 ,3]
NUMBER_DENSITY = 1.0
MAX_T = 8
# setup velocity models
model = bp.BaseModel(MASSES,
                     [(5, 5), (7, 7)],
                     base_delta=1/3)

mv = [0, 0]
print("COmpute optimal temperature range")
print("Heuristical Optimum = ", model._temperature_range_mixture_heuristic(mean_velocity=mv))
ERROR_MARGIN = 0.1
OPTIMAL_T_RANGE = model.temperature_range(mean_velocity=mv, atol=ERROR_MARGIN)
print("Error Based Optimum = ", OPTIMAL_T_RANGE)
BASE_THETA = 2
print("Theta used = ", BASE_THETA)

def single_species_maxwellian(velocities,
               number_density,
               mean_velocity,
               temperature,
               mass):
    dim = velocities.shape[-1]
    # compute exponential with matching mean velocity and temperature
    distance = np.sum((velocities - mean_velocity) ** 2, axis=-1)
    exponential = np.exp(-0.5 * mass / temperature * distance)
    # force number density normalization
    dv = np.max(np.abs(velocities[0] - velocities[1]))
    divisor = np.sum(exponential * dv**dim)
    result = number_density * exponential / divisor
    return result


def maxwellian(model,
               mean_velocity,
               temperature):
    assert isinstance(model, bp.BaseModel)
    state = np.zeros(model.nvels, dtype=float)
    for s in model.species:
        vels = model.vels[model.idx_range(s)]
        state[model.idx_range(s)] = single_species_maxwellian(
            vels,
            1.0,
            mean_velocity,
            temperature,
            model.masses[s]
        )
    return state


if __name__ == "__main__":
    print("################################################\n"
          "PLOT: Moment Differences of Maxwellian-Mixtures\n"
          "################################################")
    fig, ax = plt.subplots(2, 2,
                           # sharex="col", sharey="row",
                           figsize=(12.75, 8.))
    fig.suptitle(r"Specific Moment Differences for Maxwellians of Mixtures",
                 fontsize=fs_suptitle)
    print("Compute temperature parameter effects for centered mixtures")
    THETA = np.linspace(0.1, MAX_T, N)
    MEAN_VEL_PARAM = np.zeros(2)
    Temp_T = np.zeros((model.nspc, N))

    for i_t, theta in enumerate(THETA):
        maxw = maxwellian(model, MEAN_VEL_PARAM, theta)
        for s in model.species:
            Temp_T[s, i_t] = theta - model.cmp_temperature(maxw, s)

    MVel_T = np.zeros((model.nspc, N))
    MV_OFFSET = np.array([0.5, 0])
    for i_t, theta in enumerate(THETA):
        maxw = maxwellian(model, MEAN_VEL_PARAM + MV_OFFSET, theta)
        for s in model.species:
            MVel_T[s, i_t] = MV_OFFSET[0] - model.cmp_mean_velocity(maxw, s)[0]

    print("Compute temperature parameter effects for centered mixtures")
    V_VELS = np.zeros((N, 2))
    V_VELS[:, 0] = np.linspace(0, model.max_vel, N)
    V_VELS[:, 1] = np.linspace(0, model.max_vel, N)

    Temp_mv = np.zeros((model.nspc, N))
    for i_v, v in enumerate(V_VELS):
        maxw = maxwellian(model, v, BASE_THETA)
        for s in model.species:
            Temp_mv[s, i_v] = BASE_THETA - model.cmp_temperature(maxw, s)

    MVel_mv = np.zeros((model.nspc, N))
    for i_v, v in enumerate(V_VELS):
        maxw = maxwellian(model, v, BASE_THETA)
        for s in model.species:
            MVel_mv[s, i_v] = v[0] - model.cmp_mean_velocity(maxw, s)[0]

    print("Plot results in 2x2 Plot")
    colors = ["tab:blue", "tab:orange"]
    labels = [r"$s = 1$", r"$s = 2$"]
    styles = ["solid", "dashed"]
    for s in model.species:
        ax[1, 0].plot(THETA,
                      Temp_T[s],
                      # label=model.shapes[0],
                      linestyle=styles[s],
                      color=colors[s],
                      linewidth=3
                  )
        ax[1, 0].set_title(r"Specific Temperatures at $\widetilde{v} = (0, 0)$",
                          fontsize=fs_title)
        ax[1, 0].set_xlabel(r"Temperature Parameter $\vartheta$",
                            fontsize=fs_label)
        ax[1, 0].set_ylabel(r"$\vartheta - T^s$",
                            fontsize=fs_label)

        ax[0, 0].plot(THETA,
                      MVel_T[s],
                      label=labels[s],
                      linestyle=styles[s],
                      color=colors[s],
                      linewidth=3
                      )
        ax[0, 0].set_title(r"Specific Mean Velocities at $\widetilde{v} = "
                           + str(tuple(MV_OFFSET)) + "$",
                           fontsize=fs_title)
        ax[0, 0].set_xlabel(r"Temperature Parameter $\vartheta$",
                            fontsize=fs_label)
        ax[0, 0].set_ylabel(r"$\widetilde{v} - \overline{v}^s$",
                            fontsize=fs_label)
        ax[0, 0].legend(loc="upper right", title="Species $s \in \mathfrak{S}$",
                        fontsize=fs_legend,
                        title_fontsize=fs_legend_title)
        ax[1, 1].plot(V_VELS[:, 0],
                      Temp_mv[s, :],
                      # label=model.shapes[0],
                      linestyle=styles[s],
                      color=colors[s],
                      linewidth=3
                      )

        ax[1,1].set_title(r"Specific Temperatures at $\vartheta = %3.1f$" % BASE_THETA,
                          fontsize=fs_title)
        ax[1,1].set_xlabel(r"Velocity Parameter $\widetilde{v}$",
                           fontsize=fs_label)
        ax[1, 1].set_ylabel(r"$\vartheta - T^s$",
                            fontsize=fs_label)

        ax[0, 1].plot(V_VELS[:, 0],
                      MVel_mv[s, :],
                      # label=model.shapes[0],
                      linestyle=styles[s],
                      color=colors[s],
                      linewidth=3
                      )
        ax[0, 1].set_title(r"Specific Mean Velocities at $\vartheta = %3.1f$" % BASE_THETA,
                           fontsize=fs_title)
        ax[0, 1].set_xlabel(r"Velocity Parameter $\widetilde{v}$",
                            fontsize=fs_label)
        ax[0, 1].set_ylabel(r"$\widetilde{v} - \overline{v}^s$",
                            fontsize=fs_label)
    for a in [(0,0), (0,1), (1,0), (1,1)]:
        ax[a].tick_params(axis="both", labelsize=fs_ticks)
        ax[a].yaxis.grid(color='darkgray', linestyle='dashed', which="both",
                         linewidth=0.4)
        ax[a].xaxis.grid(color='darkgray', linestyle='dashed', which="both",
                         linewidth=0.4)
    plt.tight_layout(pad=2)
    plt.subplots_adjust(top=0.88)
    plt.savefig(bp.SIMULATION_DIR + "/grid_err_mixture.pdf")
