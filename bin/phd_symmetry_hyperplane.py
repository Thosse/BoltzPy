import boltzpy as bp
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks


for kind in ["Continuous", "Discrete"]:
    # setup plot states with inherent symmetries
    n_plots = 3
    xy_length = 35 if kind == "continuous" else 15
    values = np.empty((n_plots, xy_length ** 2), dtype=float)
    # use maxwellian state for left plot
    model = bp.BaseModel([1], [[xy_length, xy_length]])
    t_range = model.temperature_range()
    state = model.cmp_initial_state([1.0], [[0.0, 0.0]], t_range.dot([0.2, 0.3]))
    values[0] = state

    # use stress and heatflow moment functions * state for other plots
    mf_stress = model.mf_stress([[0, 0]])
    values[1] = state * mf_stress
    mf_heat = model.mf_heat_flow([[0, 0]], direction=[0, 1])
    values[2] = state * mf_heat

    values = values.reshape((n_plots, xy_length, xy_length))
    fig, axes = plt.subplots(nrows=1, ncols=n_plots,
                             figsize=(12.75, 5.05),
                             constrained_layout=True)

    images = []
    if kind == "Continuous":
        titles = ["Maxwellian $f(v)$",
                  "Stress State $\pi_{1,2}(v) f(v)$",
                  "Heat Flow State $\sigma_{1}(v) f(v)$"]
    else:
        titles = ["Maxwellian $\mathfrak{f} \in \mathbb{R}^n$",
                  "Stress State $\pi_{1,2} \mathfrak{f} \in \mathbb{R}^n$",
                  "Heat Flow State $\sigma_{1} \mathfrak{f} \in \mathbb{R}^n$"]
    interpolation = "quadric" if kind == "Continuous" else None
    for i_val, val in enumerate(values):
        # axes[i_val].set_axis_off()
        im = axes[i_val].imshow(val, cmap='coolwarm',
                                interpolation=interpolation,
                                origin="lower",
                                extent=[0, 4, 0, 4],
                                vmax=val.max() *1.2, vmin = -val.max() * 1.2)
        images.append(im)
        axes[i_val].tick_params(axis="both",
                                which="both",
                                labelbottom=False,
                                labeltop=False,
                                labelleft=False,
                                labelright=False,
                                labelsize=fs_ticks)
        axes[i_val].set_title(titles[i_val], fontsize=fs_title)
        if kind == "Continuous":
            axes[i_val].set_xlabel(r"$v_x$",
                                   fontsize=fs_label)
            axes[i_val].set_ylabel(r"$v_y$",
                                   fontsize=fs_label)
        else:
            axes[i_val].set_xlabel(r"$\mathfrak{v}_x$",
                                   fontsize=fs_label)
            axes[i_val].set_ylabel(r"$\mathfrak{v}_y$",
                                   fontsize=fs_label)

    plt.subplots_adjust(left=0.05,
                        bottom=-0.05,
                        right=1.1,
                        top=1.05,
                        wspace=0.2,
                        hspace=0.)
    # add colorbar
    cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.575)
    cbar.set_ticks([-values[-1].max(), 0,  values[-1].max()])
    cbar.set_ticklabels(['Low', 0, 'High'])
    cbar.ax.tick_params(axis="both", labelsize=fs_ticks)
    f_name = "$f$" if kind == "Continuous" else r"$\mathfrak{f}$"
    fig.suptitle("Symmetry and Antisymmetry Planes for Different "
                 + kind + " Functions",
                 fontsize=fs_suptitle)

    # add hyperplanes
    maxwellian_planes = [0, 1, 2, 3] if kind == "Continuous" else [0, 2]
    for x in maxwellian_planes:
        axes[0].plot([0, 4], [x, 4-x], "black", linestyle="dotted")
        axes[0].plot([4 - x, x], [0, 4], "black", linestyle="dotted")

    axes[1].plot([0, 4], [2, 2], "black", dashes=(8, 8), label="antisymmetry planes")
    axes[1].plot([2, 2], [0, 4], "black", dashes=(8, 8))
    axes[1].plot([0, 4], [0, 4], "black", linestyle="dotted", label="symmetry planes")
    axes[1].plot([0, 4], [4, 0], "black", linestyle="dotted")

    axes[2].plot([0, 4], [2, 2], "black", dashes=(8, 8))
    axes[2].plot([2, 2], [0, 4], "black", linestyle="dotted")

    lg = fig.legend(fontsize=fs_legend + 2,
                    loc="lower center",
                    ncol=2)

    plt.savefig(bp.SIMULATION_DIR + "/phd_" + kind + "_symmetry_axes.pdf")
