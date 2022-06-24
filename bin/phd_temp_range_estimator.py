import numpy as np

import h5py
import boltzpy as bp
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
mpl.rcParams['legend.title_fontsize'] = 14
import matplotlib.pyplot as plt
from fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

N_MEANS_VELS = 500
N_TEMPS = 500

model = bp.BaseModel([1],
                     [(7, 7)],
                     1,
                     [2]
                     )
assert model.nspc == 1, "Only use single species models here, requires adjustments otherwise"
MODEL_NAME = str(model.shapes[0])
FILENAME = "/phd_temp_range" + MODEL_NAME
FILE = h5py.File(bp.SIMULATION_DIR + FILENAME + ".hdf5", mode="a")

minmax_vel = np.min(model.max_i_vels[0])
MEAN_VELS = np.linspace(0, minmax_vel, N_MEANS_VELS)
# ignore temperature==0
MAX_TEMP = 1.2 * model.temperature_range(0)[1]
TEMPS = np.linspace(0, MAX_TEMP, N_TEMPS + 1)[1:]
ERR_ATOL = 0.1
COMPUTE = {"all": False,
           }
# delete results that are overwritten
for key, val in COMPUTE.items():
    if val and key in FILE.keys():
        del FILE[key]

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return mpl.colors.LinearSegmentedColormap('colormap',cdict,1024)


light_cw = cmap_map(lambda x: 1.5*x  , mpl.cm.RdYlGn)


print("#"*80)
print("\tCompute Heuristic and check where newton succeeds/fails")
print("#"*80)
key = "all"
if key not in FILE.keys():
    h5py_group = FILE.create_group(key)
    # create and save velocities to test
    success = []
    fail = []
    # 1 if initialization by newton succeeds, 0 else
    newton = np.zeros((N_MEANS_VELS, N_TEMPS), dtype=int)
    # temperature range, given by the heuristic
    heuristic = np.zeros((N_MEANS_VELS, 2))
    error_based = np.zeros((N_MEANS_VELS, 2))
    # initialization params, must be given as arrays
    number_dens = np.array([1.0])
    mean_vel = np.zeros((1,model.ndim))
    temp = np.array([0.0])
    for i, mv in enumerate(MEAN_VELS):
        print(i / N_MEANS_VELS, end="\r")
        mean_vel[0, 0] = mv
        heuristic[i] = model.temperature_range(mean_vel)
        try:
            error_based[i] = model.temperature_range(mv, atol=ERR_ATOL)
        except ValueError:
            error_based[i] = [-1, -1]
        for j, t in enumerate(TEMPS):
            temp[0] = t
            try:
                state = model.cmp_initial_state(number_dens,
                                                mean_vel,
                                                temp)
                assert np.allclose(model.cmp_number_density(state), number_dens)
                assert np.allclose(model.cmp_mean_velocity(state), mean_vel)
                assert np.allclose(model.cmp_temperature(state), temp)
                success.append([mv, t])
                newton[i, j] = 1
            except ValueError:
                fail.append([mv, t])
            except AssertionError:
                fail.append([mv, t])
            except RuntimeError:
                fail.append([mv, t])
    h5py_group["success"] = np.array(success)
    h5py_group["fail"] = np.array(fail)
    h5py_group["heuristic"] = heuristic
    h5py_group["newton"] = newton
    h5py_group["error_based"] = error_based
    FILE.flush()

    # compute grid errors
    grid_error = np.zeros((N_MEANS_VELS, N_TEMPS), dtype=float)
    state = np.zeros(model.nvels, dtype=float)
    # mean_vel = np.array([[0.0, 0.0]])
    for i, mv in enumerate(MEAN_VELS):
        # mean_vel[0, 0] = mv
        print(i / N_MEANS_VELS, end="\r")
        for j, t in enumerate(TEMPS):
            # compute grid error
            try:
                for s in model.species:
                    s_range = model.idx_range(s)
                    maxw = model.maxwellian(model.vels[s_range],
                                            mean_velocity=mv,
                                            temperature=t,
                                            mass=model.masses[s])
                    nd = model.cmp_number_density(maxw, s=s)
                    maxw = maxw / nd
                    state[s_range] = maxw
                grid_error[i, j] = np.abs(t - model.cmp_temperature(state))
            except ValueError:
                grid_error[i, j] = -100
    h5py_group["grid_error"] = grid_error
    FILE.flush()

print("#"*80)
print("\tPlot Heuristic Effectiveness")
print("#"*80)
h5py_group = FILE["all"]

fig = plt.figure(constrained_layout=True, figsize=(12.75, 8))
ax = fig.add_subplot()

res1 = np.array(h5py_group["newton"][()]).transpose()
hm = ax.imshow(res1, cmap=light_cw, interpolation='spline16',
               extent=(0, MEAN_VELS[-1], 0, TEMPS[-1]),
               origin="lower",
               vmin=-0.1, vmax=1.1
               )
# plot estimation area, by drawing the boundary
est_min = h5py_group["heuristic"][:, 0]
est_max = h5py_group["heuristic"][:, 1]
border = np.zeros((3 * N_MEANS_VELS, 2))
# mean velocity values
border[:, 0] = np.concatenate((MEAN_VELS, MEAN_VELS[::-1], MEAN_VELS))
# temperature values
border[:, 1] = np.concatenate((est_min, est_max[::-1], est_min))
ax.plot(*(border.transpose()), c="black", linewidth=5,
        label="Heuristic Prediction")
# add colors to legend
ax.scatter([],[], color=light_cw(900), label="Initialization Success", s=200)
ax.scatter([],[], color=light_cw(100), label="Initialization Fail", s=200)
ax.set_aspect('auto')
ax.set_title("Heuristic Prediction of Initialization Success",
             fontsize=fs_suptitle + 4)
ax.set_xlabel(r"Mean Velocity Parameter $\widetilde{v}_x$", fontsize=fs_label)
ax.set_ylabel(r"Temperature Parameter $\vartheta$", fontsize=fs_label + 2)
ax.tick_params(axis="both", labelsize=fs_ticks + 2)
ax.legend(loc="upper right", fontsize=fs_legend + 8)
if np.all(model.shapes == 7):
    ax.set_ylim(None, 17)
else:
    ax.set_title(r"Heuristic Prediction for a $"
                 + str(tuple(model.shapes[0]))
                 + r"$ shaped Grid",
                 fontsize=fs_title)
plt.savefig(bp.SIMULATION_DIR
            + "/phd_heuristic_newton"
            + str(model.shapes[0])
            + ".pdf")

##########################################################################################
print("#"*80)
print("\tPlot Heuristic Effectiveness for several grids")
print("#"*80)
##########################################################################################
print("Read previously computed results from file.\\"
      "Note: You must manually set the model shapes to the plotted values"
      "and create these results beforehand")

fig, axes = plt.subplots(3, 2,
                         figsize=(12.75, 17.75))
flaxes = axes.reshape(-1)
for i_shp, shape in enumerate([[3, 3], [501, 501],
                        [11, 11, 11], [12, 12, 12],
                        [5, 15], [15, 5]]):
    shape = np.array(shape, dtype=int)
    ax = flaxes[i_shp]
    SUB_FILENAME = "/phd_temp_range" + str(shape)
    SUB_FILE = h5py.File(bp.SIMULATION_DIR + SUB_FILENAME + ".hdf5", mode="r")
    h5py_group = SUB_FILE["all"]
    submodel = bp.BaseModel([1],
                            [shape],
                            1,
                            [2]
                            )
    sub_minmax_vel = np.min(submodel.max_i_vels[0])
    sub_MEAN_VELS = np.linspace(0, sub_minmax_vel, N_MEANS_VELS)
    # ignore temperature==0
    sub_MAX_TEMP = 1.2 * submodel.temperature_range(0)[1]
    sub_TEMPS = np.linspace(0, sub_MAX_TEMP, N_TEMPS + 1)[1:]
    res1 = np.array(h5py_group["newton"][()]).transpose()
    hm = ax.imshow(res1, cmap=light_cw, interpolation='spline16',
                   extent=(0, sub_MEAN_VELS[-1], 0, sub_TEMPS[-1]),
                   origin="lower",
                   vmin=-0.1, vmax=1.1
                   )
    # plot estimation area, by drawing the boundary
    est_min = h5py_group["heuristic"][:, 0]
    est_max = h5py_group["heuristic"][:, 1]
    border = np.zeros((3 * N_MEANS_VELS, 2))
    # mean velocity values
    border[:, 0] = np.concatenate((sub_MEAN_VELS, sub_MEAN_VELS[::-1], sub_MEAN_VELS))
    # temperature values
    border[:, 1] = np.concatenate((est_min, est_max[::-1], est_min))
    ax.plot(*(border.transpose()), c="black", linewidth=5,
            label="Heuristic Prediction")
    # add colors to legend
    ax.scatter([],[], color=light_cw(900), label="Initialization Success", s=200)
    ax.scatter([],[], color=light_cw(100), label="Initialization Fail", s=200)
    ax.set_aspect('auto')
    ax.set_title("Heuristic Prediction of Initialization Success",
                 fontsize=fs_title)
    ax.set_xlabel(r"Mean Velocity Parameter $\widetilde{v}_x$", fontsize=fs_label + 2)
    ax.set_ylabel(r"Temperature Parameter $\vartheta$", fontsize=fs_label + 2)
    ax.tick_params(axis="both", labelsize=fs_ticks + 2)
    if i_shp == 1:
        ax.legend(loc="upper right", fontsize=fs_legend + 4)

    ax.set_title(r"Grid Shape: $"
                 + str(list(shape))
                 + r"$",
                 fontsize=fs_title)

fig.suptitle("Heuristic Results and Matching Maxwellians for Different Grid Shapes",
             fontsize=fs_suptitle + 4)
plt.tight_layout(pad=2)
plt.savefig(bp.SIMULATION_DIR
            + "/phd_heuristic_newton"
            + " all grids"
            + ".pdf")

##########################################################################################
print("#"*80)
print("\tCompute where grid error metric is bad, in heuristic range")
print("#"*80)
##########################################################################################

fig, ax = plt.subplots(figsize=(12.75, 6.25))
# plot error heatmap
grid_error = FILE["all"]["grid_error"][()]
LEVELS = [0.01,
          # 0.025,
          # 0.05,
          # 0.075,
          0.1,
          # 0.2,
          1.0,
          # 2.5,
          # 5.0
          ]

heatmap = ax.imshow(grid_error.transpose(),
                    cmap=mpl.cm.coolwarm,
                    interpolation='spline16',
                    extent=(0, model.max_vel, 0, MAX_TEMP),
                    aspect=0.46,
                    origin="lower",
                    # vmin=0,
                    # vmax=1.0,
                    norm=mpl.colors.LogNorm(vmin=0.001, vmax=10)
                   )

# add colorbar (requires a separate ax, that is added by relative positions)
# cax = fig.add_axes([0.04, 0.1, 0.27, 0.025])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(heatmap,
                    # cax=cax,
                    orientation='vertical',
                    # fraction=0.05
                    )

# plot heuristic estimation area, by drawing the boundary
est_min = FILE["all"]["heuristic"][:, 0]
est_max = FILE["all"]["heuristic"][:, 1]
border = np.zeros((3 * N_MEANS_VELS, 2))
# mean velocity values
border[:, 0] = np.concatenate((MEAN_VELS, MEAN_VELS[::-1], MEAN_VELS))
# temperature values
# border[:, 1] = np.concatenate((est_min, est_max[::-1], est_min))
# ax.plot(*(border.transpose()), c="black", linewidth=1,
#         linestyle="solid",
#         label="Heuristic Prediction")

# Add height lines
ax.contour(grid_error.transpose(),
           colors='black',
           extent=(0, model.max_vel, 0, MAX_TEMP),
           origin="lower",
           levels=LEVELS,
           linewidths=3,
           linestyles="dotted",
           alpha=0.9)

ax.set_title(r"Temperature-Error Based Parameter Range",
             fontsize=fs_title)

ax.set_xlabel(r"Mean Velocity Parameter $\widetilde{v}_x$, "
              r"with $\widetilde{v}_y=0$", fontsize=fs_label)
ax.set_ylabel(r"Temperature Parameter $\vartheta$", fontsize=fs_label)
cbar.ax.set_ylabel("Grid Error", fontsize=fs_label)
cbar.ax.yaxis.set_label_position('left')
ax.tick_params(axis="both", labelsize=fs_ticks)
cbar.ax.tick_params(axis="both", labelsize=fs_ticks)
# cbar.ax.set_yticklabels(["0", "0.2", "0.4", "0.6","0.8", r"$\geq 1$"])
ax.set_ylim(None, 9)

plt.savefig(bp.SIMULATION_DIR + "/phd_temp_range_grid_error.pdf")

##########################################################################################
print("#"*80)
print("\tCompute where grid error metric is bad, in heuristic range")
print("#"*80)
newton_success = FILE["all"]["success"][()]
param_order = np.argsort(newton_success[:, 0])[::-1][:19]
bad_params = newton_success[param_order]
sec_ordering = np.argsort(bad_params[:, 1])
bad_params = bad_params[sec_ordering]
used_params = bad_params[-1]
print(used_params)
state = model.cmp_initial_state([1],
                                [[used_params[0], 0]],
                                [used_params[1]])

fig = bp.Plot.AnimatedFigure(figsize=(7.0, 6.25), dpi=None)
ax = fig.add_subplot((1, 1, 1), dim=3)
vels = model.vels
ax.plot(vels[..., 0], vels[..., 1], state)
ax.mpl_axes.view_init(elev=25.1, azim=250)
ax.mpl_axes.tick_params(axis="both", labelsize=fs_ticks)
# ax.mpl_axes.set_title("A Deformed Maxwellian with "
#                       r"$\overline{v} = 5.988$",
#                       fontsize=fs_title)
fig._plt.tight_layout(pad=4)
# add + 0.0 to remove "dark spot in 3d plot.... who knows why this effects anything :)
ax.mpl_axes.view_init(elev=25 + 0.0, azim=250)
plt.savefig(bp.SIMULATION_DIR + "/phd_deformed_maxwellian.png")
