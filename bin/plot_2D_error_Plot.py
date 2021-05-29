import boltzpy as bp
import numpy as np
import matplotlib
import matplotlib.colors as colors
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
MODEL = {31: bp.BaseModel([MASS], [(31, 31)], 1, [2]),
         7: bp.BaseModel([MASS], [(7, 7)], 1, [2])}

# number of discretization points
N = MAX_VEL * 31 + 1

STYLE = ["-", "--", ":", "-.", ":"]


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

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


cm_cw = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.coolwarm)
cm_bin = cmap_map(lambda x: x / 4, matplotlib.cm.binary)


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 3, constrained_layout=True,
                           figsize=(12.75, 6.25))

    #################################################################
    # LEFT PLOT: Heatmap of DOMAIN ERROR for v_x versus Temperature #
    #################################################################
    TEMPERATURES = np.linspace(0, MAX_T, N)[1:]
    DIRECTION = np.array([1,0])
    DIRECTION = DIRECTION / np.max(DIRECTION)
    VELS = np.zeros((N, 2), dtype=float)
    VELS[:] = np.linspace(0, MAX_VEL, N)[:, None] * DIRECTION[None, :]

    # in heatmaps it is ordered (y,x), for some reason
    res1 = np.full((3, TEMPERATURES.size, VELS.shape[0]), np.nan, dtype=float)

    loc_res = {7: 0,
               31:0}
    for t, T in enumerate(TEMPERATURES):
        for v, vel in enumerate(VELS):
            try:
                for m in MODEL.keys():
                    distr = maxwellian(MODEL[m].vels,
                                       NUMBER_DENSITY,
                                       vel,
                                       T,
                                       MASS, #force_number_density=False
                                       )
                    loc_res[m] = T - MODEL[m].cmp_temperature(
                        distr,
                        number_density=1.0,
                        mean_velocity=vel
                    )
                    # loc_res[m] = 1 - MODEL[m].cmp_number_density(
                    #     distr
                    # )
                    # loc_res[m] = MODEL[m].cmp_heat_flow(distr, mean_velocity=vel)[0]
            except ValueError:
                print("Error at v=", vel, " T=", T)
                res1[t, v] = 1
                continue
            res1[0, t, v] = loc_res[7]
            res1[1, t, v] = loc_res[31]
            res1[2, t, v] = loc_res[7] - loc_res[31]
            # res1[:, t, v] = res1[:, t, v] / T
    res1 = np.abs(res1)
    print(res1.min(), res1.max())
    MIN_VAL = -15
    MAX_VAL = 1
    res1 = np.where(res1 < np.exp(MIN_VAL), np.exp(MIN_VAL), res1)
    levels = [10**z
              for z in range(-3, 2)]

    # plot error heatmap
    for a in [0, 1, 2]:
        # hm = ax.pcolor(VELS.max(axis=-1), TEMPERATURES, res1,
        #                norm=colors.LogNorm(vmin=res1.min(), vmax=res1.max()),
        #                cmap=cm_cw,
        #                )
        hm = ax[a].imshow(res1[a], origin="lower",
                       norm=colors.LogNorm(vmin=0.001, vmax=1.0),
                       extent=(0, MAX_VEL, 0, MAX_T),
                       interpolation="bilinear",
                       cmap=cm_cw,
                       )

        CS = ax[a].contour(VELS.max(axis=-1), TEMPERATURES, res1[a], levels=levels,
                           norm=colors.LogNorm(vmin=res1.min(), vmax=res1.max()),
                           cmap=cm_bin, linewidth=0.5)
        ax[a].clabel(CS, inline=True, fontsize=10, fmt="%1.3f")

    fig.colorbar(hm, ax=ax, orientation='vertical',
                 )

    #
    # # add colorbar (requires a separate ax, that is added by relative positions)
    # cax = fig.add_axes([0.04, 0.1, 0.27, 0.025])
    # plt.colorbar(hm, cax=cax, orientation='horizontal')
    plt.show()

