import numpy as np
import boltzpy as bp

import matplotlib.pyplot as plt
import h5py
from time import process_time
from exp_collision_generations import vectorized, group_distance
##############################
#   Generation Parameters    #
##############################
FILENAME = "/exp_collision_generation_monospecies.hdf5"
FILE = h5py.File(bp.SIMULATION_DIR + FILENAME, mode="r")
masses = [25]
mass_str = "{}".format(*masses)
SHAPES = {dim: np.array([np.full((1, dim), i, dtype=int)
                         for i in range(3, 51)])
          for dim in [2, 3]}
MAX_TIME = 3600
ALGORITHMS = ["vectorized",
              "group_distance"]

#################################
#   Generate Collision Times    #
#################################
if __name__ == "__main__":
    if mass_str in FILE.keys():
        compute = (input("Precomputed Solution detected for '{}' in"
                         "\n'{}'.\n"
                         "Do you want to recompute? (yes/no)"
                         "".format(mass_str, FILENAME)) == "yes")
        if compute:
            del FILE[mass_str]
    else:
        print("No precomputed solution detected for '{}'.\n"
              "Start Computing.".format(mass_str))
        compute = True

    if not compute:
        h5py_group = FILE[mass_str]
    else:
        FILE.create_group(mass_str)
        for dim in [2, 3]:
            h5py_group = FILE[mass_str].create_group(str(dim))
            h5py_group["SHAPES"] = SHAPES[dim]
            h5py_group["ncols"] = np.full(len(SHAPES[dim]), -1, dtype=int)
            SKIP = []
            for alg in ALGORITHMS:
                h5py_group.create_group(alg)
                h5py_group[alg]["total_time"] = np.full(len(SHAPES[dim]), np.nan, dtype=float)
                h5py_group[alg]["colvel_time"] = np.full(len(SHAPES[dim]), np.nan, dtype=float)
                h5py_group[alg]["shifting_time"] = np.full(len(SHAPES[dim]), np.nan, dtype=float)
                h5py_group[alg]["get_idx_time"] = np.full(len(SHAPES[dim]), np.nan, dtype=float)
                h5py_group[alg]["choice_time"] = np.full(len(SHAPES[dim]), np.nan, dtype=float)
                h5py_group[alg]["filter_time"] = np.full(len(SHAPES[dim]), np.nan, dtype=float)
            for i, shapes in enumerate(SHAPES[dim]):
                print("Computing shape ", shapes[0], "\t", FILENAME)
                model = bp.CollisionModel(masses, shapes,
                                          collision_relations=[],
                                          collision_weights=[],
                                          setup_collision_matrix=False)
                for a, alg in enumerate(ALGORITHMS):
                    if alg in SKIP:
                        continue
                    tic = process_time()
                    result = locals()[alg](model, h5py_group[alg], i)
                    toc = process_time()
                    if h5py_group["ncols"][i][()] != -1:
                        assert h5py_group["ncols"][i][()] == result.shape[0]
                    else:
                        h5py_group["ncols"][i] = result.shape[0]
                    FILE.flush()
                    del result
                    print("\t", alg, toc - tic)
                    # dont wait too long for algorithms
                    if toc - tic > MAX_TIME:
                        print("STOP ", alg, "\nat shape= ", shapes)
                        SKIP.append(alg)

    print("Results are:")
    for d in [str(2), str(3)]:
        for alg in ALGORITHMS:
            if alg not in FILE[mass_str][d].keys():
                continue
            print(alg, "dim = ", d)
            print("total:\n", FILE[mass_str][d][alg]["total_time"][()])
            print("colvel_time:\n", FILE[mass_str][d][alg]["colvel_time"][()])
            print("shifting_time:\n", FILE[mass_str][d][alg]["shifting_time"][()])
            print("\n")

    # ################
    # # create plots #
    # ################
    COLORS = {"four_loop": "tab:brown",
              "three_loop": "tab:pink",
              "vectorized": "tab:orange",
              "group_distance": "tab:red",
              "group_sorted_distance": "tab:green",
              "group_sorted_distance_no_cutoff": "tab:olive",
              "group_norm_and_sorted_distance": "tab:blue"}
    ALL_ALGS = [["vectorized", "group_distance"]]
    plt_beg = [[2, 0]]
    plt_spacing = [[10, 2]]

    for c, CUR_ALGS in enumerate(ALL_ALGS):
        # setup plot
        fig, ax = plt.subplots(1, 2, constrained_layout=True)
        print("Algorithms = ", CUR_ALGS)
        # get max index (shape) that was computed
        max_idx = np.full(2, 0, dtype=int)
        for d, dim in enumerate([str(2),str(3)]):
            for alg in CUR_ALGS:
                res = FILE[mass_str][dim][alg]["total_time"][()]
                res = np.where(np.isnan(res), -1, res)
                max_idx[d] = np.max([max_idx[d], res.argmax() + 1])
        print("max index = ", max_idx)
        # get x positions of plot from max_idx
        x_vals = [np.arange(max_idx[d]) for d in [0, 1]]

        # plot different algorithm times
        for d, dim in enumerate([str(2), str(3)]):
            for a, alg in enumerate(CUR_ALGS):
                label = alg
                res = FILE[mass_str][dim][alg]["total_time"][:max_idx[d]]
                # widths = np.linspace(-0.5, 0.5, len(CUR_ALGS) + 2)
                # ax[d].bar(x_vals[d] + widths[a+1], res, color=COLORS[alg],
                #           width=widths[1] - widths[0], label=label,)
                ax[d].plot(x_vals[d], res, "-o",
                           color=COLORS[alg], label=label)

            ax[d].set_xlabel("Grid Shapes".format(dim))
            ax[d].set_axisbelow(True)
            ax[d].yaxis.grid(color='darkgray', linestyle='dashed')
            ax[d].xaxis.grid(color='darkgray', linestyle='dashed')

            ax[d].set_xticks(x_vals[d][plt_beg[c][d]::plt_spacing[c][d]])
            ax[d].set_xticklabels(SHAPES[d + 2][:max_idx[d], 0][plt_beg[c][d]::plt_spacing[c][d]])

        fig.suptitle("Collision Generation Time "
                     "of Non-Mixture for Different Grid Shapes".format(masses))
        ax[0].legend(title="Algorithms:", loc="upper left")
        ax[0].set_ylabel("Computation Time In Seconds")
        # plt.tight_layout()
        plt.show()
        del fig, ax
