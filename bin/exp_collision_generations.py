import numpy as np
import boltzpy as bp

import matplotlib.pyplot as plt
import h5py
from time import process_time

##############################
#   Generation Parameters    #
##############################
FILENAME = "/exp_collision_generation.hdf5"
FILE = h5py.File(bp.SIMULATION_DIR + FILENAME, mode="a")
masses = [5, 7]
mass_str = "({},{})".format(*masses)
SHAPES = {dim: np.array([np.full((2, dim), i, dtype=int)
                         for i in range(3, 51)])
          for dim in [2, 3]}
MAX_TIME = 6
ALGORITHMS = ["four_loop", "three_loop", "vectorized",
              "group_distance", "group_sorted_distance",
              "group_norm_and_sorted_distance",
              "group_distance_no_cutoff",
              "group_sorted_distance_no_cutoff"]

"""This Plot has issues with exporting it as eps.   
I don't know why, but its not worth solving.
Use png as format!"""


#######################################
#   Collision Generation Functions    #
#######################################
def four_loop(self, group=None, idx=None):
    tic = process_time()
    assert isinstance(self, bp.CollisionModel)
    relations = []
    # Iterate over Specimen pairs
    colvel = np.zeros((4, self.ndim), dtype=int)
    idx_range = np.zeros((2, 2), dtype=int)
    for s0 in self.species:
        idx_range[0] = self._idx_offset[[s0, s0+1]]
        for s1 in np.arange(s0, self.nspc):
            idx_range[1] = self._idx_offset[[s1, s1+1]]
            masses = self.masses[[s0, s0, s1, s1]]
            for i0 in np.arange(*idx_range[0]):
                colvel[0] = self.i_vels[i0]
                # for i1 in np.arange(*idx_range[0]):
                for i1 in np.arange(i0+1, idx_range[0, 1]):
                    colvel[1] = self.i_vels[i1]
                    for i2 in np.arange(*idx_range[1]):
                        colvel[2] = self.i_vels[i2]
                        for i3 in np.arange(*idx_range[1]):
                            colvel[3] = self.i_vels[i3]
                            # check if its a proper Collision
                            if self.is_collision(colvel.reshape((1, 4, self.ndim)), masses):
                                if i0 == i3 and i1 == i2:
                                    continue
                                relations.append([i0, i1, i2, i3])
                            else:
                                continue
    toc = process_time()
    group["total_time"][idx] = toc - tic
    relations = np.array(relations, dtype=int, ndmin=2)
    # remove redundant collisions
    relations = self.filter(self.key_index(relations), relations)
    # sort collisions for better comparability
    relations = self.sort(self.key_index(relations), relations)
    return relations


def three_loop(self, group=None, idx=None):
    tic = process_time()
    assert isinstance(self, bp.CollisionModel)
    relations = []
    # Iterate over Specimen pairs
    colvel = np.zeros((4, self.ndim), dtype=int)
    idx_range = np.zeros((2, 2), dtype=int)
    for s0 in self.species:
        idx_range[0] = self._idx_offset[[s0, s0+1]]
        for s1 in np.arange(s0, self.nspc):
            idx_range[1] = self._idx_offset[[s1, s1+1]]
            masses = self.masses[[s0, s0, s1, s1]]
            grid1 = self.subgrids(s1)
            for i0 in np.arange(*idx_range[0]):
                colvel[0] = self.i_vels[i0]
                for i1 in np.arange(i0+1, idx_range[0, 1]):
                    colvel[1] = self.i_vels[i1]
                    diff_v = colvel[1] - colvel[0]
                    for i2 in np.arange(*idx_range[1]):
                        colvel[2] = self.i_vels[i2]
                        # Calculate w1, using the momentum invariance
                        assert all((diff_v * masses[0]) % masses[2] == 0)
                        diff_w = -diff_v * masses[0] // masses[2]
                        w1 = colvel[2] + diff_w
                        if w1 not in grid1:
                            continue
                        colvel[3] = w1
                        # check if its a proper Collision
                        if self.is_collision(colvel.reshape((1, 4, self.ndim)), masses):
                            i3 = self.get_idx(s1, colvel[3])
                            if i0 == i3 and i1 == i2:
                                continue
                            relations.append([i0, i1, i2, i3])
                        else:
                            continue
    toc = process_time()
    group["total_time"][idx] = toc - tic
    relations = np.array(relations, dtype=int, ndmin=2)
    # remove redundant collisions
    relations = self.filter(self.key_index(relations), relations)
    # sort collisions for better comparability
    relations = self.sort(self.key_index(relations), relations)
    return relations


def cmp_relations(self, group_by, cufoff=True, group=None, idx=None):
    assert group_by in {"distance",
                        "sorted_distance",
                        "None",
                        "norm_and_sorted_distance"}
    t_get_colvels = 0.0
    t_shifting = 0.0
    t_get_idx = 0.0
    t_choice = 0.0
    t_partition = 0.0
    t0_total = process_time()
    # collect collisions in a lists
    relations = []
    # Collisions are computed for every pair of specimen
    species_pairs = np.array([(s0, s1) for s0 in self.species
                              for s1 in range(s0, self.nspc)])
    # pre initialize each species' velocity grid
    grids = self.subgrids()
    # generate symmetry matrices, for partitioned_distances
    sym_mat = self.symmetry_matrices

    # Compute collisions iteratively, for each pair
    for s0, s1 in species_pairs:
        masses = self.masses[[s0, s1]]

        # partition velocities
        t0_partition = process_time()
        if group_by == "None":
            # imitate grouping, to fit into the remaining algorithm
            grp = grids[s0].iG[:, np.newaxis]
            # no symmetries are used
            grp_sym = None
            # no extended grids necessary
            extended_grids = np.array([grids[s0], grids[s1]], dtype=object)
            # no cutoff with max_distances necessary
            max_distance = None
        elif group_by == "norm_and_sorted_distance":
            # group based on sorted velocity components and sorted distance
            sort_dist = grids[s1].key_distance(grids[s0].iG)[..., :-1]
            sort_vel = np.sort(np.abs(grids[s0].iG), axis=-1)
            sym_vel = grids[s1].key_symmetry_group(grids[s0].iG)
            # group both grp and grp_sym in one go (and with same order)
            grp, grp_sym = self.group((sort_vel, sort_dist),
                                      (grids[s0].iG, sym_vel),
                                      as_dict=False)
            del sort_dist, sort_vel, sym_vel
            # no extended grids necessary
            extended_grids = np.array([grids[s0], grids[s1]], dtype=object)
            # no cutoff with max_distances necessary
            max_distance = None
        # partition grids[s0] by distance
        # if s0 == s1, this is equivalent to partitioned distance (but faster)
        elif group_by == "distance" or s0 == s1:
            # partition based on distance to next grid point
            dist = grids[s1].key_distance(grids[s0].iG)
            norm = bp.Grid.key_norm(grids[s0].iG)
            grp = self.group(dist, grids[s0].iG, as_dict=False, sort_key=norm)
            del norm, dist
            # no symmetries are used
            grp_sym = None
            # compute representative colliding velocities in extended grids
            extended_grids = self._get_extended_grids((s0, s1), grp)
            # cutoff unnecessary velocity with max_distances,
            # when computing the reference colvels
            if cufoff:
                max_distance = self.max_i_vels[None, s0] + self.max_i_vels[[s0, s1]]
            else:
                max_distance = None
        # partition grids[s0] by distance and rotation
        elif group_by == "sorted_distance":
            # group based on distances, rotated into 0 <= x <= y <= z
            sort_dist = grids[s1].key_sorted_distance(grids[s0].iG)
            norm = bp.Grid.key_norm(grids[s0].iG)
            # group both grp and grp_sym in one go (and with same order)
            grp, grp_sym = self.group(sort_dist[..., :-1],
                                      (grids[s0].iG, sort_dist[:, -1]),
                                      as_dict=False, sort_key=norm)
            del norm, sort_dist
            # compute representative colliding velocities in extended grids
            extended_grids = self._get_extended_grids((s0, s1), grp)
            # cutoff unnecessary velocity with max_distances,
            # when computing the reference colvels
            if cufoff:
                max_distance = self.max_i_vels[None, s0] + self.max_i_vels[[s0, s1]]
            else:
                max_distance = None
        else:
            raise ValueError
        t_partition += process_time() - t0_partition

        # compute collision relations for each partition
        for p, partition in enumerate(grp):
            # choose representative velocity
            repr_vel = partition[0]
            # generate collision velocities for representative
            t0_get_colvels = process_time()
            repr_colvels = self.get_colvels(extended_grids,
                                            masses,
                                            repr_vel,
                                            max_distance)
            # to reflect / rotate repr_colvels into default symmetry region
            # multiply with transposed matrix
            if grp_sym is not None:
                repr_colvels = np.einsum("ji, nkj->nki",
                                         sym_mat[grp_sym[p][0]],
                                         repr_colvels - repr_vel)
            # shift to zero for other partition elements
            else:
                repr_colvels -= repr_vel
            t_get_colvels += process_time() - t0_get_colvels

            # compute partitions collision relations, based on repr_colvels
            for pos, v0 in enumerate(partition):
                # shift repr_colvels onto v0
                t0_shifting = process_time()
                if grp_sym is None:
                    new_colvels = repr_colvels + v0
                # rotate and shift repr_colvels onto v0
                else:
                    new_colvels = np.einsum("ij, nkj->nki",
                                            sym_mat[grp_sym[p][pos]],
                                            repr_colvels)
                    new_colvels += v0
                t_shifting += process_time() - t0_shifting
                # get indices
                t0_get_idx = process_time()
                new_rels = self.get_idx([s0, s0, s1, s1], new_colvels)
                t_get_idx += process_time() - t0_get_idx
                # remove out-of-bounds or useless collisions
                t0_choice = process_time()
                choice = np.where(
                    # must be in the grid
                    np.all(new_rels >= 0, axis=1)
                    # must be effective
                    & (new_rels[..., 0] != new_rels[..., 3])
                    & (new_rels[..., 0] != new_rels[..., 1])
                )
                t_choice += process_time() - t0_choice
                # add relations to list
                relations.extend(new_rels[choice])
    t_total = process_time() - t0_total
    # convert list into array
    relations = np.array(relations, dtype=int)
    # remove redundant collisions
    t0_filter = process_time()
    relations = self.filter(self.key_index(relations), relations)
    t_filter = process_time() - t0_filter
    # sort collisions for better comparability
    relations = self.sort(self.key_index(relations), relations)
    group["total_time"][idx] = t_total
    group["colvel_time"][idx] = t_get_colvels
    group["shifting_time"][idx] = t_shifting
    group["get_idx_time"][idx] = t_get_idx
    group["choice_time"][idx] = t_choice
    group["filter_time"][idx] = t_filter
    return relations


def vectorized(self, group=None, idx=None):
    return cmp_relations(self, group_by="None", cufoff=True, group=group, idx=idx)


def group_distance(self, group=None, idx=None):
    return cmp_relations(self, group_by="distance", cufoff=True, group=group, idx=idx)


def group_distance_no_cutoff(self, group=None, idx=None):
    return cmp_relations(self, group_by="distance", cufoff=False, group=group, idx=idx)


def group_sorted_distance(self, group=None, idx=None):
    return cmp_relations(self, group_by="sorted_distance", cufoff=True, group=group, idx=idx)


def group_sorted_distance_no_cutoff(self, group=None, idx=None):
    return cmp_relations(self, group_by="sorted_distance", cufoff=False, group=group, idx=idx)


def group_norm_and_sorted_distance(self, group=None, idx=None):
    return cmp_relations(self, group_by="norm_and_sorted_distance", cufoff=True, group=group, idx=idx)


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
    COLORS = {"four_loop": "tab:red",
              "three_loop": "tab:pink",
              "vectorized": "tab:orange",
              "group_distance": "tab:purple",
              "group_sorted_distance": "tab:green",
              "group_sorted_distance_no_cutoff": "tab:olive",
              "group_norm_and_sorted_distance": "tab:blue"}
    # plot speedup from 4-loot to 3-loop
    ALL_ALGS = [["four_loop"],
                ["four_loop", "three_loop"],
                ["three_loop", "vectorized"],
                ["vectorized", "group_distance"],
                ["group_distance", "group_sorted_distance"],
                ["group_sorted_distance", "group_sorted_distance_no_cutoff"],
                ["group_sorted_distance", "group_norm_and_sorted_distance"],
                ]


    for CUR_ALGS in ALL_ALGS:
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
                widths = np.linspace(-0.5, 0.5, len(CUR_ALGS) + 2)
                ax[d].bar(x_vals[d] + widths[a+1], res, color=COLORS[alg],
                          width=widths[1] - widths[0], label=label,)
                ax[d].set_ylabel("Computation Time In Seconds")

            ax[d].set_ylabel("Computation Time In Seconds")
            ax[d].set_xlabel("Grid Shapes".format(dim))
            ax[d].set_axisbelow(True)
            ax[d].yaxis.grid(color='darkgray', linestyle='dashed')
            plt_beg = 0
            plt_spacing = 1
            ax[d].set_xticks(x_vals[d][plt_beg::plt_spacing])
            ax[d].set_xticklabels(SHAPES[d + 2][:max_idx[d], 0][plt_beg::plt_spacing])


        fig.suptitle("Collision Generation Time "
                        "for Masses = {} and Different Grid Shapes".format(masses))
        ax[0].legend(title="Algorithms:", loc="upper left")
        plt.tight_layout()
        plt.show()
        del fig, ax
