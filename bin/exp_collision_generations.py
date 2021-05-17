import numpy as np
import boltzpy as bp

import matplotlib.pyplot as plt
import h5py
from time import process_time


def four_loop(model, group=None, idx=None):
    tic = process_time()
    assert isinstance(model, bp.CollisionModel)
    relations = []
    # Iterate over Specimen pairs
    colvel = np.zeros((4, model.ndim), dtype=int)
    idx_range = np.zeros((2, 2), dtype=int)
    for s0 in model.species:
        idx_range[0] = model._idx_offset[[s0, s0+1]]
        for s1 in np.arange(s0, model.nspc):
            idx_range[1] = model._idx_offset[[s1, s1+1]]
            masses = model.masses[[s0, s0, s1, s1]]
            for i0 in np.arange(*idx_range[0]):
                colvel[0] = model.i_vels[i0]
                # for i1 in np.arange(*idx_range[0]):
                for i1 in np.arange(i0+1, idx_range[0, 1]):
                    colvel[1] = model.i_vels[i1]
                    for i2 in np.arange(*idx_range[1]):
                        colvel[2] = model.i_vels[i2]
                        for i3 in np.arange(*idx_range[1]):
                            colvel[3] = model.i_vels[i3]
                            # check if its a proper Collision
                            if model.is_collision(colvel.reshape((1, 4, model.ndim)), masses):
                                if i0 == i3 and i1 == i2:
                                    continue
                                relations.append([i0, i1, i2, i3])
                            else:
                                continue
    toc = process_time()
    group["total_time"][idx] = toc - tic
    relations = np.array(relations, dtype=int, ndmin=2)
    # remove redundant collisions
    relations = model.filter(model.key_index(relations), relations)
    # sort collisions for better comparability
    relations = model.sort(model.key_index(relations), relations)
    return relations


def three_loop(model, group=None, idx=None):
    tic = process_time()
    assert isinstance(model, bp.CollisionModel)
    relations = []
    # Iterate over Specimen pairs
    colvel = np.zeros((4, model.ndim), dtype=int)
    idx_range = np.zeros((2, 2), dtype=int)
    for s0 in model.species:
        idx_range[0] = model._idx_offset[[s0, s0+1]]
        for s1 in np.arange(s0, model.nspc):
            idx_range[1] = model._idx_offset[[s1, s1+1]]
            masses = model.masses[[s0, s0, s1, s1]]
            grid1 = model.subgrids(s1)
            for i0 in np.arange(*idx_range[0]):
                colvel[0] = model.i_vels[i0]
                for i1 in np.arange(i0+1, idx_range[0, 1]):
                    colvel[1] = model.i_vels[i1]
                    diff_v = colvel[1] - colvel[0]
                    for i2 in np.arange(*idx_range[1]):
                        colvel[2] = model.i_vels[i2]
                        # Calculate w1, using the momentum invariance
                        assert all((diff_v * masses[0]) % masses[2] == 0)
                        diff_w = -diff_v * masses[0] // masses[2]
                        w1 = colvel[2] + diff_w
                        if w1 not in grid1:
                            continue
                        colvel[3] = w1
                        # check if its a proper Collision
                        if model.is_collision(colvel.reshape((1, 4, model.ndim)), masses):
                            i3 = model.get_idx(s1, colvel[3])
                            if i0 == i3 and i1 == i2:
                                continue
                            relations.append([i0, i1, i2, i3])
                        else:
                            continue
    toc = process_time()
    group["total_time"][idx] = toc - tic
    relations = np.array(relations, dtype=int, ndmin=2)
    # remove redundant collisions
    relations = model.filter(model.key_index(relations), relations)
    # sort collisions for better comparability
    relations = model.sort(model.key_index(relations), relations)
    return relations


def vectorized(model, group=None, idx=None):
    tic = process_time()
    # collect collisions in a lists
    relations = []
    # Collisions are computed for every pair of specimen
    species_pairs = np.array([(s0, s1) for s0 in model.species
                              for s1 in range(s0, model.nspc)])
    # pre initialize each species' velocity grid
    grids = model.subgrids()

    # Compute collisions iteratively, for each pair
    for s0, s1 in species_pairs:
        masses = model.masses[[s0, s1]]
        # imitate grouping, to fit into the remaining algorithm
        grp = grids[s0].iG[:, np.newaxis]
        # no extended grids necessary
        extended_grids = np.array([grids[s0], grids[s1]], dtype=object)
        # compute collision relations for each partition
        for p, partition in enumerate(grp):
            # choose representative velocity
            repr_vel = partition[0]
            # generate collision velocities for representative
            repr_colvels = model.get_colvels(extended_grids,
                                             masses,
                                             repr_vel)
            # get indices
            new_rels = model.get_idx([s0, s0, s1, s1], repr_colvels)
            # remove out-of-bounds or useless collisions
            choice = np.where(
                # must be in the grid
                np.all(new_rels >= 0, axis=1)
                # must be effective
                & (new_rels[..., 0] != new_rels[..., 3])
                & (new_rels[..., 0] != new_rels[..., 1])
            )
            # add relations to list
            relations.extend(new_rels[choice])

    toc = process_time()
    group["total_time"][idx] = toc - tic
    # convert list into array
    relations = np.array(relations, dtype=int)
    # remove redundant collisions
    relations = model.filter(model.key_index(relations), relations)
    # sort collisions for better comparability
    relations = model.sort(model.key_index(relations), relations)
    return relations


def group_distance(self, group=None, idx=None):
    t_get_colvels = 0.0
    t_shifting = 0.0
    tic = process_time()
    # collect collisions in a lists
    relations = []
    # Collisions are computed for every pair of specimen
    species_pairs = np.array([(s0, s1) for s0 in self.species
                              for s1 in range(s0, self.nspc)])
    # pre initialize each species' velocity grid
    grids = self.subgrids()

    # Compute collisions iteratively, for each pair
    for s0, s1 in species_pairs:
        masses = self.masses[[s0, s1]]

        # partition grid of first specimen into partitions
        # collisions can be shifted (and rotated) in each partition
        # this saves computation time.
        # The collisions must be found in larger grids
        # to find all possible collisions
        # No partitioning at all
        # partition grids[s0] by distance
        # if s0 == s1, this is equivalent to partitioned distance (but faster)
        # partition based on distance to next grid point
        grp_keys = grids[s1].key_distance(grids[s0].iG)
        norm = bp.Grid.key_norm(grids[s0].iG)
        grp = self.group(grp_keys, grids[s0].iG, as_dict=False, sort_key=norm)
        del norm, grp_keys
        # compute representative colliding velocities in extended grids
        extended_grids = self._get_extended_grids((s0, s1), grp)
        # cutoff unnecessary velocity with max_distances,
        # when computing the reference colvels
        max_distance = self.max_i_vels[None, s0] + self.max_i_vels[[s0, s1]]

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
            repr_colvels -= repr_vel
            t_get_colvels += process_time() - t0_get_colvels
            # compute partitions collision relations, based on repr_colvels
            for pos, v0 in enumerate(partition):
                # shift repr_colvels onto v0
                t0_shifting = process_time()
                new_colvels = repr_colvels + v0
                t_shifting += process_time() - t0_shifting
                # get indices
                new_rels = self.get_idx([s0, s0, s1, s1], new_colvels)
                # remove out-of-bounds or useless collisions
                choice = np.where(
                    # must be in the grid
                    np.all(new_rels >= 0, axis=1)
                    # must be effective
                    & (new_rels[..., 0] != new_rels[..., 3])
                    & (new_rels[..., 0] != new_rels[..., 1])
                )
                # add relations to list
                relations.extend(new_rels[choice])
    toc = process_time()
    group["total_time"][idx] = toc - tic
    group["colvel_time"][idx] = t_get_colvels
    group["shifting_time"][idx] = t_shifting
    # convert list into array
    relations = np.array(relations, dtype=int)
    # remove redundant collisions
    relations = self.filter(self.key_index(relations), relations)
    # sort collisions for better comparability
    relations = self.sort(self.key_index(relations), relations)
    return relations

def group_distance_no_cutoff(self, group=None, idx=None):
    t_get_colvels = 0.0
    t_shifting = 0.0
    tic = process_time()
    # collect collisions in a lists
    relations = []
    # Collisions are computed for every pair of specimen
    species_pairs = np.array([(s0, s1) for s0 in self.species
                              for s1 in range(s0, self.nspc)])
    # pre initialize each species' velocity grid
    grids = self.subgrids()

    # Compute collisions iteratively, for each pair
    for s0, s1 in species_pairs:
        masses = self.masses[[s0, s1]]

        # partition grid of first specimen into partitions
        # collisions can be shifted (and rotated) in each partition
        # this saves computation time.
        # The collisions must be found in larger grids
        # to find all possible collisions
        # No partitioning at all
        # partition grids[s0] by distance
        # if s0 == s1, this is equivalent to partitioned distance (but faster)
        # partition based on distance to next grid point
        grp_keys = grids[s1].key_distance(grids[s0].iG)
        norm = bp.Grid.key_norm(grids[s0].iG)
        grp = self.group(grp_keys, grids[s0].iG, as_dict=False, sort_key=norm)
        del norm, grp_keys
        # compute representative colliding velocities in extended grids
        extended_grids = self._get_extended_grids((s0, s1), grp)

        # compute collision relations for each partition
        for p, partition in enumerate(grp):
            # choose representative velocity
            repr_vel = partition[0]
            # generate collision velocities for representative
            t0_get_colvels = process_time()
            repr_colvels = self.get_colvels(extended_grids,
                                            masses,
                                            repr_vel)
            # to reflect / rotate repr_colvels into default symmetry region
            # multiply with transposed matrix
            repr_colvels -= repr_vel
            t_get_colvels += process_time() - t0_get_colvels
            # compute partitions collision relations, based on repr_colvels
            for pos, v0 in enumerate(partition):
                # shift repr_colvels onto v0
                t0_shifting = process_time()
                new_colvels = repr_colvels + v0
                t_shifting += process_time() - t0_shifting
                # get indices
                new_rels = self.get_idx([s0, s0, s1, s1], new_colvels)
                # remove out-of-bounds or useless collisions
                choice = np.where(
                    # must be in the grid
                    np.all(new_rels >= 0, axis=1)
                    # must be effective
                    & (new_rels[..., 0] != new_rels[..., 3])
                    & (new_rels[..., 0] != new_rels[..., 1])
                )
                # add relations to list
                relations.extend(new_rels[choice])
    toc = process_time()
    group["total_time"][idx] = toc - tic
    group["colvel_time"][idx] = t_get_colvels
    group["shifting_time"][idx] = t_shifting
    # convert list into array
    relations = np.array(relations, dtype=int)
    # remove redundant collisions
    relations = self.filter(self.key_index(relations), relations)
    # sort collisions for better comparability
    relations = self.sort(self.key_index(relations), relations)
    return relations


def group_rotated_distance(self, group=None, idx=None):
    t_get_colvels = 0.0
    t_shifting = 0.0
    tic = process_time()
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

        # partition grid of first specimen into partitions
        # collisions can be shifted (and rotated) in each partition
        # this saves computation time.
        # The collisions must be found in larger grids
        # to find all possible collisions
        # partition grids[s0] by distance and rotation
        # group based on distances, rotated into 0 <= x <= y <= z
        grp_keys = grids[s1].key_partitioned_distance(grids[s0].iG)
        norm = bp.Grid.key_norm(grids[s0].iG)
        # group both grp and grp_sym in one go (and with same order)
        grp, grp_sym = self.group(grp_keys[..., :-1],
                                  (grids[s0].iG, grp_keys[:, -1]),
                                  as_dict=False, sort_key=norm)
        del norm, grp_keys
        # compute representative colliding velocities in extended grids
        extended_grids = self._get_extended_grids((s0, s1), grp)
        # cutoff unnecessary velocity with max_distances,
        # when computing the reference colvels
        max_distance = self.max_i_vels[None, s0] + self.max_i_vels[[s0, s1]]

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
            repr_colvels = np.einsum("ji, nkj->nki",
                                     sym_mat[grp_sym[p][0]],
                                     repr_colvels - repr_vel)

            t_get_colvels += process_time() - t0_get_colvels
            # compute partitions collision relations, based on repr_colvels
            for pos, v0 in enumerate(partition):
                # rotate and shift repr_colvels onto v0
                t0_shifting = process_time()
                new_colvels = np.einsum("ij, nkj->nki",
                                        sym_mat[grp_sym[p][pos]],
                                        repr_colvels)
                new_colvels += v0
                t_shifting += process_time() - t0_shifting
                # get indices
                new_rels = self.get_idx([s0, s0, s1, s1], new_colvels)
                # remove out-of-bounds or useless collisions
                choice = np.where(
                    # must be in the grid
                    np.all(new_rels >= 0, axis=1)
                    # must be effective
                    & (new_rels[..., 0] != new_rels[..., 3])
                    & (new_rels[..., 0] != new_rels[..., 1])
                )
                # add relations to list
                relations.extend(new_rels[choice])
    toc = process_time()
    group["total_time"][idx] = toc - tic
    group["colvel_time"][idx] = t_get_colvels
    group["shifting_time"][idx] = t_shifting
    # convert list into array
    relations = np.array(relations, dtype=int)
    # remove redundant collisions
    relations = self.filter(self.key_index(relations), relations)
    # sort collisions for better comparability
    relations = self.sort(self.key_index(relations), relations)
    return relations


def group_rotated_distance_no_cutoff(self, group=None, idx=None):
    t_get_colvels = 0.0
    t_shifting = 0.0
    tic = process_time()
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

        # partition grid of first specimen into partitions
        # collisions can be shifted (and rotated) in each partition
        # this saves computation time.
        # The collisions must be found in larger grids
        # to find all possible collisions
        # partition grids[s0] by distance and rotation
        # group based on distances, rotated into 0 <= x <= y <= z
        grp_keys = grids[s1].key_partitioned_distance(grids[s0].iG)
        norm = bp.Grid.key_norm(grids[s0].iG)
        # group both grp and grp_sym in one go (and with same order)
        grp, grp_sym = self.group(grp_keys[..., :-1],
                                  (grids[s0].iG, grp_keys[:, -1]),
                                  as_dict=False, sort_key=norm)
        del norm, grp_keys
        # compute representative colliding velocities in extended grids
        extended_grids = self._get_extended_grids((s0, s1), grp)

        # compute collision relations for each partition
        for p, partition in enumerate(grp):
            # choose representative velocity
            repr_vel = partition[0]
            # generate collision velocities for representative
            t0_get_colvels = process_time()
            repr_colvels = self.get_colvels(extended_grids,
                                            masses,
                                            repr_vel)
            # to reflect / rotate repr_colvels into default symmetry region
            # multiply with transposed matrix
            repr_colvels = np.einsum("ji, nkj->nki",
                                     sym_mat[grp_sym[p][0]],
                                     repr_colvels - repr_vel)

            t_get_colvels += process_time() - t0_get_colvels
            # compute partitions collision relations, based on repr_colvels
            for pos, v0 in enumerate(partition):
                # rotate and shift repr_colvels onto v0
                t0_shifting = process_time()
                new_colvels = np.einsum("ij, nkj->nki",
                                        sym_mat[grp_sym[p][pos]],
                                        repr_colvels)
                new_colvels += v0
                t_shifting += process_time() - t0_shifting
                # get indices
                new_rels = self.get_idx([s0, s0, s1, s1], new_colvels)
                # remove out-of-bounds or useless collisions
                choice = np.where(
                    # must be in the grid
                    np.all(new_rels >= 0, axis=1)
                    # must be effective
                    & (new_rels[..., 0] != new_rels[..., 3])
                    & (new_rels[..., 0] != new_rels[..., 1])
                )
                # add relations to list
                relations.extend(new_rels[choice])
    toc = process_time()
    group["total_time"][idx] = toc - tic
    group["colvel_time"][idx] = t_get_colvels
    group["shifting_time"][idx] = t_shifting
    # convert list into array
    relations = np.array(relations, dtype=int)
    # remove redundant collisions
    relations = self.filter(self.key_index(relations), relations)
    # sort collisions for better comparability
    relations = self.sort(self.key_index(relations), relations)
    return relations


FILENAME = bp.SIMULATION_DIR + "/exp_collision_generation.hdf5"
FILE = h5py.File(FILENAME, mode="a")
masses = [5, 7]
dim = 2
mass_str = "masses=({},{})_dim={}".format(*masses, dim)
SHAPES = np.array([np.full((2, 2), i, dtype=int)
                   for i in range(3, 50)])
ALGORITHMS = ["four_loop", "three_loop", "vectorized",
              "group_distance", "group_rotated_distance",
              "group_distance_no_cutoff",
              "group_rotated_distance_no_cutoff"]
SKIP = []

if mass_str in FILE.keys():
    compute = (input("Precomputed Solution detected for a {}-dimensional "
                     "{}x{} model.\n"
                     "Do you want to recompute? (yes/no)"
                     "".format(dim, *masses)) == "yes")
    if compute:
        del FILE[mass_str]
else:
    compute = True

if not compute:
    h5py_group = FILE[mass_str]
else:
    h5py_group = FILE.create_group(mass_str)
    FILE[mass_str]["SHAPES"] = SHAPES
    for alg in ALGORITHMS:
        FILE[mass_str].create_group(alg)
        FILE[mass_str][alg]["total_time"] = np.full(len(SHAPES), np.nan, dtype=float)
        FILE[mass_str][alg]["colvel_time"] = np.full(len(SHAPES), np.nan, dtype=float)
        FILE[mass_str][alg]["shifting_time"] = np.full(len(SHAPES), np.nan, dtype=float)

    for i, shapes in enumerate(SHAPES):
        print("Computing shape ", shapes[0])
        model = bp.CollisionModel(masses, shapes,
                                  collision_relations=[],
                                  collision_weights=[],
                                  setup_collision_matrix=False)
        results = dict()
        for a, alg in enumerate(ALGORITHMS):
            if alg in SKIP:
                continue
            tic = process_time()
            results[alg] = locals()[alg](model, FILE[mass_str][alg], i)
            toc = process_time()
            print("\t", alg, toc - tic)
            # dont wait too long for algorithms
            if toc - tic > 3600:
                print("STOP ", alg, "\nat shape= ", shapes)
                SKIP.append(alg)

        ref_results = model.key_index(results["group_rotated_distance"])
        ref_results = model.sort(ref_results, ref_results)
        for key in results.keys():
            key_results = model.key_index(results[key])
            ref_results = model.sort(key_results, key_results)
            assert np.all(key_results == ref_results)
        FILE.flush()


print("Results are:")
for alg in ALGORITHMS:
    print(alg)
    print("total:\n", FILE[mass_str][alg]["total_time"][()])
    print("colvel_time:\n", FILE[mass_str][alg]["colvel_time"][()])
    print("shifting_time:\n", FILE[mass_str][alg]["shifting_time"][()])
    print("\n")

# ################
# # create plots #
# ################
# plot speedup from 4-loot to 3-loop
ALL_ALGS = [["four_loop", "three_loop"],
            ["three_loop", "vectorized"],
            ["vectorized", "group_distance"],
            ["group_distance", "group_rotated_distance"],
            ["four_loop", "three_loop", "vectorized",
             "group_distance", "group_rotated_distance"],
            ]
ALL_PLOTS = [["bar"],
             ["bar"],
             ["bar"],
             ["bar"],
             ["logbar"],]

for CUR_ALGS, CUR_PLOTS in zip(ALL_ALGS, ALL_PLOTS):
    # get computed shape (until all algorithms, stopped)
    max_idx = -1
    for alg in CUR_ALGS:
        res = FILE[mass_str][alg]["total_time"][()]
        res = np.where(np.isnan(res), -1, res)
        max_idx = np.max([max_idx, res.argmax()])
    x_vals = np.arange(max_idx) + 1

    # plot different algorithm times
    # setup plot
    fig, ax = plt.subplots(1, len(CUR_PLOTS), constrained_layout=True, sharex="all")
    ax = np.array([ax], dtype=object).flatten()
    for p, style in enumerate(CUR_PLOTS):
        for a, alg in enumerate(CUR_ALGS):
            label = alg if p == 0 else None
            res = FILE[mass_str][alg]["total_time"][:max_idx]
            if style in ["bar", "logbar"]:
                widths = np.linspace(-0.5, 0.5, len(CUR_ALGS) + 2)
                ax[p].bar(x_vals + widths[a+1], res,
                          width=widths[1] - widths[0], label=label)
            if style in ["logbar"]:
                ax[p].set_yscale('log')
                ax[p].set_ylabel("Logarithmic Computation Time")
            else:
                ax[p].set_ylabel("Computation Time in seconds")
            plt.xticks(x_vals, SHAPES[:max_idx, 0, 0])
            ax[p].set_xlabel("Grid Widths Of Both Grids")
            ax[p].set_axisbelow(True)
            ax[p].yaxis.grid(color='gray', linestyle='dashed')

    ax[0].legend()
    plt.show()
