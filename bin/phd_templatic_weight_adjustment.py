import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from time import process_time

########################
#   Weight Balancing   #
########################
import boltzpy as bp

# Adjust a 3-Species Mixture
model = bp.CollisionModel(masses=[2, 3],
                          shapes=[[9, 9],
                                  [9, 9]],
                          base_delta=0.25,
                          collision_factors=[5e4]
                          )


# Choose Mean Velocity and Temperature Parameters
T = 3
MV = [0, 0]
# Verify parameters with model.temp_range()
model.temperature_range(rtol=1e-3,
                        mean_velocity=MV)

from boltzpy.Tools import WeightAdjustment
# balance gains for a reference Maxwellian
# WeightAdjustment inherits from bp.HomogeneousRule
wadj = WeightAdjustment(
    number_densities=[1, 1],
    mean_velocities=[MV, MV],
    temperatures=[T, T],
    **model.__dict__)

# balance gains for species and energy transfer
k_spc = wadj.key_species()[:, 1:3]
k_et = wadj.key_energy_transfer()
grp_spc_et = wadj.group((k_spc, k_et))

# define ax descriptions
titles = {(0,0,0): r"Intraspecies "
                   r"$\mathfrak{C}^{1,1}$",
          (1,1,0): r"Intraspecies "
                   r"$\mathfrak{C}^{2,2}$",
          (0,1,0): r"ET Collisions "
                   r"$\mathfrak{C}^{1,2}_{ET}$",
          (0,1,1): r"NET Collisions "
                   r"$\mathfrak{C}^{1,2}_{NET}$"}
suptitle="Collision Gains Grouped by " \
         "Species and Energy Transfer"
ylabels=("Species 1", "Species 2")

# plot current gains
wadj.plot_gains(grp_spc_et,
                titles=titles,
                suptitle=suptitle,
                ylabels=ylabels,
                file_address=bp.SIMULATION_DIR + "/phd_weight_adj_1.pdf")

# define gain ratios for each group
gain_ratios = {key: 1.0 for key in grp_spc_et.keys()}
# balance specific gains
wadj.balance_gains(grp_spc_et, gain_ratios, verbose=True)
# apply changes to original model
model.update_collisions(model.collision_relations,
                        wadj.collision_weights)

# plot balanced gains
wadj.plot_gains(grp_spc_et,
                titles=titles,
                suptitle="Balanced " + suptitle,
                ylabels=ylabels,
                file_address=bp.SIMULATION_DIR + "/phd_weight_adj_2.pdf")

#################################
#   Angular Weight Adjustment   #
#################################
from boltzpy.Tools import AngularWeightAdjustment
# methods for angular weight adjustments
# inherits from WeightAdjustment
abwa = AngularWeightAdjustment(
    number_densities=[1, 1],
    mean_velocities=[[0, 0], [0,0]],
    temperatures=[T, T],
    **model.__dict__)

# print current directional viscosities
abwa.get_viscosities()

# first adjust and use only single species
# then use all collisions for computation
# but adjust only the intraspecies collisions
grp_spc = abwa.group(abwa.key_species()[:, 1:3])
for key in [(0, 0), (1, 1), (0, 1)]:
    cols_used = grp_spc[key] if key[0] == key[1] else None
    abwa.balance_angles(cols_adj=grp_spc[key],
                        cols_used=cols_used,
                        species_used=key,
                        rtol=1e-2,
                        verbose=True)

# print adjusted directional viscosities
abwa.get_viscosities()

# apply changes to original model
model.update_collisions(model.collision_relations,
                        abwa.collision_weights)

# store adjusted model on disk
import h5py
file_adress = bp.SIMULATION_DIR + "/adjusted_model.hdf5"
with h5py.File(file_adress, mode="w") as file:
    model.save(file)

del abwa


# print("plot balanced gains")
# abwa.plot_gains(grp_spc_et,
#                 titles=titles,
#                 suptitle="Balanced " + suptitle,
#                 ylabels=ylabels,
#                 file_address=bp.SIMULATION_DIR + "/phd_weight_adj_3.pdf")
#
# print("plot persistence over altered number densities")
# N_POINTS = 10
# ND_PARAMS = [(1, p) for p in np.linspace(0, 1, N_POINTS)]
# ND_PARAMS += [(p, 1) for p in np.linspace(0, 1, N_POINTS)[1:]][::-1]
#
# VISC_RESULTS = []
# REL_ERRS = []
# for nd in ND_PARAMS:
#     initial_state = abwa.cmp_initial_state(
#         number_densities=nd,
#         mean_velocities=np.full((model.nspc, model.ndim),MV),
#         temperatures=np.full(model.nspc, T))
#     abwa.initial_state = initial_state
#     VISC_RESULTS.append(abwa.get_viscosities())
#     REL_ERRS.append(max(VISC_RESULTS[-1]) / min(VISC_RESULTS[-1]) - 1)
#     print(nd, " : ", VISC_RESULTS[-1])
#
# VISC_RESULTS = np.array(VISC_RESULTS)
# # plt.plot(VISC_RESULTS[:, 0], c="r")
# # plt.plot(VISC_RESULTS[:, 1], c="b")
# plt.plot(REL_ERRS)
# plt.show()


#################################
#       Model Reduction         #
#################################
# try to balance by species and energy transfer
class_keys = model.merge_keys(k_spc, k_et)
# add collisions based on shape
sub_keys = model.key_shape(model.collision_relations)

from boltzpy.Tools import GainBasedModelReduction
# setup (randomized) collision reduction
gbmr = GainBasedModelReduction(
    balance_keys=class_keys,
    selection_keys=sub_keys,
    force_normality_collisions=True,
    gain_factor_normality_collisions=0.0,
    number_densities=[1, 1],
    mean_velocities=[[0, 0], [0, 0]],
    temperatures=[T, T],
    **model.__dict__)

gbmr.plot_reduction(
    legend_title="Collision Groups",
    legend_ncol=2,
    yscale="log",
    file_address=bp.SIMULATION_DIR + "/phd_col_red.pdf")


# print selection indices of vertical lines
for key, idx in gbmr.log_empty_times.items():
    print(key, " at ", idx)

print("apply collision selection")
idx = gbmr.log_empty_times[(0, 0, 0)]
choice = gbmr.get_selection(idx)
col_rels = model.collision_relations[choice]
ncols = col_rels.shape[0]
model.update_collisions(col_rels,
                        np.full(ncols, 1e5))

# rebalance weights
wadj = WeightAdjustment(
    number_densities=[1, 1],
    mean_velocities=[MV, MV],
    temperatures=[T, T],
    **model.__dict__)
# group reduced collisions
k_spc = wadj.key_species()[:, 1:3]
k_et = wadj.key_energy_transfer()
grp_spc_et = wadj.group((k_spc, k_et))
wadj.balance_gains(grp_spc_et, gain_ratios, verbose=True)
model.update_collisions(model.collision_relations,
                        wadj.collision_weights)

# rebalance angles
# Note: This often fails for single species,
# if the collision set is not rich enough
# For mixtures we often require a larger range for initial weights
abwa = AngularWeightAdjustment(
    cur_dt=1e-7,
    number_densities=[1, 1],
    mean_velocities=[[0, 0], [0,0]],
    temperatures=[T, T],
    **model.__dict__)
grp_spc = abwa.group(abwa.key_species()[:, 1:3])
for key in [(0, 0), (1, 1), (0, 1)]:
    print("Enforcing Angular invariace for ", key)
    cols_used = grp_spc[key] if key[0] == key[1] else None
    abwa.balance_angles(cols_adj=grp_spc[key],
                        cols_used=cols_used,
                        species_used=key,
                        rtol=1e-2,
                        initial_weights=[0.1, 8.0],
                        verbose=True)
model.update_collisions(model.collision_relations,
                        abwa.collision_weights)







#
#
# # fig, axes = plt.subplots(nrows=1, ncols=rule.nspc,
# #                          figsize=(12.75, 5.05),
# #                          constrained_layout=True)
#
# # rule.initial_state[...] = 1.0
# # for key, val in grp.items():
# #     gain_arr = rule.gain_term(val)
# #     fig.suptitle(key)
# #     for s in rule.species:
# #         arr = gain_arr[rule.idx_range(s)]
# #         arr = arr.reshape(rule.shapes[s])
# #         axes[s].imshow(arr, cmap='coolwarm',
# #                    interpolation="quadric",
# #                    origin="lower",
# #                    vmax=gain_arr.max() * 1.2,
# #                    vmin=0)
# #     plt.savefig(str(key) + ".png")
#
# # determine a proper temperature range of the model
# # compute per species to compare ranges
# temp_range = {s: model.temperature_range( s=s,
#                                          rtol=1e-3,
#                                          mean_velocity=MV)
#               for s in model.species}
# print("Temperatures Range of each Grid:\n"
#       "(if range varies too much, change shape)")
# for key, val in temp_range.items():
#     print(key, ":\t", list(val))
#
# # Define reference Temperature for Adjustments
#
#
#
# # path = bp.SIMULATION_DIR + "/" + "phd_templatic_reduction.hdf5"
# tic = process_time()
# # if os.path.exists(path):
# #     print("Load DVM from File...", end="", flush=True)
# #     FILE = h5py.File(path, mode="a")
# #     model = bp.CollisionModel.load(FILE)
# # else:
# #     print("Creating DVM....", end="", flush=True)
# model = bp.CollisionModel(masses=[4, 5, 6],
#                           shapes=[[7, 7],
#                                   [9, 9],
#                                   [9, 9]],
#                           base_delta=0.1
#                           )
#     # FILE = h5py.File(path, mode="w")
#     # model.save(FILE)
#     # FILE.flush()
# print("Done!")
# toc = process_time()
# print("Time taken: %.3f seconds" % (toc-tic))
#
# # Define a reference Mean Velocity
# # should be the a maximum for the simulation
# MAX_MV = 2
#
# # determine a proper temperature range of the model
# # compute per species to compare ranges
# temp_range = {s: model.temperature_range( s=s,
#                                          rtol=1e-3,
#                                          mean_velocity=MV)
#               for s in model.species}
# print("Temperatures Range of each Grid:\n"
#       "(if range varies too much, change shape)")
# for key, val in temp_range.items():
#     print(key, ":\t", list(val))
#
# # Define reference Temperature for Adjustments
# T = 19
#
# # use a homogeneous simulation to compute gains
# # based on a reference Maxwellian
# rule = bp.HomogeneousRule(
#     number_densities=np.full(model.nspc, 1),
#     mean_velocities=np.full((model.nspc, model.ndim), MV),
#     temperatures=np.full(model.nspc, T),
#     **model.__dict__)   # use model parameters
#
# # Apply non-gain-based weight adjustments
# pass
# # We reccommend to save the adjusted model on the disc
#
#
# # balance gains for species and energy transfer
# k_spc = rule.key_species(rule.collision_relations)[:, 1:3]
# k_et = rule.key_energy_transfer(rule.collision_relations)
# class_keys = rule.merge_keys(k_spc, k_et)
#
# # add collisions based on shape
# sub_keys = rule.key_orbit(rule.collision_relations)
# reduction = GainBasedModelReduction(rule, class_keys, sub_keys,
#                                     gain_factor_normality_collisions=1e-2)
#
# grp = model.group((k_spc, k_et))
# for k, idx in reduction.log_empty_times.items():
#     print(k, ": ", grp[k].shape[0], " / ", reduction.log_ncols[idx])
#
# reduction.plot(legend_ncol=3, yscale="log")
















# rule.plot_collisions(rule.collision_relations)
# spc = rule.key_species(rule.collision_relations)
# et = rule.key_energy_transfer(rule.collision_relations)
# grp = rule.group((spc, et), rule.collision_relations)
# for key, cols in grp.items():
#     print("key = ", key)
#     print("ncols = ", cols.shape[0])
#     rule.plot_collisions(cols)


#
# # compute gains for intraspecies, ET and NET collisions
# k_spc = rule.key_species(rule.collision_relations)[:, 1:3]
# k_et = rule.key_energy_transfer(rule.collision_relations)
# grp = rule.group((k_spc, k_et))
#
# # group collisions further by shape
# k_shape = rule.key_shape(rule.collision_relations)
# subgrp = rule.group((k_spc, k_et, k_shape))
# # assign each group a list of its subkeys
# subkeys = {key: [subkey for subkey in subgrp.keys()
#                  if subkey[:3] == key]
#            for key in grp.keys()}
# # assign each subkey a probability, based on its weights
# # This determines its chance to be picked into the DVM
# # use maximum weight to keep important collisions
# subprobs = {
#     key: [np.max(rule.collision_weights[subgrp[subkey]])
#           for subkey in subkeys[key]]
#     for key in grp.keys()
# }
#
# # assign each subgroup its impact on the gains
# subgains = {key: [] for key in grp.keys()}
# # based on a desired ratio
# GAIN_INTRA = 1.0
# GAIN_ET = 0.6
# GAIN_NET = 0.4
# # compute subgains
# for key in grp.items():
#     # define s, r, and is_et
#     (s, r, is_et) = key
#     for subkey in subkeys[key]:
#         # compute the per velocity gain (array)
#         gain_array = rule.gain_term(subgrp[subkey])
#         # use a number density based gain
#         gain_val = rule.cmp_number_density(gain_array)
#         # choose gain (ET or NET)
#         if s == r:
#             subgains[key] = gain_val / GAIN_INTRA
#         elif is_et:
#             subgains[key] = gain_val / GAIN_ET
#         else:
#             subgains[key] = gain_val / GAIN_NET
#
# # define a new array of collision weights
# # to collect picked collisions
# weights = np.zeros(rule.ncols, dtype=float)
# # add required collisions for normality (intraspecies)
# is_required = rule.key_is_normality_collision(rule.collision_relations)
# weights[is_required] = 1
#
# # compute gains, with correction for desired ratio
# GAINS = dict()
#
#
#
# MAX_COLS = 1000
# while rule.ncols > MAX_COLS:
#     print("\r %d / %d" % (rule.ncols, model.ncols))
#     rule.collision_weights[:] = 1
#     rule.update_collisions()
#     LOG_NCOLS.append(rule.ncols)
#     # group collisions by species and energy transfer
#     # for gain based weight adjustments
#     k_et = rule.key_energy_transfer(rule.collision_relations)
#     # use only 2 elements for species keys
#     k_spc = rule.key_species(rule.collision_relations)[:, 1:3]
#     grp = rule.group((k_spc, k_et))
#
#     # normalize intraspecies collisions by gain
#     GAIN_INTRA = 1.0
#     GAIN_ET = 0.6
#     GAIN_NET = 0.4
#
#     # normalize interspecies collisions by gain
#     for key in grp.keys():
#         # define s, r, and is_et
#         (s, r, is_et) = key
#         # compute the per velocity gain (array)
#         gain_array = rule.gain_term(grp[key])
#         # use a number density based gain
#         gain_val = rule.cmp_number_density(gain_array)
#         LOG_GAINS[key].append(gain_val)
#         # choose gain (ET or NET)
#         if s == r:
#             GAIN = GAIN_INTRA
#         elif is_et:
#             GAIN = GAIN_ET
#         else:
#             GAIN = GAIN_NET
#         # adjust collision weights to match gain
#         rule.collision_weights[grp[key]] *= GAIN / gain_val
#     # update collision computation matrix!
#     # only then are new weights applied in cmputations
#     rule.update_collisions()
#
#     probability = np.full(rule.ncols, -1, dtype=float)
#     k_shape = rule.key_shape(rule.collision_relations)
#     grp = rule.group(k_shape)
#     for pos in grp.values():
#         gain_array = rule.gain_term(pos)
#         gain_val = rule.cmp_number_density(gain_array)
#         probability[pos] = gain_val
#     assert np.all(probability >= 0)
#     probability = probability.max() - probability
#     # # eliminate some collisions randomly,
#     # # transform weights to a probability
#     # probability = rule.collision_weights
#     # probability = probability.max() - probability
#     # remove collision shapes at random
#     rule.remove_collisions(probability=probability,
#                            key_function=rule.key_orbit,
#                            update_collision_matrix=False)
#     rule.update_collisions()
#     for key, val in LOG_GAINS.items():
#         if len(LOG_NCOLS) != len(val):
#             print(key, len(LOG_NCOLS), len(val))
#
# # plot gain values over reduction process
# fig, ax = plt.subplots()
# for key, val in LOG_GAINS.items():
#     ax.plot(LOG_NCOLS, val, "-",
#             label=key)
# ax.set_xlim(max(LOG_NCOLS), min(LOG_NCOLS))
# ax.set_yscale("log")
# plt.legend()
# plt.show()
#
# rule.plot_collisions(rule.collision_relations)
# spc = rule.key_species(rule.collision_relations)
# et = rule.key_energy_transfer(rule.collision_relations)
# grp = rule.group((spc, et), rule.collision_relations)
# for key, cols in grp.items():
#     print("key = ", key)
#     print("ncols = ", cols.shape[0])
#     rule.plot_collisions(cols)

