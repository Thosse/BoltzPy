import h5py
import os
import numpy as np
import boltzpy as bp
import matplotlib.pyplot as plt
from time import process_time
from boltzpy.Tools import GainBasedModelReduction
from boltzpy.Tools import balance_gains
from boltzpy.Tools import plot_gains

# Example: Gain based weight adjustment
import boltzpy as bp
from boltzpy.Tools import balance_gains

# Adjust a 3-Species Mixture
model = bp.CollisionModel(masses=[2, 3],
                          shapes=[[9, 9],
                                  [9, 9]],
                          base_delta=1.0
                          )
# Choose Mean Velocity and Temperature Parameters
MAX_MV = 0
T = 45
# Verify parameters with model.temp_range()
print(model.temperature_range(rtol=1e-3,
                        mean_velocity=MAX_MV))

# use a homogeneous simulation
# to compute gains based on a reference Maxwellian
rule = bp.HomogeneousRule(
    number_densities=np.full(model.nspc, 1),
    mean_velocities=np.full((model.nspc, model.ndim),
                            MAX_MV),
    temperatures=np.full(model.nspc, T),
    **model.__dict__)

# balance gains for species and energy transfer
col_rels = rule.collision_relations
k_spc = rule.key_species(col_rels)[:, 1:3]
k_et = rule.key_energy_transfer(col_rels)
grp = rule.group((k_spc, k_et))

plot_gains(rule,
           grp=grp)

# specify gain ratios for each group
GAINS = {"INTRA": 1.0,
         "ET": 1,
         "NET": 1}

# create a dictionary with the desired gain ratios
gain_ratios=dict()
for key in grp.keys():
    # define s, r, and is_et
    (s, r, is_et) = key
    if s == r:
        gain_ratios[key] = GAINS["INTRA"]
    elif is_et:
        gain_ratios[key] = GAINS["ET"]
    else:
        gain_ratios[key] = GAINS["NET"]

# apply desired gain_ratios
print("Balance Gains!")
balance_gains(rule, grp, gain_ratios, verbose=True)

plot_gains(rule,
           grp=grp)

# fig, axes = plt.subplots(nrows=1, ncols=rule.nspc,
#                          figsize=(12.75, 5.05),
#                          constrained_layout=True)

# rule.initial_state[...] = 1.0
# for key, val in grp.items():
#     gain_arr = rule.gain_term(val)
#     fig.suptitle(key)
#     for s in rule.species:
#         arr = gain_arr[rule.idx_range(s)]
#         arr = arr.reshape(rule.shapes[s])
#         axes[s].imshow(arr, cmap='coolwarm',
#                    interpolation="quadric",
#                    origin="lower",
#                    vmax=gain_arr.max() * 1.2,
#                    vmin=0)
#     plt.savefig(str(key) + ".png")

# determine a proper temperature range of the model
# compute per species to compare ranges
temp_range = {s: model.temperature_range( s=s,
                                         rtol=1e-3,
                                         mean_velocity=MAX_MV)
              for s in model.species}
print("Temperatures Range of each Grid:\n"
      "(if range varies too much, change shape)")
for key, val in temp_range.items():
    print(key, ":\t", list(val))

# Define reference Temperature for Adjustments



# path = bp.SIMULATION_DIR + "/" + "phd_templatic_reduction.hdf5"
tic = process_time()
# if os.path.exists(path):
#     print("Load DVM from File...", end="", flush=True)
#     FILE = h5py.File(path, mode="a")
#     model = bp.CollisionModel.load(FILE)
# else:
#     print("Creating DVM....", end="", flush=True)
model = bp.CollisionModel(masses=[4, 5, 6],
                          shapes=[[7, 7],
                                  [9, 9],
                                  [9, 9]],
                          base_delta=0.1
                          )
    # FILE = h5py.File(path, mode="w")
    # model.save(FILE)
    # FILE.flush()
print("Done!")
toc = process_time()
print("Time taken: %.3f seconds" % (toc-tic))

# Define a reference Mean Velocity
# should be the a maximum for the simulation
MAX_MV = 2

# determine a proper temperature range of the model
# compute per species to compare ranges
temp_range = {s: model.temperature_range( s=s,
                                         rtol=1e-3,
                                         mean_velocity=MAX_MV)
              for s in model.species}
print("Temperatures Range of each Grid:\n"
      "(if range varies too much, change shape)")
for key, val in temp_range.items():
    print(key, ":\t", list(val))

# Define reference Temperature for Adjustments
T = 19

# use a homogeneous simulation to compute gains
# based on a reference Maxwellian
rule = bp.HomogeneousRule(
    number_densities=np.full(model.nspc, 1),
    mean_velocities=np.full((model.nspc, model.ndim), MAX_MV),
    temperatures=np.full(model.nspc, T),
    **model.__dict__)   # use model parameters

# Apply non-gain-based weight adjustments
pass
# We reccommend to save the adjusted model on the disc


# balance gains for species and energy transfer
k_spc = rule.key_species(rule.collision_relations)[:, 1:3]
k_et = rule.key_energy_transfer(rule.collision_relations)
class_keys = rule.merge_keys(k_spc, k_et)

# add collisions based on shape
sub_keys = rule.key_orbit(rule.collision_relations)
reduction = GainBasedModelReduction(rule, class_keys, sub_keys,
                                    gain_factor_normality_collisions=1e-2)

grp = model.group((k_spc, k_et))
for k, idx in reduction.log_empty_times.items():
    print(k, ": ", grp[k].shape[0], " / ", reduction.log_ncols[idx])

reduction.plot(legend_ncol=3, yscale="log")

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

