import h5py
import os
import numpy as np
import boltzpy as bp
import matplotlib.pyplot as plt


class gain_group:
    def __init__(self, rule, class_key, grp):
        self.class_key = class_key
        self.grp = grp
        self.keys = [key for key in grp.keys()
                     if class_key == key[:len(class_key)]]

        # determine probabilities to pick a group
        self.probabilities = np.zeros(len(self.keys), dtype=float)
        for k, key in enumerate(self.keys):
            weights = rule.collision_weights[grp[key]]
            self.probabilities[k] = np.max(weights)
        self.probabilities /= self.probabilities.sum()

        # compute gains of each collision group
        self.gains = np.zeros(self.probabilities.shape, dtype=float)
        for k, key in enumerate(self.keys):
            gain_array = rule.gain_term(grp[key])
            self.gains[k] = rule.cmp_number_density(gain_array)

        # store chosen collision groups
        self.choice = np.zeros(self.probabilities.shape, dtype=bool)
        # store gain of current collision relations
        self.current_gain = 0.0
        self.unused_grps = self.probabilities.size
        return

    @property
    def collision_relations(self):
        chosen_idx = np.where[self.choice][0]
        chosen_keys = self.keys[chosen_idx]
        chosen_rels = [self.grp[key] for key in chosen_keys]
        col_rels = np.concatenate(chosen_rels, axis=0)
        return col_rels

    def choose(self, key=None, gain_factor=1.0):
        assert self.unused_grps > 0
        if key is None:
            index = np.random.choice(self.probabilities.size,
                                     p=self.probabilities)
        else:
            key = tuple(key)
            index = self.keys.index(key)
        assert self.choice[index] == 0
        self.choice[index] = 1
        self.current_gain += self.gains[index] * gain_factor
        self.probabilities[index] = 0
        self.unused_grps -= 1
        prob_sum = self.probabilities.sum()
        if np.isclose(prob_sum, 0):
            assert self.unused_grps == 0
        else:
            self.probabilities /= self.probabilities.sum()
        return self.keys[index]

class model_reduction:
    def __init__(self, rule, class_keys, sub_keys, execute=True,
                 force_normality_collisions=True,
                 gain_factor_normality_collisions=1.0):
        assert isinstance(rule, bp.HomogeneousRule)
        self.rule = rule
        for k in [class_keys, sub_keys]:
            assert k.ndim == 2
            assert k.dtype == int
        assert class_keys.shape[0] == sub_keys.shape[0]
        self.len_class = class_keys.shape[1]
        self.len_sub = sub_keys.shape[0]
        self.keys = rule.merge_keys(class_keys, sub_keys)
        self.col_grp = rule.group(self.keys)

        # construct array of gain groups
        self.unique_keys = rule.filter(class_keys, class_keys)
        self.unique_keys = [tuple(key) for key in self.unique_keys]
        self.gain_groups = [gain_group(rule, key, self.col_grp)
                            for key in self.unique_keys]
        self.gain_groups = np.array(self.gain_groups)

        # log added keys, gains, and ncols
        self.log_keys = []
        self.log_gains = [[0 for k in self.unique_keys]]
        self.log_ncols = [0]
        self.log_empty_times = {key: -1 for key in self.unique_keys}
        self.execute(
            force_normality_collisions=force_normality_collisions,
            gain_factor_normality_collisions=gain_factor_normality_collisions)
        return

    @property
    def class_keys(self):
        return [gg.class_key for gg in self.gain_groups]

    def current_gains(self):
        return [gg.current_gain for gg in self.gain_groups]

    def next_gain_group(self):
        usable_groups = [gg for gg in self.gain_groups
                         if gg.unused_grps > 0]
        idx = np.argmin([gg.current_gain for gg in usable_groups])
        return usable_groups[idx]

    def get_reduction(self, idx):
        keys = self.log_keys[:idx]
        col_grps = [self.grp[key] for key in keys]
        col_rels = np.concatenate(col_grps, axis=0)
        return col_grps


    def add_collisions(self, key=None, gain_factor=1.0):
        if key is None:
            gg = self.next_gain_group()
            key = gg.choose(gain_factor=gain_factor)
            if gg.unused_grps == 0:
                self.log_empty_times[gg.class_key] = len(self.log_keys)
        else:
            # find gain group that contains the key
            key = tuple(key)
            class_key = key[:self.len_class]
            gg_idx = self.class_keys.index(class_key)
            gg = self.gain_groups[gg_idx]
            gg.choose(key, gain_factor)
        self.log_keys.append(key)
        self.log_gains.append(self.current_gains())
        self.log_ncols.append(self.log_ncols[-1] + self.col_grp[key].shape[0])
        # todo handle partner keys here
        # e.g. if an et or net collision was added, add its counterpart(s) as well
        # this should be hidden behind an optional parameter
        # check if it is useful!

    def execute(self,
                force_normality_collisions=True,
                gain_factor_normality_collisions=1.0):
        print("Begin Model Reduction!")

        if force_normality_collisions:
            print("Add Normality Collisions...\r", end="")
            col_rels = self.rule.collision_relations
            is_required = self.rule.key_is_normality_collision(col_rels)
            required_keys = self.keys[np.where(is_required)[0]]
            required_keys = self.rule.filter(required_keys, required_keys)
            for key in required_keys:
                self.add_collisions(key,
                                    gain_factor=gain_factor_normality_collisions)
            print("Done!")

        print("Add Collisions to Balance Gains...")
        while self.log_ncols[-1] != self.rule.ncols:
            print("\rCollisions = %7d / %7d"
                  % (self.log_ncols[-1], self.rule.ncols),
                  end="")
            self.add_collisions()
        return



path = bp.SIMULATION_DIR + "/" + "phd_templatic_reduction.hdf5"
if os.path.exists(path):
    print("Loading DVM from File....")
    FILE = h5py.File(path, mode="a")
    model = bp.HomogeneousRule.load(FILE)
else:
    print("Creating DVM....")
    # by default, mass adjusted spacings are used
    model = bp.CollisionModel(masses=[4, 5, 6],
                              shapes=[[7, 7, 7],
                                      [9, 9, 9],
                                      [9, 9, 9]],
                              base_delta=0.1
                              )
    FILE = h5py.File(path, mode="w")
    model.save(FILE)
    FILE.flush()


# model = bp.CollisionModel(masses=[2, 3],
#                           shapes=[[5, 5],
#                                   [5, 5]],
#                           )

# Define a reference Mean Velocity
# should be the a maximum for the simulation
MV = 2

# determine a proper temperature range of the model
# compute per species to compare ranges
temp_range = {s: model.temperature_range(s=s,
                                         rtol=1e-3,
                                         mean_velocity=MV)
              for s in model.species}
print("Temperatures Range of each Grid:"
      "\n(if range varies too much, change shape)")
for key, val in temp_range.items():
    print(key, ":\t", list(val))

# Define reference Temperature for Adjustments
T = 19

# use a homogeneous simulation to compute gains
# this should be the same for all weight adjustments
rule = bp.HomogeneousRule(
    number_densities=np.full(model.nspc, 1),
    mean_velocities=np.full((model.nspc, model.ndim), MV),
    temperatures=np.full(model.nspc, T),
    **model.__dict__)   # use model parameters

# Apply non-gain-based weight adjustments
pass
# We reccommend to save this adjusted model on the disc


# balance gains for species and energy transfer
k_spc = rule.key_species(rule.collision_relations)[:, 1:3]
k_et = rule.key_energy_transfer(rule.collision_relations)
class_keys = rule.merge_keys(k_spc, k_et)

# add collisions based on shape
sub_keys = rule.key_shape(rule.collision_relations)
reduction = model_reduction(rule, class_keys, sub_keys,
                            gain_factor_normality_collisions=1e-2)

for k, idx in reduction.log_empty_times.items():
    print(k, ": ", reduction.log_ncols[idx])

# plot gain values over reduction process
fig, ax = plt.subplots()

gains = np.array(reduction.log_gains)
for k, key in enumerate(reduction.class_keys):
    ax.plot(reduction.log_ncols,
            gains[:, k],
            "-",
            label=key)
# ax.set_yscale("log")
plt.legend()
plt.show()

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

