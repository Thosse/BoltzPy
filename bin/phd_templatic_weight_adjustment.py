import numpy as np
import boltzpy as bp
import matplotlib.pyplot as plt

# create a DVM with 3 species
# by default, mass adjusted spacings are used
model = bp.CollisionModel(masses=[2, 3, 4],
                          shapes=[[7, 7],
                                  [9, 9],
                                  [11, 11]],
                          )

# Define a reference Mean Velocity
# should be the a maximum for the simulation
MV = 4

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
T = 180

# the class for spatially homogeneous simulations
# provides methods for gain based adjustments
# based on the reference parameters
rule = bp.HomogeneousRule(
    number_densities=[1, 1, 1],
    mean_velocities=np.full((3, 2), MV),
    temperatures=np.full(3, T),
    **model.__dict__)   # use model parameters

# set up log of gains (before adjustment)
k_et = rule.key_energy_transfer(rule.collision_relations)
k_spc = rule.key_species(rule.collision_relations)[:, 1:3]
grp = rule.group((k_spc, k_et))
LOG_GAINS = {key: [] for key in grp.keys()}
LOG_NCOLS = []

MAX_COLS = 1000
while rule.ncols > MAX_COLS:
    print("\r %d / %d" % (rule.ncols, model.ncols))
    rule.collision_weights[:] = 1
    rule.update_collisions()
    LOG_NCOLS.append(rule.ncols)
    # group collisions by species and energy transfer
    # for gain based weight adjustments
    k_et = rule.key_energy_transfer(rule.collision_relations)
    # use only 2 elements for species keys
    k_spc = rule.key_species(rule.collision_relations)[:, 1:3]
    grp = rule.group((k_spc, k_et))

    # normalize intraspecies collisions by gain
    GAIN_INTRA = 1.0
    GAIN_ET = 0.6
    GAIN_NET = 0.4

    # normalize interspecies collisions by gain
    for key in grp.keys():
        # define s, r, and is_et
        (s, r, is_et) = key
        # compute the per velocity gain (array)
        gain_array = rule.gain_term(grp[key])
        # use a number density based gain
        gain_val = rule.cmp_number_density(gain_array)
        LOG_GAINS[key].append(gain_val)
        # choose gain (ET or NET)
        if s == r:
            GAIN = GAIN_INTRA
        elif is_et:
            GAIN = GAIN_ET
        else:
            GAIN = GAIN_NET
        # adjust collision weights to match gain
        rule.collision_weights[grp[key]] *= GAIN / gain_val
    # update collision computation matrix!
    # only then are new weights applied in cmputations
    rule.update_collisions()

    # Apply further weight adjustments
    # based on angle, shape or orbit

    probability = np.full(rule.ncols, -1, dtype=float)
    k_shape = rule.key_shape(rule.collision_relations)
    grp = rule.group(k_shape)
    for pos in grp.values():
        gain_array = rule.gain_term(pos)
        gain_val = rule.cmp_number_density(gain_array)
        probability[pos] = gain_val
    assert np.all(probability >= 0)
    probability = probability.max() - probability
    # # eliminate some collisions randomly,
    # # transform weights to a probability
    # probability = rule.collision_weights
    # probability = probability.max() - probability
    # remove collision shapes at random
    rule.remove_collisions(probability=probability,
                           key_function=rule.key_orbit,
                           update_collision_matrix=False)
    rule.update_collisions()
    for key, val in LOG_GAINS.items():
        if len(LOG_NCOLS) != len(val):
            print(key, len(LOG_NCOLS), len(val))

# plot gain values over reduction process
fig, ax = plt.subplots()
for key, val in LOG_GAINS.items():
    ax.plot(LOG_NCOLS, val, "-",
            label=key)
ax.set_xlim(max(LOG_NCOLS), min(LOG_NCOLS))
ax.set_yscale("log")
plt.legend()
plt.show()

rule.plot_collisions(rule.collision_relations)
spc = rule.key_species(rule.collision_relations)
et = rule.key_energy_transfer(rule.collision_relations)
grp = rule.group((spc, et), rule.collision_relations)
for key, cols in grp.items():
    print("key = ", key)
    print("ncols = ", cols.shape[0])
    rule.plot_collisions(cols)

