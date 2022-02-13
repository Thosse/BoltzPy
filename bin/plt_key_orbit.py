
# Desired Command / Road Map
import matplotlib.pyplot as plt
import boltzpy as bp
import numpy as np


def plot_key_differences(model,
                         save=False,
                         keys=None):
    if keys is None:
        keys = model.key_orbit(model.collision_relations)

    ##############################################################
    # create plot, that beta_hat ist not sufficient as key_orbit #
    ##############################################################
    grp = model.group(keys,
                      model.collision_relations)
    i = 0
    for key, val in grp.items():
        # compute orbits of vals
        orbs = model._key_orbit(val)
        unique_orbs = len(np.unique(orbs, axis=0))
        # show all
        if unique_orbs == 1:
            continue
        print("Model parameters:")
        print("masses = ", model.masses,
              "\nshapes = ", model.shapes,
              "\nspacings = ", model.spacings)
        print("unique orbs = ", unique_orbs)
        # create plots
        projection = "3d" if model.ndim == 3 else None
        fig_height = 12.75 / unique_orbs
        fig, axes = plt.subplots(1, unique_orbs,
                                 constrained_layout=True, sharey="all",
                                 subplot_kw=dict(projection=projection),
                                 figsize=(12.75, fig_height))
        print("key = ", key)
        grp2 = model.group(orbs, val)
        # if unique_orbs == 1:
        #     axes = [axes]
        for (k, v), ax in zip(grp2.items(), axes):
            print("shape = ", v.shape)
            model.plot_collisions(v, plot_object=ax)

            # print key velocities in black
            kvels = model.vels[np.tile(key[:4], 2)]
            print("kvels = ", kvels[:4])
            kvels = kvels.transpose()
            ax.plot(*kvels, color="red", lw=2)

            # print repr velocities in red
            rvels = model.vels[np.tile(v[0], 2)]
            print("rvels = ", rvels[:4])
            rvels = rvels.transpose()
            ax.plot(*rvels, color="black", lw=2)
        if save:
            plt.savefig(bp.SIMULATION_DIR + "/plt_key_orbit_fails" + str(i) + ".eps")
            i += 1
        else:
            plt.show()


# m = bp.CollisionModel([16, 71, 16], [[6,6], [10, 10], [4,4]], spacings=[12, 16, 12])
# grp = m.group(m.key_orbit(m.collision_relations), m.collision_relations)
# print(grp.keys())
# print(grp[(21, 22, 147, 151, 144, 576)])
# m.plot_collisions(grp[(21, 22, 147, 151, 144, 576)])

# def ref_orbit(self, relations):
#     """"Determines an unique id for the collisons orbit."""
#     # get colliding velocities from indices
#     colvels = self.i_vels[relations]
#     # get all elements of the orbit
#     orbvels = np.einsum("sij, ckj -> cski",
#                         self.symmetry_matrices,
#                         colvels)
#     # group colvels by species, entries are indices of relations
#     grp = self.group(self.key_species(relations))
#     # get relations from colliding velocities
#     keys = np.zeros(orbvels.shape[:3], dtype=int)
#     for spc, idx in grp.items():
#         keys[idx] = self.get_idx(spc, orbvels[idx])
#     # sort relations
#     keys.sort(axis=-1)
#     for o, orbit in enumerate(keys):
#         keys[o] = self.sort(orbit, orbit)
#     # each key must be a 1d array
#     keys = keys.reshape((keys.shape[0], -1))
#     return keys

#########################################
#   Create Plots of Failing key_orbit   #
#########################################
# # plot orbits that require shape
# model = bp.CollisionModel([1],
#                           [(5, 5)])
# keys = model.key_orbit(model.collision_relations)[:, :4]
# plot_key_differences(model, False, keys)
#
# # plot orbits that require angle or center of gravity
# model = bp.CollisionModel([1, 3],
#                           [(7, 7), (9, 9)])
# plot_key_differences(model, False)


#############################
#   Do Randomized Tests     #
#############################
MAX_MASS = 100
MAX_SHAPE = {2: 13, 3: 7}

print("Beginning Randomized Tests! Press CTRL + C to exit.")
for _ in range(500000000):
    try:
        nspc = 2 #np.random.randint(1, 5)
        ndim = 3 # np.random.randint(2, 4)
        masses = np.random.randint(1, MAX_MASS, size=nspc)
        if np.unique(masses).size < masses.size:
            continue
        spacings = None # 2 * np.random.randint(1, 13, size=nspc)
        shapes = np.random.randint(3, MAX_SHAPE[ndim], size=(nspc, ndim))
        use_cubic_grids = True #bool(np.random.randint(0, 2))
        if use_cubic_grids:
            shapes[...] = shapes[:, 0, None]
        i = 0

        model = bp.CollisionModel(masses, shapes,
                                  spacings=spacings,
                                  setup_collision_matrix=False)
        keys = (model.key_orbit(model.collision_relations),
                model.key_center_of_gravity(model.collision_relations, False),
                model.key_angle3(model.collision_relations))
        plot_key_differences(model, False, keys)
    except :
        print("Stopped at #", _)
        raise Exception

# masses =  [ 6 33]
# shapes =  [[3 3 3]
#  [4 4 4]]
# spacings =  [22  4]
