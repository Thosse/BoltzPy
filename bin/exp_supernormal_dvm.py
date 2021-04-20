import boltzpy as bp
import numpy as np
masses = [2, 3, 5]

def n_collision_invariants(masses, vels0, vels1=None):
    # construct a matrix with a collision invariant in each line
    matrix = np.zeros((3 + vels0.shape[1], vels0.shape[0] + vels1.shape[0]),
                      dtype=int)
    # number density 0
    matrix[0, 0: vels0.shape[0]] = 1
    # number density 1
    matrix[1, vels0.shape[0]:] = 1
    # total energy
    matrix[2, 0: vels0.shape[0]] = masses[0] * np.sum(vels0**2, axis=1)
    matrix[2, vels0.shape[0]:] = masses[1] * np.sum(vels1**2, axis=1)
    # each components of total momentum is a separate line
    for i in range(vels0.shape[1]):
        matrix[3+i, 0: vels0.shape[0]] = masses[0] * vels0[:, i]
        matrix[3+i, vels0.shape[0]:] = masses[1] * vels1[:, i]
    invariants = np.linalg.matrix_rank(matrix)
    return invariants


model = bp.CollisionModel(masses=masses,
                          shapes=((5, 5), (5, 5), (5,5)),
                          base_delta=0.5,
                          spacings=(30, 20, 12))
ticks = [-30, -20, -15, -12, -10, -6, 0, 6, 10, 12, 15, 20, 30]

print("2d, normal, complete model")
print("number of collision invariants   = ",
      model.collision_invariants)
print("maximal physical Collision Invariants = ",
        n_collision_invariants(
            masses,
            model.vels[model.idx_range(0)],
            model.vels[model.idx_range(1)]))

# group collisions by species
grp = model.group(relations=model.collision_relations, key_function=model.key_species)
grp_colls = {key: model.collision_relations[grp[key]]
             for key in {(0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 1)}}
model.plot_collisions(relations=grp_colls[(0,0,0,0)],
                      save_as="normal_25+25_dvm_C00.png",
                      )
model.plot_collisions(relations=grp_colls[(1,1,1,1)],
                      save_as="normal_25+25_dvm_C11.png",
                      )
model.plot_collisions(relations=grp_colls[(0,0,1,1)],
                      save_as="normal_25+25_dvm_C01.png",
                      )

print("2d, normal only-intra-species-collisions model")
new_model = bp.CollisionModel(masses=masses,
                              shapes=shapes,
                              base_delta=0.5,
                              spacings=spacings,
                              collision_relations=grp_colls[(0, 0, 1, 1)])
print("Interspecies Model:")
print("number of collision invariants   = ",
      new_model.collision_invariants)
print("maximal physical Collision Invariants = ",
        n_collision_invariants(
            masses,
            model.vels[model.idx_range(0)],
            model.vels[model.idx_range(1)]))
