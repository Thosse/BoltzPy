import boltzpy as bp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import h5py

def balance_gains(homogeneous_rule,
                  grp,
                  gain_ratios,
                  verbose=False,
                  update_collision_matrix=True):
    assert isinstance(homogeneous_rule, bp.HomogeneousRule)
    r = homogeneous_rule
    ncols = sum([v.shape[0] for v in grp.values()])
    assert ncols == r.ncols
    assert isinstance(gain_ratios, dict)
    assert set(grp.keys()) == set(gain_ratios.keys())
    # normalize gain_ratios to retain original total gain
    norm = r.cmp_number_density(r.gain_array())
    norm /= np.sum(list(gain_ratios.values()))
    # compute max sring length of keys for printing
    maxlen = max([len(str(k)) for k in grp.keys()])
    # Compute and apply weight adjustment factors:
    factors = dict()
    for key, rels in grp.items():
        gain_array = r.gain_array(grp[key])
        gain = r.cmp_number_density(gain_array)
        # apply normalized weight adjustment factor
        factors[key] = gain_ratios[key] * norm / gain
        r.collision_weights[rels] *= factors[key]
        if verbose:
            # print key  : factor aligned as a table
            print("{0:<{1}}: ".format(str(key), maxlen), factors[key])

    if update_collision_matrix:
        r.update_collisions()
    print(r.cmp_number_density(r.gain_array()))
    return factors

def plot_gains(self,
               species=None,
               grp=None,
               file_address=None,
               fig=None,
               figsize=None,
               constrained_layout=True,
               vmax=None):
    assert isinstance(self, bp.HomogeneousRule)
    if species is None:
        species = self.species

    if grp is None:
        k_spc = self.key_species(self.collision_relations)
        k_et = self.key_energy_transfer(self.collision_relations)
        grp = self.group((k_spc, k_et))
        del k_spc, k_et

    if file_address is not None:
        assert isinstance(file_address, str)

    # compute all gains
    gains = np.empty((len(grp.keys()), len(species)),
                     dtype=object)
    for k, (key, cols) in enumerate(grp.items()):
        gain_arr = self.gain_array(cols)
        gain_arr *= self.get_array(self.dv)
        for s in species:
            gains[k, s] = gain_arr[self.idx_range(s)]
            gains[k, s] = gains[k, s].reshape(self.shapes[s])
    if vmax is None:
        vmax = max(max(np.max(gains[k,s]) for s in species)
                   for k in range(len(grp.keys())))

    # create Figure, with subfigures
    if fig is None:
        fig = plt.figure(figsize=figsize,
                         constrained_layout=constrained_layout)
    else:
        assert isinstance(fig. mpl.figure.Figure())
    subfigs = fig.subfigures(nrows=len(grp.keys()),
                             ncols=1)
    for k, (key, cols) in enumerate(grp.items()):
        axes = subfigs[k].subplots(nrows=1, ncols=len(species))
        for s in species:
            axes[s].imshow(gains[k, s],
                           cmap='coolwarm',
                           interpolation="quadric",
                           origin="lower",
                           vmin=-vmax,
                           vmax=vmax)
            axes[s].set_xticks([])
            axes[s].set_yticks([])
    if file_address is not None:
        plt.savefig(file_address)
    else:
        plt.show()
    return





# def cmp_visc(i, normalize=False):
#     result = np.empty(m[i].ndim)
#     if m[i].ndim == 3:
#         result[0] = m[i].cmp_viscosity(
#             number_densities=nd[i],
#             temperature=temp[i],
#             directions=[[1, 1, 1], [0, 1, -1]],
#             dt=dt,
#             normalize=normalize)
#         result[1] = m[i].cmp_viscosity(
#             number_densities=nd[i],
#             temperature=temp[i],
#             directions=[[0, 1, 1], [0, 1, -1]],
#             dt=dt,
#             normalize=normalize)
#         result[2] = m[i].cmp_viscosity(
#             number_densities=nd[i],
#             temperature=temp[i],
#             directions=[[0, 0, 1], [1, 0, 0]],
#             dt=dt,
#             normalize=normalize)
#
#     elif m[i].ndim == 2:
#         result[0] = m[i].cmp_viscosity(
#             number_densities=nd[i],
#             temperature=temp[i],
#             directions=[[1, 1], [1, -1]],
#             dt=dt,
#             normalize=normalize)
#         result[1] = m[i].cmp_viscosity(
#             number_densities=nd[i],
#             temperature=temp[i],
#             directions=[[0, 1], [1, 0]],
#             dt=dt,
#             normalize=normalize)
#
#     return result

#
# def enforce_angular_invariance(self,
#                                hdf5_log=None,
#                                cols=None,
#                                initial_weights=[0.1, 3.0],
#                                rtol=1e-3,
#                                maxiter=1000):
#     assert isinstance(self, bp.HomogeneousRule)
#     if hdf5_log is None:
#         hdf5_log = h5py.File(bp.SIMULATION_DIR + "/_tmp_data.hdf5",
#                                mode="w")
#     assert isinstance(hdf5_log, h5py.Group)
#     # store original collisions in log
#     hdf5_log["orignal_relations"] = self.collision_relations
#     hdf5_log["orignal_weights"] = self.collision_weights
#     # apply collision subset
#     if cols is None:
#         cols = np.s_[::]    # slice over all elements
#     self.collision_relations = self.collision_relations[cols]
#     self.collision_weights = self.collision_weights[cols]
#     self.update_collisions()
#     # TODO store logs in maxlength array
#     log_viscosities = hdf5_log.create_dataset("log_viscosities",
#                                               shape=(maxiter, 2))
#     log_weights = hdf5_log.create_dataset("log_viscosity",
#                                             shape=(maxiter, self.ndim))
#     # compute visocities for intitial weights
#     for w in biscetion_weights:
#         m[ndim].collision_weights[:] = DEFAULT_WEIGHT
#         angular_weights[used_weight] = w
#         adjust_weight_by_angle(m[ndim], angular_weights)
#         visc.append(cmp_visc(ndim))
#
#
#
#     print("\nSetup hdf groups")
#     subgrp = hdf_group.create_group(str(ndim))
#     subgrp["viscosities"] = np.zeros((NUMBER_DENSITIES.shape[0], ndim))
#
#     # print("Load {}D model...".format(ndim))
#     # model_group = file["bisection_3d_pairwise"][str(ndim)]["model"]
#     # model = bp.CollisionModel.load(model_group)
#     spc_keys = m[ndim].key_species(m[ndim].collision_relations)[:, 1:3]
#     spc_grp = m[ndim].group(spc_keys)
#     spc_pairs = [(0, 0), (1, 1), (0, 1)]
#     for pair in spc_pairs:
#         affected_collisions = spc_grp[pair]
#         weight_group = file["bisection_3d_pairwise"][str(ndim)][str(pair)]
#         angular_weights = np.ones(ndim)
#         angular_weights[ndim - 1] = weight_group["bisection_weights"][-1]
#         weight_factors = m[ndim].simplified_angular_weight_adjustment(
#             angular_weights,
#             m[ndim].collision_relations[affected_collisions])
#         adjusted_weights = (weight_factors * DEFAULT_WEIGHT)
#         m[ndim].collision_weights[affected_collisions] = adjusted_weights
#     m[ndim].update_collisions(m[ndim].collision_relations,
#                               m[ndim].collision_weights)
#     print("Compute Viscosites for Altered Number Densities")
#     for i_nd, _nd in enumerate(NUMBER_DENSITIES):
#         print("\r%3d / %3d"
#               % (i_nd, NUMBER_DENSITIES.shape[0]),
#               end="")
#         subgrp["viscosities"][i_nd] = cmp_visc_ext(
#             m[ndim],
#             _nd,
#             temp[ndim]
#         )
#         file.flush()