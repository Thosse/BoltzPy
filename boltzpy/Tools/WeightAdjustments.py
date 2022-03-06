import boltzpy as bp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import process_time
import h5py


class WeightAdjustment(bp.HomogeneousRule):
    def __init__(self, **rule_params):
        bp.HomogeneousRule.__init__(self, **rule_params)
        return

    def cmp_grp_gains(self,
                      grp_cols=None,
                      species=None,
                      as_array=False,
                      resolve_spacings=False,
                      initial_state=None):
        if grp_cols is None:
            k_spc = self.key_species(self.collision_relations)
            grp_cols = self.group(k_spc)
            del k_spc
        if species is None:
            species = self.species
        species = np.array(species, ndmin=1, copy=False)
        assert species.ndim == 1

        gains = dict()
        for key, pos in grp_cols.items():
            gain_arr = self.gain_array(pos,
                                       initial_state=initial_state)
            if resolve_spacings:
                gain_arr *= self.get_array(self.dv ** self.ndim)
            if not as_array:
                gains[key] = self.cmp_number_density(gain_arr,
                                                     separately=True)
            else:
                gains[key] = np.empty(species.size, dtype=object)
                for s in species:
                    spc_gain = gain_arr[self.idx_range(s)]
                    gains[key][s] = spc_gain.reshape(self.shapes[s])
        return gains

    def balance_gains(self,
                      grp_cols,
                      gain_ratios,
                      initial_state=None,
                      verbose=True,
                      update_collision_matrix=True):
        assert isinstance(gain_ratios, dict)
        assert set(grp_cols.keys()) == set(gain_ratios.keys())

        # normalize gain_ratios to retain
        # original total number density gain
        gains = self.cmp_grp_gains(grp_cols,
                                   initial_state=initial_state)
        # use total gain (not specific gain)
        gains = {key: sum(val) for key, val in gains.items()}
        norm = sum(gains.values())
        norm /= sum(gain_ratios.values())

        # compute max string length of keys for printing
        maxlen = max([len(str(k)) for k in grp_cols.keys()])

        # balance weights
        factors = dict()
        for key, pos in grp_cols.items():
            # apply normalized weight adjustment factor
            factors[key] = gain_ratios[key] * norm / gains[key]
            self.collision_weights[pos] *= factors[key]
            if verbose:
                # print key  : factor aligned as a table
                print("{0:<{1}}: ".format(str(key), maxlen), factors[key])

        if update_collision_matrix:
            self.update_collisions()
        return factors

    def plot_gains(self,
                   grp_cols=None,
                   species=None,
                   initial_state=None,
                   file_address=None,
                   fig=None,
                   figsize=None,
                   constrained_layout=True,
                   vmax=None):
        assert self.ndim == 2
        if species is None:
            species = self.species
        if grp_cols is None:
            k_spc = self.key_species(self.collision_relations)
            grp_cols = self.group(k_spc)
            del k_spc
        if initial_state is None:
            initial_state = self.initial_state

        if file_address is not None:
            assert isinstance(file_address, str)

        # compute all gains as array
        gains = self.cmp_grp_gains(grp_cols,
                                   species,
                                   as_array=True,
                                   resolve_spacings=True,
                                   initial_state=initial_state)
        if vmax is None:
            vmax = max(max(np.max(gains[k][s])
                           for s in species)
                       for k in gains.keys()
                       )

        # create Figure, with subfigures
        if fig is None:
            fig = plt.figure(figsize=figsize,
                             constrained_layout=constrained_layout)
        else:
            assert isinstance(fig, mpl.figure.Figure)
        subfigs = fig.subfigures(nrows=len(grp_cols.keys()),
                                 ncols=1)
        if not isinstance(subfigs, np.ndarray):
            subfigs = [subfigs]
        for k, key in enumerate(grp_cols.keys()):
            axes = subfigs[k].subplots(nrows=1, ncols=len(species))
            for s in species:
                axes[s].imshow(gains[key][s],
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


class AngularWeightAdjustment(WeightAdjustment):
    def __init__(self,
                 hdf5_log=None,
                 **rule_params):
        WeightAdjustment.__init__(self, **rule_params)
        # keep track of important simulation parameters
        # is properly initialized in each execution
        self.cur_rels_used = np.empty((0,), dtype=int)
        self.cur_rels_adj = np.empty((0,), dtype=int)
        self.cur_grp = dict()
        self.cur_dt = np.inf
        self.cur_adjustment = -1
        self.cur_iter = -1
        self.cur_ref_angle = -1

        # log file for bisection history
        if hdf5_log is None:
            self.log = h5py.File(bp.SIMULATION_DIR + "/_tmp_data.hdf5",
                                 mode="w")
        else:
            assert type(hdf5_log) in [h5py.Group, h5py.File]
        # store original collisions in h5py log
        self.log["orignal_relations"] = self.collision_relations
        self.log["orignal_weights"] = self.collision_weights
        self.log["adjusted_weights"] = self.collision_weights

        return

    @property
    def cur_unused_cols(self):
        unused = np.array([i for i in range(self.ncols)
                           if i not in self.cur_cols_used],
                          dtype=int)
        return unused

    @property
    def cur_unadj_cols(self):
        unused = np.array([i for i in range(self.ncols)
                           if i not in self.cur_cols_adj],
                          dtype=int)
        return unused

    @property
    def reference_angles(self):
        ref_angle_mat = np.array([[1, 0, 0],
                                  [1, 1, 0],
                                  [1, 1, 1]])
        if self.ndim == 3:
            return ref_angle_mat
        if self.ndim == 2:
            return ref_angle_mat[1:, 1:]
        else:
            raise ValueError

    @property
    def angle_base_transfer_matrix(self):
        # Base transfer for simplified angles
        # Convert into multiples of the Reference-Angles
        abt_mat = np.linalg.inv(self.reference_angles)
        int_abt_mat = np.array(abt_mat, dtype=int)
        assert np.allclose(abt_mat, int_abt_mat)
        return int_abt_mat

    #################################
    #       Log Related Members     #
    #################################
    @property
    def cur_log(self):
        return self.log[str(self.cur_adjustment)]

    def _init_adjustment(self,
                         dt,
                         cols_adj=None,
                         cols_used=None,
                         initial_weights=(0.2, 4.0),
                         maxiter=100000,
                         ref_angle_idx=-1):
        self.cur_adjustment += 1
        self.cur_iter = -1
        assert dt > 0
        self.cur_dt = dt

        # log bisection parameters in h5py arrays
        # to allow posterior investigations
        self.log.create_group(str(self.cur_adjustment))

        # by default use and adjust all collisions
        if cols_adj is None:
            cols_adj = np.arange(self.ncols)
        self.cur_cols_adj = cols_adj
        self.cur_log["cols_adj"] = self.cur_cols_adj
        if cols_used is None:
            cols_used = np.arange(self.ncols)
        self.cur_cols_used = cols_used
        self.cur_log["cols_used"] = self.cur_cols_used
        # apply collision choice
        self.collision_weights = self.log["adjusted_weights"][()]
        self.collision_weights[self.cur_unused_cols] = 0
        # log weights (adjusted ones are edited after execution)
        self.cur_log["original_weights"] = self.collision_weights
        self.cur_log["adjusted_weights"] = self.collision_weights
        # group to be adjusted collisions
        self.cur_grp = self._cmp_grp_angles()
        self.ref_angle_idx = ref_angle_idx

        # log all visited weights
        self.cur_log.create_dataset(
            "weights",
            shape=maxiter,
            maxshape=(maxiter,))
        self.cur_log["weights"][0:2] = initial_weights
        # directional viscosities for each weight
        self.cur_log.create_dataset(
            "viscosities",
            shape=(maxiter, 2),
            maxshape=(maxiter, 2))
        # the lower and upper bounds at each bisection step
        self.cur_log.create_dataset(
            "bounds",
            shape=(maxiter, 2),
            maxshape=(maxiter, 2))
        return

    def log_results(self):
        for key in ["weights", "viscosities", "bounds"]:
            shape = list(self.cur_log[key].shape)
            shape[0] = self.cur_iter + 1
            self.cur_log[key].resize(shape)
        adj_weights = self.collision_weights[self.cur_cols_adj]
        self.cur_log["adjusted_weights"][self.cur_cols_adj] = adj_weights
        self.log["adjusted_weights"][self.cur_cols_adj] = adj_weights
        return

    @property
    def original_collisions(self):
        return self.log["original_collisions"]

    def original_weights(self, adjustment_idx):
        if adjustment_idx == -1:
            return self.log["original_weights"][()]
        else:
            sub_log = self.log[str(adjustment_idx)]
            return sub_log["original_weights"][()]

    def adjusted_weights(self, adjustment_idx):
        if adjustment_idx == -1:
            return self.log["adjusted_weights"]
        else:
            sub_log = self.log[str(adjustment_idx)]
            return sub_log["adjusted_weights"]

    @property
    def cur_weights(self):
        return self.cur_log["weights"]

    @property
    def cur_viscs(self):
        return self.cur_log["viscosities"]

    @property
    def cur_bounds(self):
        return self.cur_log["bounds"]

    def rel_errors(self, idx=None):
        if idx is None:
            idx = np.s_[:, :, :]
        visc = self.cur_log["viscosities"][idx]
        return np.max(visc, axis=-1) / np.min(visc, axis=-1) - 1

    def abs_errors(self, idx=None, absolute=True):
        if idx is None:
            idx = np.s_[:, :, :]
        visc = self.cur_log["viscosities"][idx]
        diff = visc[..., 1] - visc[..., 0]
        if absolute:
            return np.abs(diff)
        else:
            return diff

    @property
    def cur_rel_err(self):
        return self.rel_errors(self.cur_iter)

    @property
    def cur_diff(self):
        return self.abs_errors(self.cur_iter, absolute=False)

    @property
    def cur_abs_err(self):
        return self.abs_errors(self.cur_iter)

    @property
    def cur_weight(self):
        return self.cur_weights[self.cur_iter]

    def _cmp_grp_angles(self):
        simplified_angles = self.key_simplified_angle(normalize=True)
        # Base transfer simplified angles
        # into multiples of the Reference-Angle
        _abt_mat = self.angle_base_transfer_matrix
        simplified_angles = np.einsum("ij, kj -> ki",
                                      _abt_mat,
                                      simplified_angles,)
        # ignoew used but unadjusted collisions
        ign = self.cur_unadj_cols
        simplified_angles[ign] = -1
        grp_angles = self.group(simplified_angles)
        ign_key = (-1,) * self.ndim
        if ign_key in grp_angles.keys():
            del grp_angles[ign_key]
        return grp_angles

    def execute(self,
                dt,
                cols_adj=None,
                cols_used=None,
                initial_weights=(0.2, 4.0),
                maxiter=100000,
                ref_angle_idx=-1,
                rtol=1e-2,
                verbose=True):
        if verbose:
            print("Initialize Adjustment Parameters and Log-File")
        self._init_adjustment(dt, cols_adj, cols_used,
                              initial_weights, maxiter,
                              ref_angle_idx)

        if verbose:
            print("Compute Initial Viscosities as Upper/Lower Bounds")
        assert self.cur_iter == -1
        self.bisect(self.cur_weights[0], [np.inf, np.inf])
        self.bisect(self.cur_weights[1])
        if np.any(self.cur_bounds[self.cur_iter] == np.inf):
            raise ValueError("Bad Inital Values with no change of sign")
        assert self.cur_iter == 1

        if verbose:
            print("Execute Bisection Algorithm")
        tic = process_time()
        while self.cur_rel_err > rtol:
            self.bisect()
            if verbose:
                print("\ri = %6d"
                      "  -  w = %0.6e "
                      "  -  err = %0.3e"
                      % (self.cur_iter, self.cur_weight, self.cur_rel_err),
                      end="",
                      flush=True)
        toc = process_time()
        if verbose:
            print("\nTime taken: %0.3f seconds" % (toc - tic))

        self.log_results()
        return

    def bisect(self, new_weight=None, new_bounds=None):
        if new_weight is None:
            new_weight = np.sum(self.cur_bounds[self.cur_iter]) / 2
        # fill new iterations lof entries
        self.cur_iter += 1
        self.cur_weights[self.cur_iter] = new_weight
        self.simplified_angular_weight_adjustment(new_weight)
        new_visc = self.get_viscosities()
        self.cur_viscs[self.cur_iter] = new_visc

        if new_bounds is None:
            new_bounds = self.cur_bounds[self.cur_iter - 1]
        assert len(new_bounds) == 2
        if self.cur_diff < 0:
            new_bounds[0] = new_weight
        else:
            new_bounds[1] = new_weight
        self.cur_bounds[self.cur_iter] = new_bounds
        self.log.flush()
        return

    def simplified_angular_weight_adjustment(self,
                                             weight_coefficient,
                                             update_collision_matrix=True):
        # reset collision weights
        self.collision_weights[:] = self.original_weights(self.cur_adjustment)
        # set up weights of reference angles
        reference_weights = np.ones(self.ndim)
        reference_weights[self.ref_angle_idx] = weight_coefficient
        # apply reference weight to collision groups
        for key, pos in self.cur_grp.items():
            # interpolate weights based on reference angles
            factor = np.dot(key, reference_weights)
            self.collision_weights[pos] *= factor

        if update_collision_matrix:
            self.update_collisions()
        return self.collision_weights

    def get_viscosities(self, normalize=False):
        directions = np.array([[[0, 1, 1], [0, 1, -1]],
                               [[0, 0, 1], [0, 1, 0]]])
        if self.ndim == 2:
            directions = directions[:, :, 1:]
        elif self.ndim != 3:
            raise AttributeError
        return [self.cmp_viscosity(
                    directions=angle_pair,
                    dt=self.cur_dt,
                    normalize=normalize)
                for angle_pair in directions]
