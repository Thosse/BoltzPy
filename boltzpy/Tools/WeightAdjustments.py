import boltzpy as bp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from time import process_time
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
        assert isinstance(fig, mpl.figure.Figure)
    subfigs = fig.subfigures(nrows=len(grp.keys()),
                             ncols=1)
    if not isinstance(subfigs, np.ndarray):
        subfigs = [subfigs]
    for k, key in enumerate(grp.keys()):
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


class AngularWeightAdjustment(bp.HomogeneousRule):
    def __init__(self,
                 rule,
                 dt,
                 collision_choice=None,
                 initial_weights=(0.2, 4.0),
                 ref_angle_idx=None,
                 hdf5_log=None,
                 max_iterations=100000):
        assert isinstance(rule, bp.HomogeneousRule)
        bp.HomogeneousRule.__init__(self, **rule.__dict__)
        self.rule = rule
        # time step for computation
        assert dt > 0
        self.dt = dt
        # log file for bisection history
        if hdf5_log is None:
            self.log = h5py.File(bp.SIMULATION_DIR + "/_tmp_data.hdf5",
                                 mode="w")
        else:
            assert type(hdf5_log) in [h5py.Group, h5py.File]
        self.setup_log(max_iterations)
        # index to point at current iteration
        self.iter = 0
        # index of the reference angle for the bisection
        if ref_angle_idx is None:
            ref_angle_idx = self.ndim - 1
        assert ref_angle_idx in range(self.ndim)
        self.ref_angle_idx = ref_angle_idx

        # choose collisions
        if collision_choice is None:
            collision_choice = np.arange(self.ncols)
        self.log["collision_choice"] = collision_choice
        # use only the specifies subset of collisions
        self.collision_relations = self.collision_relations[collision_choice]
        self.collision_weights = self.collision_weights[collision_choice]
        # dont work directly on the collisions of rule
        self.collision_relations = np.copy(self.collision_relations)
        self.collision_weights = np.copy(self.collision_weights)
        self.update_collisions()
        # group collisions based on simpülified angles
        simplified_angles = self.key_simplified_angle(normalize=True)
        # project simplified_angles to convex hull of reference angles
        simplified_angles /= simplified_angles[:, -1, None]
        # Base transfer simplified angles
        # into multiples of the Reference-Angle
        _abt_mat = self.angle_base_transfer_matrix
        simplified_angles = np.einsum("ij, kj -> ki",
                                      _abt_mat,
                                      simplified_angles,)
        self.grp_angles = self.group(simplified_angles)
        # compute intitial state and fill log entries
        self.bisect(initial_weights[0], [np.inf, np.inf])
        self.bisect(initial_weights[1])
        if np.any(self.bounds[self.iter] == np.inf):
            print("Bad Inital Values with change of sign")
        assert self.iter == 2
        return

    @property
    def original_weights(self):
        choice = self.log["collision_choice"][()]
        return self.log["all_orignal_weights"][choice]

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
        # Base transfer simplified angles
        # into multiples of the Reference-Angle
        abt_mat = np.linalg.inv(self.reference_angles)
        int_abt_mat = np.array(abt_mat, dtype=int)
        assert np.allclose(abt_mat, int_abt_mat)
        return int_abt_mat

    # Log Entries
    def setup_log(self, max_iterations):
        # store original collisions in log
        self.log["all_orignal_relations"] = self.collision_relations
        self.log["all_orignal_weights"] = self.collision_weights
        # log bisection parameters in h5py arrays
        # all visited weights
        self.log.create_dataset(
            "weights",
            shape=max_iterations,
            maxshape=(max_iterations,))
        # directional viscosities for each weight
        self.log.create_dataset(
            "viscosities",
            shape=(max_iterations, 2),
            maxshape=(max_iterations, 2))
        # the lower and upper bounds at each bisection step
        self.log.create_dataset(
            "bounds",
            shape=(max_iterations, 2),
            maxshape=(max_iterations, 2))
        return

    def resize_log(self):
        for key in ["weights", "viscosities", "bounds"]:
            shape = list(self.log[key].shape)
            shape[0] = self.iter
            self.log[key].resize(shape)
        return

    @property
    def weights(self):
        return self.log["weights"]

    @property
    def viscosities(self):
        return self.log["viscosities"]

    @property
    def bounds(self):
        return self.log["bounds"]

    def rel_errors(self, idx=None):
        if idx is None:
            idx = np.s_[:, :, :]
        visc = self.log["viscosities"][idx]
        return np.max(visc, axis=-1) / np.min(visc, axis=-1) - 1

    def abs_errors(self, idx=None, absolute=True):
        if idx is None:
            idx = np.s_[:, :, :]
        visc = self.log["viscosities"][idx]
        diff = visc[..., 1] - visc[..., 0]
        if absolute:
            return np.abs(diff)
        else:
            return diff

    @property
    def cur_rel_err(self):
        return self.rel_errors(self.iter)

    @property
    def cur_diff(self):
        return self.abs_errors(self.iter, absolute=False)

    @property
    def cur_abs_err(self):
        return self.abs_errors(self.iter)

    @property
    def cur_weight(self):
        return self.weights[self.iter]

    def execute(self, rtol=1e-2, verbose=True, apply_to_rule=True):
        if verbose:
            print("Apply Angular Weight Adjustment")
        tic = process_time()
        while self.cur_rel_err > rtol:
            if verbose:
                print("\ri = %6d"
                      "  -  w = %0.6e "
                      "  -  err = %0.3e"
                      % (self.iter, self.cur_weight, self.cur_rel_err),
                      end="",
                      flush=True)
            self.bisect()
        toc = process_time()
        if verbose:
            print("\nTime taken: %0.3f seconds" % (toc - tic))

        self.resize_log()

        if apply_to_rule:
            cc = self.log["collision_choice"][()]
            self.rule.collision_weights[cc] = self.collision_weights
        return

    def bisect(self, new_weight=None, new_bounds=None):
        if new_weight is None:
            new_weight = np.sum(self.bounds[self.iter]) / 2
        # fill new iterations lof entries
        self.iter += 1
        self.weights[self.iter] = new_weight
        self.simplified_angular_weight_adjustment(new_weight)
        new_visc = self.get_viscosities()
        self.viscosities[self.iter] = new_visc

        if new_bounds is None:
            new_bounds = self.bounds[self.iter - 1]
        assert len(new_bounds) == 2
        if self.cur_diff < 0:
            new_bounds[0] = new_weight
        else:
            new_bounds[1] = new_weight
        self.bounds[self.iter] = new_bounds
        self.log.flush()
        return

    def simplified_angular_weight_adjustment(self,
                                             weight_coefficient,
                                             update_collision_matrix=True):
        # reset collision weights
        self.collision_weights = self.original_weights
        # set up weights of reference angles
        reference_weights = np.ones(self.ndim)
        reference_weights[self.ref_angle_idx] = weight_coefficient
        # apply reference weight to collision groups
        for key, pos in self.grp_angles.items():
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
                    dt=self.dt,
                    normalize=normalize)
                for angle_pair in directions]
