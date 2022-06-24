import os
import numpy as np
import boltzpy as bp
import matplotlib.pyplot as plt
from time import process_time
from boltzpy.Tools.fonts import fs_title, fs_legend, fs_label, fs_suptitle, fs_ticks

# Use default color cycle from matplotlib
_prop_cycle = plt.rcParams['axes.prop_cycle']
_colors = _prop_cycle.by_key()['color']


class GainGroup:
    def __init__(self, rule, class_key, grp):
        assert isinstance(rule, bp.HomogeneousRule)
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

        # compute gains of each collision group,
        # use uniform weights for balancing
        ori_weights = np.copy(rule.collision_weights)
        rule.collision_weights[:] = 1.0
        rule.update_collisions(weights= rule.collision_weights)
        self.gains = np.zeros(self.probabilities.shape, dtype=float)
        for k, key in enumerate(self.keys):
            gain_array = rule.gain_array(grp[key])
            self.gains[k] = rule.cmp_number_density(gain_array)

        # store chosen collision groups
        self.choice = np.zeros(self.probabilities.shape, dtype=bool)
        # store gain of current collision relations
        self.current_gain = 0.0
        self.unused_grps = self.probabilities.size
        # restore original weights of rule
        rule.collision_weights[:] = ori_weights
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


class GainBasedModelReduction(bp.HomogeneousRule):
    def __init__(self,
                 balance_keys,
                 selection_keys,
                 execute=True,
                 force_normality_collisions=True,
                 gain_factor_normality_collisions=1.0,
                 **rule_params):
        bp.HomogeneousRule.__init__(self, **rule_params)
        for k in [balance_keys, selection_keys]:
            assert k.ndim == 2
            assert k.dtype == int
        assert balance_keys.shape[0] == selection_keys.shape[0]
        self.len_class = balance_keys.shape[1]
        self.len_sub = selection_keys.shape[0]
        self.keys = self.merge_keys(balance_keys, selection_keys)
        self.col_grp = self.group(self.keys)

        # construct array of gain groups
        self.unique_keys = self.filter(balance_keys, balance_keys)
        self.unique_keys = [tuple(key) for key in self.unique_keys]
        self.gain_groups = [GainGroup(self, key, self.col_grp)
                            for key in self.unique_keys]
        self.gain_groups = np.array(self.gain_groups)

        # log added keys, gains, and ncols
        self.log_keys = []
        self.log_gains = [[0 for k in self.unique_keys]]
        self.log_ncols = [0]
        self.log_empty_times = {key: -1 for key in self.unique_keys}
        if execute:
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

    def get_selection(self, idx):
        keys = self.log_keys[:idx]
        col_grps = [self.col_grp[key] for key in keys]
        col_rels = np.concatenate(col_grps, axis=0)
        return col_rels

    def add_collisions(self, key=None, gain_factor=1.0):
        if key is None:
            gg = self.next_gain_group()
            key = gg.choose(gain_factor=gain_factor)
            if gg.unused_grps == 0:
                self.log_empty_times[gg.class_key] = len(self.log_keys) - 1
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
        tic = process_time()
        if force_normality_collisions:
            print("Add Normality Collisions...", end="", flush=True)
            col_rels = self.collision_relations
            is_required = self.key_is_normality_collision(col_rels)
            required_keys = self.keys[np.where(is_required)[0]]
            required_keys = self.filter(required_keys, required_keys)
            for key in required_keys:
                self.add_collisions(key,
                                    gain_factor=gain_factor_normality_collisions)
            print("Done!")

        print("Add Collisions to Balance Gains...", end="", flush=True)
        while self.log_ncols[-1] != self.ncols:
            print("\rCollisions = %7d / %7d              "
                  % (self.log_ncols[-1], self.ncols),
                  flush=True,
                  end="")
            self.add_collisions()
        toc = process_time()
        print("\nTime taken: %.3f seconds" % (toc-tic))
        return

    def plot_reduction(
            self,
            figsize=(12.75, 6.25),
            constrained_layout=True,
            file_address=None,
            class_keys=None,
            legend_ncol=1,
            legend_title=None,
            lw=2,
            lw_vlines=0.5,
            ls="solid",
            xlabel="Number of Collisions",
            ylabel="Individual Gains",
            title="Weight- and Gain-Based Collision Reduction",
            xscale="linear",
            yscale="linear",
            ymin=1e-6,
            ymax=None):
        fig = plt.figure(figsize=figsize,
                         constrained_layout=constrained_layout)
        ax = fig.add_subplot()
        if file_address is not None:
            assert isinstance(file_address, str)
        if class_keys is None:
            class_keys = self.class_keys
        # convert gains  to array for easier access
        gains = np.array(self.log_gains)
        if ymax is None:
            ymax = 1.1 * gains[-1].max()
        # plot individual gains
        for k, key in enumerate(class_keys):
            color = _colors[k]
            ax.plot(self.log_ncols,
                    gains[:, k],
                    c=color,
                    ls=ls,
                    lw=lw,
                    label=key)
            # add vlines to mark good reductions
            last_idx = self.log_empty_times[key]
            last_ncols = self.log_ncols[last_idx]
            ax.vlines(last_ncols,
                      ymin, ymax,
                      lw=lw_vlines,
                      ls="dotted",
                      colors=color,
                      zorder=-1)
        ax.set_ylim(ymin, ymax)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        leg = plt.legend(title=legend_title,
                         title_fontsize=fs_legend,
                         fontsize=fs_legend,
                         ncol=legend_ncol)
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)
        ax.set_title(title,
                     fontsize=fs_title)
        ax.tick_params(axis="both", labelsize=fs_ticks)
        ax.set_xlabel(xlabel, fontsize=fs_label)
        ax.set_ylabel(ylabel, fontsize=fs_label)

        if file_address is not None:
            plt.savefig(file_address)
        else:
            plt.show()
        return
