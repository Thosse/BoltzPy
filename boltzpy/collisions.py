
import numpy as np
import boltzpy as bp


# Any single relation can be permutated without changing its effect
INTRASPECIES_PERMUTATION = np.array([[0, 1, 2, 3], [1, 2, 3, 0],
                                     [2, 3, 0, 1], [3, 0, 1, 2],
                                     [3, 2, 1, 0], [0, 3, 2, 1],
                                     [1, 0, 3, 2], [2, 1, 0, 3]],
                                    dtype=int)


INTERSPECIES_PERMUTATION = np.array([[0, 1, 2, 3],
                                     [2, 3, 0, 1],
                                     [3, 2, 1, 0],
                                     [1, 0, 3, 2]],
                                    dtype=int)


    #
    #
    # #####################################
    # #           Verification            #
    # #####################################
    # def check_integrity(self, context=None):
    #     """Sanity Check"""
    #     if context is not None:
    #         assert isinstance(context, bp.Simulation)
    #     if self.relations is not None or self.weights is not None:
    #         assert self.relations is not None and self.weights is not None
    #         assert isinstance(self.relations, np.ndarray)
    #         assert isinstance(self.weights, np.ndarray)
    #         assert self.relations.dtype == int
    #         assert self.weights.dtype == float
    #         assert self.relations.ndim == 2
    #         assert self.weights.ndim == 1
    #         assert self.relations.shape == (self.weights.size, 4)
    #         for col in self.relations:
    #             # assert col[0] < col[1]
    #             # assert col[0] < col[2]
    #             if context is not None:
    #                 model = context.model
    #                 di_0 = (model.iMG[col[1]] - model.iMG[col[0]])
    #                 di_1 = (model.iMG[col[3]] - model.iMG[col[2]])
    #                 assert all(np.array(di_1 + di_0) == 0)
    #                 s = model.get_spc(col)
    #                 assert s[0] == s[1] and s[2] == s[3]
    #                 # Todo add conserves energy check
    #         assert all(w > 0 for w in self.weights.flatten())
    #     return



    # def issubset(self, other):
    #     """Checks if self.relations are a subset of other.relations.
    #     Weights are not checked and may differ.
    #
    #     Parameters
    #     ----------
    #     other : :class:`Collisions`
    #
    #     Returns
    #     -------
    #     :obj:`bool`
    #     """
    #     assert isinstance(other, Collisions)
    #     # group/filter by index
    #     grp_self = self.group(mode="index")
    #     grp_other = other.group(mode="index")
    #     # assert that each keys has only a single value
    #     assert all([len(value) == 1 for value in grp_self.values()])
    #     assert all([len(value) == 1 for value in grp_other.values()])
    #     # use set of keys to check for subset relationship
    #     set_self = set(grp_self.keys())
    #     set_other = set(grp_other.keys())
    #     return set_self.issubset(set_other)
