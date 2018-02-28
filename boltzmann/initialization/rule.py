import numpy as np


class Rule:
    """Base class for all Rules.
    Encapsulates basic information for initialization and categorization.

    Initializes the velocity space of each specimen
    based on their conserved quantities
    mass (:attr:`rho`),
    mean velocity (:attr:`drift`)
    and temperature (:attr:`temp`).


    Parameters
    ----------
    cat : int
        Specifies the behavior in the calculation.
        Index of an Element in
        :const:`~RuleArray.SPECIFIED_CATEGORIES`.
    rho_list : array_like
        List of the rho-parameters for each specimen.
        Rho correlates to the total weight/amount of particles in
        the area of the P-Grid point.
    drift_list : array_like
        List of the drift-parameters for each specimen.
        Drift describes the mean velocity.
    temp_list : array_like
        List of the temp-parameters for each specimen.
        Temp describes the Temperature.
    name : str, optional
        Sets a name, for the points initialized with this rule.

    Attributes
    ----------
    cat : int
    name : str
    rho : array
    drift : array
    temp : array
    """

    def __init__(self,
                 cat,
                 rho_list,
                 drift_list,
                 temp_list,
                 name=''):
        assert len(rho_list) is len(drift_list)
        assert len(rho_list) is len(temp_list)
        assert all([len(drift) in [2, 3]
                    and len(drift) is len(drift_list[0])
                    for drift in drift_list])
        self.cat = cat
        self.name = name
        self.rho = np.array(rho_list)
        self.drift = np.array(drift_list)
        self.temp = np.array(temp_list)
        return

    def check_integrity(self):
        assert len(self.rho.shape) is 1
        assert len(self.drift.shape) is 2
        assert len(self.temp.shape) is 1
        n_species = self.rho.shape[0]
        dim_p = self.drift.shape[1]
        assert dim_p in [1, 2, 3]
        assert self.rho.shape == (n_species,)
        assert self.drift.shape == (n_species, dim_p)
        assert self.temp.shape == (n_species,)
        assert self.cat in [0, 1, 2, 3]
        return

    def print(self,
              list_of_category_names=None):
        print('Name of Rule = {}'.format(self.name))
        if list_of_category_names is not None:
            print('Category = {}'
                  ''.format(list_of_category_names[self.cat]))
        print('Category index = {}'.format(self.cat))
        print('Rho: '
              '{}'.format(self.rho))
        print('Drift:\n'
              '{}'.format(self.drift))
        print('Temperature: '
              '{}'.format(self.temp))
        print('')
        return


# Todo is an extra class useful for inner points?
# class InnerPointRule(Rule):
#     def __init__(self,
#                  rho_list,
#                  drift_list,
#                  temp_list,
#                  name=''):
#         Rule.__init__(self,
#                       0,
#                       rho_list,
#                       drift_list,
#                       temp_list,
#                       name)
#         return
