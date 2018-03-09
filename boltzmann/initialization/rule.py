
import numpy as np


class Rule:
    """Base class for all Rules.
    Encapsulates basic information for initialization and categorization.

    A :obj:`Rule` can be applied on any
    P-:class:`~boltzmann.configuration.Grid` point p,
    where it initializes the velocity space of each specimen
    based on their conserved quantities
    mass (:attr:`rho`),
    mean velocity (:attr:`drift`)
    and temperature (:attr:`temp`).
    Furthermore the behaviour of p during
    :class:`~boltzmann.calculation.Calculation`
    depends on its Category
    (see :attr:`cat` and :attr:`Initialization.supported_categories`).


    Parameters
    ----------
    cat : int
    rho_list : :obj:`array` or :obj:`list`
        Is converted into :class:`~numpy.ndarray` :attr:`rho`.
    drift_list : :obj:`array` or :obj:`list`
        Is converted into :class:`~numpy.ndarray` :attr:`drift`.
    temp_list : :obj:`array` or :obj:`list`
        Is converted into :class:`~numpy.ndarray` :attr:`temp`.
    name : str, optional
    """

    def __init__(self,
                 cat,
                 rho_list,
                 drift_list,
                 temp_list,
                 name=''):
        self._cat = cat
        self._name = name
        self._rho = np.array(rho_list)
        self._drift = np.array(drift_list)
        self._temp = np.array(temp_list)
        assert self.rho.shape == self.temp.shape == self.drift.shape[0:-1]
        assert self.drift.shape[-1] in [2, 3]
        self.check_integrity()
        return

    @property
    def cat(self):
        """:obj:`int`:
        Specifies the behavior in the
        :class:`~boltzmann.calculation.Calculation`.
        Index of an element in
        :attr:`Initialization.supported_categories`.
        """
        return self._cat

    @property
    def name(self):
        """:obj:`str`:
        Sets a name to this :obj:`Rule` and the
        P-:class:`~boltzmann.configuration.Grid` points
        on which it's applied.
        """
        return self._name

    @property
    def rho(self):
        """:obj:`~numpy.ndarray`:
        Array of the rho parameters for each specimen.
        Rho correlates to the total weight/amount of particles in
        the area of the
        P-:class:`~boltzmann.configuration.Grid` point.
        """
        return self._rho

    @property
    def drift(self):
        """:obj:`~numpy.ndarray`:
        Array of the drift parameters for each specimen.
        Drift describes the mean velocity,
        i.e. the first moment (expectancy value) of the
        velocity distribution.
        """
        return self._drift

    # Todo get a clear understanding of the meaning of temperature
    @property
    def temp(self):
        """:obj:`~numpy.ndarray`:
        Array of the temperature parameters for each specimen.
        Temp describes the Temperature,
        i.e. the second moment (variance) of the
        velocity distribution.
        """
        return self._temp

    def check_integrity(self):
        """Sanity Check"""
        assert type(self.cat) is int
        assert self.cat >= 0
        assert len(self.rho.shape) is 1
        assert len(self.drift.shape) is 2
        assert len(self.temp.shape) is 1
        n_species = self.rho.shape[0]
        dim_p = self.drift.shape[1]
        assert dim_p in [1, 2, 3]
        assert self.rho.shape == (n_species,)
        assert self.drift.shape == (n_species, dim_p)
        assert self.temp.shape == (n_species,)
        assert np.min(self.rho) > 0
        assert np.min(self.temp) > 0
        assert type(self.name) is str
        return

    def print(self, list_of_category_names=None):
        """Prints all Properties for Debugging Purposes

        If a list of category names is given,
        then the category of the :class:`Rule` is printed.
        Otherwise the category index
        (:attr:`cat`) is printed"""
        print('Name of Rule = {}'.format(self.name))
        if list_of_category_names is not None:
            print('Category = {}'
                  ''.format(list_of_category_names[self.cat]))
        else:
            print('Category index = {}'.format(self.cat))
        print('Rho: '
              '{}'.format(self.rho))
        print('Drift:\n'
              '{}'.format(self.drift))
        print('Temperature: '
              '{}'.format(self.temp))
        print('')
        return
