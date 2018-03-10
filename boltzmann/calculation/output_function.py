import numpy as np


class OutputFunction:
    """Processes and stores results for animation.

    The class provides the
    :meth:`apply` method which iteratively calculates the specified moments
    (see :attr:`~boltzmann.configuration.Configuration.animated_moments`)
    by applying the respective functions in
    :attr:`f_arr` on the current
    :attr:`~boltzmann.calculation.Calculation.data` PSV-Grid.
    The results are currently returned.
    In the near future they will be logged/appended to a file.

    .. todo::
      - choose if Positional grid should be dimensional?
      - implement rest of moment functions
      - figure out if using self.cnf in f_mass(psv)
        leads to the Configuration never being deleted from memory

    Parameters
    ----------
    cnf : :class:`~boltzmann.configuration.Configuration`
    """
    def __init__(self,
                 cnf):
        self._cnf = cnf
        self._f_arr = self.create_function_list()
        return

    @property
    def f_arr(self):
        """:obj:`list` of :obj:`function`:
        List of moment generating functions."""
        return self._f_arr

    # Todo Implement write to file routine
    # @property
    # def files(self):
    #     """:obj:`list` of :obj:`function`:
    #         files : list(str)
    #     files[i] is the name of the file, to which the results
    #     of f_arr[i] are appended.
    #     List of moment generating functions."""
    #     return self._files

    def create_function_list(self):
        """Generates the list of moment generating functions"""
        f_arr = []
        for mom in self._cnf.animated_moments:
            if mom is 'Mass':
                f = self._get_f_mass()
            elif mom is 'Mass_Flow':
                f = self._get_f_mass_flow()
            elif mom is 'Momentum':
                f = self._get_f_momentum()
            elif mom is 'Momentum_Flow':
                f = self._get_f_momentum_flow()
            elif mom is 'Energy':
                f = self._get_f_energy()
            elif mom is 'Energy_Flow':
                f = self._get_f_energy_flow()
            else:
                assert False, 'Unspecified Moment = ' \
                              '{}'.format(mom)
            f_arr.append(f)
        return f_arr

    # For write_2_HDD this needs a complete Revision
    def apply(self, data):
        """Applies functions in :attr:`f_arr` onto the given
        :attr:`Calculation.data` and returns the result.
        In the future, the results are written directly onto the HDD.

        Parameters
        ----------
        data : :obj:`~numpy.ndarray` of :obj:`float`
            The current :attr:`Calculation.data`
            during :class:`Calculation`.
        """
        n_mom = len(self._cnf.animated_moments)
        data_p_shape = data.shape[0:-1]
        n_specimen = self._cnf.s.n
        shape = (n_mom,) + data_p_shape + (n_specimen,)
        # shape == (len(self.f_arr), self._cnf.p.size, self._cnf.s.n)
        result = np.zeros(shape, dtype=float)
        for (i_f, f) in enumerate(self.f_arr):
            result[i_f, ...] = f(data)
        # Todo Add write to file routine here
        return result

    def _get_f_mass(self):
        p_shape = (self._cnf.p.size,)
        s_n = self._cnf.s.n
        shape = p_shape + (s_n,)

        def f_mass(data):
            """Returns Mass Distribution of given data."""
            mass = np.zeros(shape, dtype=float)
            for i_s in range(s_n):
                [beg, end] = self._cnf.sv.index[i_s: i_s+2]
                # mass = sum over velocity grid of specimen (last axis)
                mass[..., i_s] = np.sum(data[..., beg:end], axis=-1)
            return mass
        return f_mass
