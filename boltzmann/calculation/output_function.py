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
        for mom in self._cnf.animated_moments.flatten():
            if mom == 'Mass':
                f = self._get_f_mass()
            # Todo Implement Mass_Flow? == Momentum?
            # elif mom is 'Mass_Flow':
            #     f = self._get_f_mass_flow()
            elif mom == 'Momentum_X':
                f = self._get_f_momentum(0)
            elif mom == 'Momentum_Y':
                f = self._get_f_momentum(1)
            elif mom == 'Momentum_Z':
                f = self._get_f_momentum(2)
            elif mom == 'Momentum_Flow_X':
                f = self._get_f_momentum_flow(0)
            elif mom == 'Momentum_Flow_X':
                f = self._get_f_momentum_flow(1)
            elif mom == 'Momentum_Flow_X':
                f = self._get_f_momentum_flow(2)
            elif mom == 'Energy':
                f = self._get_f_energy()
            elif mom == 'Energy_Flow_X':
                f = self._get_f_energy_flow(0)
            elif mom == 'Energy_Flow_Y':
                f = self._get_f_energy_flow(1)
            elif mom == 'Energy_Flow_Z':
                f = self._get_f_energy_flow(2)
            else:
                message = 'Unsupported Output: {}'.format(mom)
                raise AttributeError(message)
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
        n_mom = self._cnf.animated_moments.size
        data_p_shape = data.shape[0:-1]
        n_specimen = self._cnf.s.n
        shape = (n_mom,) + (n_specimen,) + data_p_shape
        result = np.zeros(shape, dtype=float)
        for (i_f, f) in enumerate(self.f_arr):
            result[i_f, ...] = f(data)
        # Todo Add write to file routine here
        return result

    def _get_f_mass(self):
        p_shape = (self._cnf.p.size,)
        s_n = self._cnf.s.n
        shape = (s_n,) + p_shape

        def f_mass(data):
            """Returns Mass Distribution of given data."""
            mass = np.zeros(shape, dtype=float)
            for i_s in range(s_n):
                [beg, end] = self._cnf.sv.index[i_s: i_s+2]
                # mass = sum over velocity grid of specimen (last axis)
                mass[i_s, :] = np.sum(data[..., beg:end], axis=-1)
            return mass
        return f_mass

    def _get_f_momentum(self, direction):
        assert direction in [0, 1, 2]
        p_shape = (self._cnf.p.size,)
        s_n = self._cnf.s.n
        shape = (s_n,) + p_shape

        def f_momentum(data):
            """Returns Mass Distribution of given data."""
            momentum = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self._cnf.sv.index[s: s+2]
                V_dir = self._cnf.sv.G[beg:end, direction]
                momentum[s, :] = np.sum(V_dir * data[..., beg:end],
                                        axis=1)
                momentum[s, :] *= self._cnf.s.mass[s]
            return momentum
        return f_momentum

    def _get_f_momentum_flow(self, direction):
        p_shape = (self._cnf.p.size,)
        s_n = self._cnf.s.n
        shape = (s_n,) + p_shape

        def f_momentum_flow(data):
            """Returns Mass Distribution of given data."""
            momentum_flow = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self._cnf.sv.index[s: s+2]
                V_dir = np.array(self._cnf.sv.G[beg:end, direction])
                momentum_flow[s, :] = np.sum(V_dir**2 * data[..., beg:end],
                                             axis=1)
                momentum_flow[s, :] *= self._cnf.s.mass[s]
            return momentum_flow
        return f_momentum_flow

    def _get_f_energy(self):
        p_shape = (self._cnf.p.size,)
        s_n = self._cnf.s.n
        shape = (s_n,) + p_shape

        def f_energy(data):
            """Returns Mass Distribution of given data."""
            energy = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self._cnf.sv.index[s: s+2]
                V = np.array(self._cnf.sv.G[beg:end, :])
                V_norm = np.sqrt(np.sum(V**2, axis=1))
                energy[s, :] = np.sum(V_norm * data[..., beg:end],
                                      axis=1)
                energy[s, :] *= 0.5 * self._cnf.s.mass[s]
            return energy
        return f_energy

    def _get_f_energy_flow(self, direction):
        p_shape = (self._cnf.p.size,)
        s_n = self._cnf.s.n
        shape = (s_n,) + p_shape

        def f_energy_flow(data):
            """Returns Mass Distribution of given data."""
            energy_flow = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self._cnf.sv.index[s: s + 2]
                V = np.array(self._cnf.sv.G[beg:end, :])
                V_norm = np.sqrt(np.sum(V ** 2, axis=1))
                V_dir = np.array(self._cnf.sv.G[beg:end, direction])
                energy_flow[s, :] = np.sum(V_norm * V_dir * data[..., beg:end],
                                           axis=1)
                energy_flow[s, :] *= 0.5 * self._cnf.s.mass[s]
            return energy_flow

        return f_energy_flow
