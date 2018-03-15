import numpy as np


class OutputFunction:
    """Provides the :meth:`apply` method which
    processes the :attr:`Calculation.data`
    and stores the results on the disk.

    The class generates functions (:attr:`f_arr`) for all moments in
    :attr:`~boltzmann.configuration.Configuration.animated_moments`.
    These functions
    take a PSV-Grid (like :attr:`Calculation.data`) as a parameter
    and return the respective physical property as the result.
    The :meth:`apply` method iteratively calls all
    the functions in :attr:`f_arr`
    and writes each results to a single file on the disk.

    Parameters
    ----------
    cnf : :class:`~boltzmann.configuration.Configuration`
    """
    def __init__(self,
                 cnf):
        self._cnf = cnf
        self._f_arr = np.array([], dtype=object)
        self._setup_f_arr()
        return

    @property
    def cnf(self):
        """:obj:`~boltzmann.configuration.Configuration`:
        Points at the Configuration"""
        return self._cnf

    @property
    def f_arr(self):
        """:obj:`list` of :obj:`function`:
        List of moment generating functions."""
        return self._f_arr

    def _setup_f_arr(self):
        """Sets up :attr:`f_arr`"""
        f_arr = []
        for mom in self.cnf.animated_moments.flatten():
            if mom == 'Mass':
                f = self._get_f_mass()
            # Todo Mass_Flow == Momentum? Ask Hans
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
                raise NotImplementedError(message)
            f_arr.append(f)
        self._f_arr = np.array(f_arr)
        return

    def apply(self, calc):
        """Processes the current :attr:`Calculation.data`
        and writes the results to the disk.

        Iteratively applies all functions in :attr:`f_arr`
        onto the current :attr:`Calculation.data`
        and writes each result
        to a single file on the disk.

        Parameters
        ----------
        calc : :obj:`Calculation`
        """
        moments = self._cnf.animated_moments.flatten()
        data_p_shape = calc.data.shape[0:-1]
        n_specimen = self.cnf.s.n
        result_shape = (n_specimen,) + data_p_shape
        for (i_m, moment) in enumerate(moments):
            file_address = self._cnf.get_file_address(moment,
                                                      'npy',
                                                      t=calc.t_cur)
            result = self.f_arr[i_m](calc.data)

            assert result.shape == result_shape
            np.save(file_address, result)
        return

    def _get_f_mass(self):
        """Generates and returns generating function for Mass"""
        p_shape = (self.cnf.p.size,)
        s_n = self.cnf.s.n
        shape = (s_n,) + p_shape

        def f_mass(data):
            """Returns Mass for each
            P-:class:`boltzmann.configuration.Grid` point,
            based on given data
            """
            mass = np.zeros(shape, dtype=float)
            for i_s in range(s_n):
                [beg, end] = self.cnf.sv.index[i_s: i_s+2]
                # mass = sum over velocity grid of specimen (last axis)
                mass[i_s, :] = np.sum(data[..., beg:end], axis=-1)
            return mass
        return f_mass

    def _get_f_momentum(self, direction):
        """Generates and returns generating function for Momentum"""
        assert direction in [0, 1, 2]
        p_shape = (self.cnf.p.size,)
        s_n = self.cnf.s.n
        shape = (s_n,) + p_shape

        def f_momentum(data):
            """Returns Momentum for each
            P-:class:`boltzmann.configuration.Grid` point,
            based on given data
            """
            momentum = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self.cnf.sv.index[s: s+2]
                V_dir = self.cnf.sv.G[beg:end, direction]
                momentum[s, :] = np.sum(V_dir * data[..., beg:end],
                                        axis=1)
                momentum[s, :] *= self.cnf.s.mass[s]
            return momentum
        return f_momentum

    def _get_f_momentum_flow(self, direction):
        """Generates and returns generating function for Momentum Flow"""
        p_shape = (self.cnf.p.size,)
        s_n = self.cnf.s.n
        shape = (s_n,) + p_shape

        def f_momentum_flow(data):
            """Returns Momentum Flow for each
            P-:class:`boltzmann.configuration.Grid` point,
            based on given data
            """
            momentum_flow = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self.cnf.sv.index[s: s+2]
                V_dir = np.array(self.cnf.sv.G[beg:end, direction])
                momentum_flow[s, :] = np.sum(V_dir**2 * data[..., beg:end],
                                             axis=1)
                momentum_flow[s, :] *= self.cnf.s.mass[s]
            return momentum_flow
        return f_momentum_flow

    def _get_f_energy(self):
        """Generates and returns generating function for Energy"""
        p_shape = (self.cnf.p.size,)
        s_n = self.cnf.s.n
        shape = (s_n,) + p_shape

        def f_energy(data):
            """Returns Energy for each
            P-:class:`boltzmann.configuration.Grid` point,
            based on given data
            """
            energy = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self.cnf.sv.index[s: s+2]
                V = np.array(self.cnf.sv.G[beg:end, :])
                V_norm = np.sqrt(np.sum(V**2, axis=1))
                energy[s, :] = np.sum(V_norm * data[..., beg:end],
                                      axis=1)
                energy[s, :] *= 0.5 * self.cnf.s.mass[s]
            return energy
        return f_energy

    def _get_f_energy_flow(self, direction):
        """Generates and returns generating function for Energy Flow"""
        p_shape = (self.cnf.p.size,)
        s_n = self.cnf.s.n
        shape = (s_n,) + p_shape

        def f_energy_flow(data):
            """Returns Energy Flow for each
            P-:class:`boltzmann.configuration.Grid` point,
            based on given data
            """
            energy_flow = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self.cnf.sv.index[s: s + 2]
                V = np.array(self.cnf.sv.G[beg:end, :])
                V_norm = np.sqrt(np.sum(V ** 2, axis=1))
                V_dir = np.array(self.cnf.sv.G[beg:end, direction])
                energy_flow[s, :] = np.sum(V_norm * V_dir * data[..., beg:end],
                                           axis=1)
                energy_flow[s, :] *= 0.5 * self.cnf.s.mass[s]
            return energy_flow

        return f_energy_flow
