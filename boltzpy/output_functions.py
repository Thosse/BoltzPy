
import numpy as np
import h5py


# Todo Redo this, add new functionalities from new SVGrid
class OutputFunction:
    """Provides the :meth:`apply` method which
    processes the :attr:`Calculation.data`
    and writes the results to the simulation file.

    The class generates functions (:attr:`f_arr`) for all 
    :attr:`Simulation.output_parameters`.
    The :meth:`apply` method iteratively calls all
    the functions in :attr:`f_arr`
    and writes each results to a single file on the disk.

    Parameters
    ----------
    simulation : :class:`~boltzpy.Simulation`
    """
    def __init__(self, simulation):
        self._sim = simulation
        self._f_arr = np.array([], dtype=object)
        # Todo This must be changed
        # Todo  -> call setup method separately, not t initialization
        # Todo  -> call setup method separately, not t initialization
        # self._setup_f_arr()
        # self._setup_hdf5_subgroups()
        return

    @property
    def f_arr(self):
        """:obj:`~numpy.ndarray` of :obj:`function`:
        Array of moment generating functions.

        These functions
        take :attr:`Calculation.data` as a parameter
        and return the respective physical property as the result."""
        return self._f_arr

    def apply(self, calc):
        """Processes the current :attr:`Calculation.data`
        and writes the results to the disk.

        Iteratively applies all moment generating functions
        in :attr:`f_arr` to the current :attr:`Calculation.data`
        and writes the results to the
        :class:`boltzpy.Simulation` file.

        Parameters
        ----------
        calc : :obj:`Calculation`
        """
        output_flat = self._sim.output_parameters.flatten()
        species = self._sim.s.names
        time_index = calc.t_cur // self._sim.t.multi
        hdf5_group= h5py.File(self._sim.file_address + '.hdf5')["Computation"]
        for (output_idx, output) in enumerate(output_flat):
            result = self.f_arr[output_idx](calc.data)
            for (i_s, specimen) in enumerate(species):
                file_m = hdf5_group[specimen][output]
                file_m[time_index] = result[i_s]
        return

    def setup_hdf5_subgroups(self):
        file = h5py.File(self._sim.file_address + '.hdf5')
        # Todo don't overwrite existing results!
        if "Computation" not in file.keys():
            file.create_group("Computation")
        hdf5_group = file["Computation"]
        for specimen in self._sim.s.names:
            if specimen not in hdf5_group.keys():
                hdf5_group.create_group(specimen)
            file_s = hdf5_group[specimen]
            for moment in self._sim.output_parameters.flatten():
                if moment not in file_s.keys():
                    # Todo make property for shape?
                    # todo this simplifies complete output?
                    shape = (self._sim.t.n[0],
                             self._sim.p.iG.shape[0])
                    file_s.create_dataset(moment,
                                          shape=shape,
                                          dtype=float)
        return

    def setup_f_arr(self):
        """Sets up :attr:`f_arr`"""
        f_arr = []
        for mom in self._sim.output_parameters.flatten():
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

    def _get_f_mass(self):
        """Generates and returns generating function for Mass"""
        p_shape = (self._sim.p.size,)
        s_n = self._sim.s.n
        shape = (s_n,) + p_shape

        def f_mass(data):
            """Returns Mass for each
            P-:class:`boltzpy.Grid` point,
            based on given data
            """
            mass = np.zeros(shape, dtype=float)
            for i_s in range(s_n):
                [beg, end] = self._sim.sv.range_of_indices(i_s)
                # mass = sum over velocity grid of specimen (last axis)
                mass[i_s, :] = np.sum(data[..., beg:end], axis=-1)
            return mass
        return f_mass

    def _get_f_momentum(self, direction):
        """Generates and returns generating function for Momentum"""
        assert direction in [0, 1, 2]
        p_shape = (self._sim.p.size,)
        s_n = self._sim.s.n
        shape = (s_n,) + p_shape

        def f_momentum(data):
            """Returns Momentum for each
            P-:class:`boltzpy.Grid` point,
            based on given data
            """
            momentum = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self._sim.sv.range_of_indices(s)
                V_dir = self._sim.sv.iMG[beg:end, direction]
                momentum[s, :] = np.sum(V_dir * data[..., beg:end],
                                        axis=1)
                momentum[s, :] *= self._sim.s.mass[s]
            return momentum
        return f_momentum

    def _get_f_momentum_flow(self, direction):
        """Generates and returns generating function for Momentum Flow"""
        p_shape = (self._sim.p.size,)
        s_n = self._sim.s.n
        shape = (s_n,) + p_shape

        def f_momentum_flow(data):
            """Returns Momentum Flow for each
            P-:class:`boltzpy.Grid` point,
            based on given data
            """
            momentum_flow = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self._sim.sv.range_of_indices(s)
                # Todo rename direction into axis or something like that
                V_dir = np.array(self._sim.sv.iMG[beg:end, direction])
                momentum_flow[s, :] = np.sum(V_dir**2 * data[..., beg:end],
                                             axis=1)
                momentum_flow[s, :] *= self._sim.s.mass[s]
            return momentum_flow
        return f_momentum_flow

    def _get_f_energy(self):
        """Generates and returns generating function for Energy"""
        p_shape = (self._sim.p.size,)
        s_n = self._sim.s.n
        shape = (s_n,) + p_shape

        def f_energy(data):
            """Returns Energy for each
            P-:class:`boltzpy.Grid` point,
            based on given data
            """
            energy = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self._sim.sv.range_of_indices(s)
                V = np.array(self._sim.sv.iMG[beg:end, :])
                V_norm = np.sqrt(np.sum(V**2, axis=1))
                energy[s, :] = np.sum(V_norm * data[..., beg:end],
                                      axis=1)
                energy[s, :] *= 0.5 * self._sim.s.mass[s]
            return energy
        return f_energy

    def _get_f_energy_flow(self, direction):
        """Generates and returns generating function for Energy Flow"""
        p_shape = (self._sim.p.size,)
        s_n = self._sim.s.n
        shape = (s_n,) + p_shape

        def f_energy_flow(data):
            """Returns Energy Flow for each
            P-:class:`boltzpy.Grid` point,
            based on given data
            """
            energy_flow = np.zeros(shape, dtype=float)
            for s in range(s_n):
                [beg, end] = self._sim.sv.range_of_indices(s)
                V = np.array(self._sim.sv.iMG[beg:end, :])
                V_norm = np.sqrt(np.sum(V ** 2, axis=1))
                V_dir = np.array(self._sim.sv.iMG[beg:end, direction])
                energy_flow[s, :] = np.sum(V_norm * V_dir * data[..., beg:end],
                                           axis=1)
                energy_flow[s, :] *= 0.5 * self._sim.s.mass[s]
            return energy_flow

        return f_energy_flow
