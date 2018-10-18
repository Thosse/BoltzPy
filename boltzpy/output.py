
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
        self.func = None
        return

    def shape(self, mom):
        if mom in {'Mass',
                   'Momentum_X',
                   'Momentum_Y',
                   'Momentum_Z',
                   'Momentum_Flow_X',
                   'Momentum_Flow_Y',
                   'Momentum_Flow_Z',
                   'Energy',
                   'Energy_Flow_X',
                   'Energy_Flow_Y',
                   'Energy_Flow_Z'}:
            return (self._sim.t.iG.shape[0],
                    self._sim.p.iG.shape[0],
                    self._sim.s.size,
                    )
        elif mom == 'Complete_Distribution':
            return (self._sim.t.iG.shape[0],
                    self._sim.p.iG.shape[0],
                    self._sim.sv.size)
        else:
            message = 'Unsupported Output: {}'.format(mom)
            raise NotImplementedError(message)

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
        self.func(calc)

        return

    def output_function(self, hdf_group_name):
        # set up hdf5 datasets to store results in
        dataset_list = self.setup_hdf5_datasets(hdf_group_name)
        # setup output functions
        f_out_list = self.setup_f_arr()
        # combine both lists to iterate over the tuples
        # Todo why doesn't output_list = zip(f_out_list, dataset_list) work?
        # Todo it only saves the value for t = 0, all the other values are 0
        output_list = [(f_out_list[i], dataset_list[i])
                       for i in range(len(dataset_list))]

        # setup output function
        def func(calc):
            time_idx = calc.t_cur // self._sim.t.multi
            for (f_out, hdf5_dataset) in output_list:
                result = f_out(calc.data)
                hdf5_dataset[time_idx] = result
            return

        self.func = func
        return func

    def setup_hdf5_datasets(self, hdf5_group_name):
        hdf5_file = h5py.File(self._sim.file_address + '.hdf5')
        assert hdf5_group_name not in {'Collisions',
                                       'Initialization',
                                       'Position_Grid',
                                       'Species',
                                       'Time_grid',
                                       'Velocity_Grids'}
        # create hdf5 group, for results
        if hdf5_group_name not in hdf5_file.keys():
            hdf5_file.create_group(hdf5_group_name)
        hdf5_group = hdf5_file[hdf5_group_name]
        # setup datasets for all outputs
        output_list = self._sim.output_parameters.flatten()
        for output in output_list:
            # clear previous results, if any
            if output in hdf5_group.keys():
                del hdf5_group[output]
            shape = self.shape(output)
            hdf5_group.create_dataset(output,
                                      shape=shape,
                                      dtype=float)
        return [hdf5_group[output]
                for output in output_list]

    # Todo replace the different types of flows by a direction parameter
    # Todo this is a vector and allows more flexibility
    def setup_f_arr(self):
        """Sets up :attr:`f_arr`"""
        self._f_arr = []
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
            elif mom == 'Complete_Distribution':
                f = self._get_f_complete()
            else:
                message = 'Unsupported Output: {}'.format(mom)
                raise NotImplementedError(message)
            self._f_arr.append(f)
        return self._f_arr

    def _get_f_mass(self):
        """Generates and returns generating function for Mass"""
        # ignore time dimension
        shape = self.shape("Mass")[1:]

        def f_mass(data):
            mass = np.zeros(shape, dtype=float)
            for s_idx in range(shape[-1]):
                [beg, end] = self._sim.sv.idx_range(s_idx)
                # mass is the sum over velocity grid of specimen
                mass[..., s_idx] = np.sum(data[..., beg:end],
                                          axis=-1)
            return mass

        return f_mass

    def _get_f_momentum(self, direction):
        """Generates and returns generating function for Momentum"""
        assert direction in [0, 1, 2]
        # ignore time dimension
        shape = self.shape("Momentum_X")[1:]

        def f_momentum(data):
            momentum = np.zeros(shape, dtype=float)
            for s_idx in range(shape[-1]):
                [beg, end] = self._sim.sv.idx_range(s_idx)
                V_dir = self._sim.sv.iMG[beg:end, direction]
                momentum[..., s_idx] = np.sum(V_dir * data[..., beg:end],
                                              axis=1)
                momentum[..., s_idx] *= self._sim.s.mass[s_idx]
            return momentum

        return f_momentum

    def _get_f_momentum_flow(self, direction):
        """Generates and returns generating function for Momentum Flow"""
        # ignore time dimension
        shape = self.shape("Momentum_Flow_X")[1:]

        def f_momentum_flow(data):
            momentum_flow = np.zeros(shape, dtype=float)
            for s_idx in range(shape[-1]):
                [beg, end] = self._sim.sv.idx_range(s_idx)
                V_dir = np.array(self._sim.sv.iMG[beg:end, direction])
                momentum_flow[..., s_idx] = np.sum(V_dir ** 2 * data[...,
                                                                beg:end],
                                                   axis=1)
                momentum_flow[..., s_idx] *= self._sim.s.mass[s_idx]
            return momentum_flow

        return f_momentum_flow

    def _get_f_energy(self):
        """Generates and returns generating function for Energy"""
        # ignore time dimension
        shape = self.shape("Energy")[1:]

        def f_energy(data):
            energy = np.zeros(shape, dtype=float)
            for s_idx in range(shape[-1]):
                [beg, end] = self._sim.sv.idx_range(s_idx)
                V = np.array(self._sim.sv.iMG[beg:end, :])
                V_norm = np.sqrt(np.sum(V ** 2, axis=1))
                energy[..., s_idx] = np.sum(V_norm * data[..., beg:end],
                                            axis=1)
                energy[..., s_idx] *= 0.5 * self._sim.s.mass[s_idx]
            return energy

        return f_energy

    def _get_f_energy_flow(self, direction):
        """Generates and returns generating function for Energy Flow"""
        # ignore time dimension
        shape = self.shape("Energy_Flow_X")[1:]

        def f_energy_flow(data):
            energy_flow = np.zeros(shape, dtype=float)
            for s_idx in range(shape[-1]):
                [beg, end] = self._sim.sv.idx_range(s_idx)
                V = np.array(self._sim.sv.iMG[beg:end, :])
                V_norm = np.sqrt(np.sum(V ** 2, axis=1))
                V_dir = np.array(self._sim.sv.iMG[beg:end, direction])
                energy_flow[..., s_idx] = np.sum(V_norm
                                                 * V_dir
                                                 * data[..., beg:end],
                                                 axis=1)
                energy_flow[..., s_idx] *= 0.5 * self._sim.s.mass[s_idx]
            return energy_flow

        return f_energy_flow

    def _get_f_complete(self):
        """Generates and returns generating function for Complete
        Distribution"""

        def f_complete(data):
            """Returns complete distribution of given data
            """
            return data

        return f_complete
