import numpy as np


class OutputFunction:
    """Processes and stores results for animation.

    The class provides the
    :meth:`apply` method which generates the specified moments
    by iteratively applying the functions in
    :attr:`f_arr` on the current
    :attr:`~boltzmann.calculation.Calculation.data` PSV-Grid.
    The results are logged/appended to a file.

    .. todo::
      - choose if Posisional grid schould be dimensional?
      - implement rest of moment functions
      - figure out if using self.cnf in f_mass(psv)
        leads to the Configuration never being deleted from memory

    Attributes
    ----------
    f_arr : list(function)
        List of moment generating functions.
    files : list(str)
        files[i] is the name of the file, to which the results
        of f_arr[i] are appended.
    cnf : :class:`~boltzmann.configuration.Configuration`

    Parameters
    ----------
    moments : list(str)
        List/Array of Quantities/Properties to be animated.
    cnf : :class:`~boltzmann.configuration.Configuration`
        Configuration data.

    """
    SUPPORTED_OUTPUT = {'Mass',
                        'Mass_Flow',
                        'Momentum',
                        'Momentum_Flow',
                        'Energy',
                        'Energy_Flow'}

    def __init__(self,
                 moments,
                 cnf):
        assert all([mom in OutputFunction.SUPPORTED_OUTPUT
                    for mom in moments])
        self.cnf = cnf
        self.f_arr = self.create_function_list(moments)
        self.files = self.create_file_names(moments)
        return

    def create_file_names(self, moments):
        files = [self.cnf.file_name + '_' + mom
                 for mom in moments]
        return files

    def create_function_list(self, moments):
        f_arr = []
        for mom in moments:
            if mom is 'Mass':
                f = self.get_f_mass()
            elif mom is 'Mass_Flow':
                f = self.get_f_mass_flow()
            elif mom is 'Momentum':
                f = self.get_f_momentum()
            elif mom is 'Momentum_Flow':
                f = self.get_f_momentum_flow()
            elif mom is 'Energy':
                f = self.get_f_energy()
            elif mom is 'Energy_Flow':
                f = self.get_f_energy_flow()
            else:
                assert False, 'Unspecified Moment = ' \
                              '{}'.format(mom)
            f_arr.append(f)
        return f_arr

    def get_f_mass(self):
        p_shape = (self.cnf.p.size,)
        s_n = self.cnf.s.n
        shape = p_shape + (s_n,)

        def f_mass(psv):
            mass = np.zeros(shape, dtype=float)
            for i_s in range(s_n):
                beg = self.cnf.sv.index[i_s]
                end = self.cnf.sv.index[i_s + 1]
                # calculate sum over velocity grid (last axis)
                mass[..., i_s] = np.sum(psv[..., beg:end], axis=-1)
            return mass

        return f_mass
