import numpy as np
from scipy.optimize import newton as sp_newton
import boltzpy as bp


class BaseModel(bp.BaseClass):
    r"""Manages the Velocity Grids of all
    :class:`~boltzpy.Species`.


    Note
    ----
    Just as in the :class:`Grid` class,
    the parameter :attr:`iMG` describes the
    position/physical values of all  Grid points.
    All entries must be viewed as multiples of :attr:`delta:

        :math:`pMG = iMG \cdot d`.

    Note that velocity grid points may occur in multiple
    :class:`Velocity Grids <boltzpy.Grid>`.
    Array of shape (:attr:`size`, :attr:`ndim`)

    Parameters
    ----------
    masses : :obj:`~numpy.array` [:obj:`int`]
        Denotes the masses of all specimen.
    shapes : :obj:`~numpy.array` [:obj:`int`]
        Denotes the shape of each :class:`boltzpy.Grid`.
    base_delta : :obj:`float`
        Internal step size (delta) of all :class:`Grids <boltzpy.Grid>`.
        This is NOT the physical distance between grid points.
    spacings : :obj:`~numpy.array` [:obj:`int`]
        Denotes the spacing of each :class:`boltzpy.Grid`.

    Attributes
    ----------
    ndim : :obj:`int`
        The dimension of all Velocity Grids`.
    nspc : :obj:`int` :
        The number of different specimen / velocity grids.
    nvels : :obj:`int` :
        The total number of velocity grid points over all grids.
    i_vels : :obj:`~numpy.array` [:obj:`int`]
        The array of all integer velocities.
        These are the concatenated integer velocities of each specimen
    vels : :obj:`~numpy.array` [:obj:`float`]
        The array of all velocities.
        These are the concatenated velocity grids for each specimen
    _idx_offset : :obj:`~numpy.array` [:obj:`int`]
        Denotes the beginning of the respective velocity grid
        in the multi grid :attr:`iMG`.
    """

    def __init__(self,
                 masses,
                 shapes,
                 base_delta,
                 spacings=None):
        self.masses = np.array(masses, dtype=int)
        self.shapes = np.array(shapes, dtype=int)
        self.base_delta = np.float(base_delta)
        # give default value = default_spacing
        if spacings is None:
            self.spacings = 2 * np.lcm.reduce(self.masses) // self.masses
        else:
            self.spacings = np.array(spacings, dtype=int)

        self.ndim = self.shapes.shape[-1]
        self.nspc = self.masses.size
        self.nvels = np.sum(np.prod(self.shapes, axis=-1))
        self.dv = self.base_delta * self.spacings
        self.i_vels = np.concatenate([G.iG for G in self.subgrids()])
        self.vels = self.base_delta * self.i_vels

        self._idx_offset = np.zeros(self.nspc + 1, dtype=int)
        for s in self.species:
            self._idx_offset[s + 1:] += self.subgrids(s).size

        self.spc_matrix = np.zeros((self.nvels, self.nspc), dtype=int)
        for s in self.species:
            self.spc_matrix[self.idx_range(s), s] = 1

        if type(self) == BaseModel:
            raise NotImplementedError
        return

    @staticmethod
    def parameters():
        params = {"masses",
                  "shapes",
                  "base_delta",
                  "spacings"}
        return params

    @staticmethod
    def attributes():
        attrs = BaseModel.parameters()
        attrs.update({"ndim",
                      "nvels",
                      "nspc",
                      "vels",
                      "_idx_offset",
                      "spc_matrix",
                      "species",
                      "max_vel"})
        return attrs

    #####################################
    #           Properties              #
    #####################################
    @property
    def species(self):
        return np.arange(self.nspc)

    def subgrids(self, s=None):
        if np.issubdtype(s, np.integer):
            return bp.Grid(self.shapes[s],
                           self.base_delta,
                           self.spacings[s],
                           is_centered=True)
        grids = np.array([bp.Grid(self.shapes[s],
                                  self.base_delta,
                                  self.spacings[s],
                                  is_centered=True)
                          for s in self.species],
                         dtype=bp.Grid)
        if s is None:
            return grids
        else:
            return grids[s]

    @property
    def max_vel(self):
        """:obj:`float`
        Maximum physical velocity of all sub grids."""
        return np.max(np.abs(self.vels))

    def c_vels(self, mean_velocity, s=None):
        mean_velocity = np.array(mean_velocity)
        assert mean_velocity.shape[-1] == self.ndim
        dim = self.ndim
        velocities = self.vels[self.idx_range(s), :]
        # mean_velocity may have ndim > 1, thus reshape into 3D
        shape = mean_velocity.shape[:-1]
        size = np.prod(shape, dtype=int)
        mean_velocity = mean_velocity.reshape((size, 1, dim))
        result = velocities[np.newaxis, ...] - mean_velocity
        return result.reshape(shape + (velocities.shape[0], dim))

    @staticmethod
    def p_vels(velocities, direction):
        direction = np.array(direction)
        assert direction.shape == (velocities.shape[-1],)
        assert np.linalg.norm(direction) > 1e-8
        shape = velocities.shape[:-1]
        size = np.prod(shape, dtype=int)
        velocities = velocities.reshape((size, direction.size))
        angle = direction / np.linalg.norm(direction)
        result = np.dot(velocities, angle)
        return result.reshape(shape)

    def get_array(self, parameter):
        assert parameter.shape[-1] == self.nspc
        shape = parameter.shape[:-1] + (self.nvels,)
        parameter = parameter.reshape((-1, 1, self.nspc))
        spc_matrix = self.spc_matrix[:, np.newaxis, :]
        result = np.sum(spc_matrix * parameter, axis=-1)
        return result.reshape(shape)

    def temperature_range(self, mean_velocity=0):
        max_v = self.max_vel
        mean_v = np.max(np.abs(mean_velocity))
        assert mean_v < max_v
        min_mass = np.min(self.masses)
        max_temp = (max_v - mean_v)**2 / min_mass
        min_temp = 3 * np.max(self.spacings * self.base_delta)**2 / min_mass
        return np.array([min_temp, max_temp], dtype=float)

    #####################################
    #               Indexing            #
    #####################################
    def idx_range(self, s=None):
        if s is None:
            return np.s_[:]
        else:
            return np.s_[self._idx_offset[s]: self._idx_offset[s+1]]

    def get_idx(self,
                species,
                integer_velocities):
        """Find index of given grid_entry in :attr:`i_vels`
        Returns None, if the value is not in the specified Grid.

        Parameters
        ----------
        species : :obj:`~numpy.array` [:obj:`int`]
        integer_velocities : :obj:`~numpy.array` [:obj:`int`]

        Returns
        -------
        global_index : :obj:`int` of :obj:`None`

        """
        assert integer_velocities.dtype == int
        # turn species into array of ndim at least 1
        species = np.array(species, ndmin=1)
        assert integer_velocities.shape[-1] == self.ndim
        # reshape velocities for vectorization
        shape = integer_velocities.shape[:-1]
        integer_velocities = integer_velocities.reshape((-1, species.size, self.ndim))

        subgrids = self.subgrids()
        indices = np.zeros(integer_velocities.shape[0:2], dtype=int)
        for idx_s, s in enumerate(species):
            local_indices = subgrids[s].get_idx(integer_velocities[:, idx_s, :])
            indices[..., idx_s] = np.where(local_indices >= 0,
                                           local_indices + self._idx_offset[s],
                                           -1)
        return indices.reshape(shape)

    def get_spc(self, indices):
        """Get the specimen of given indices of :attr:`i_vels`.

        Parameters
        ----------
        indices : :obj:`~numpy.array` [ :obj:`int` ]

        Returns
        -------
        species : :obj:`~numpy.array` [ :obj:`int` ]
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        species = np.full(indices.shape, -1, dtype=int)
        for s in self.species:
            offset = self._idx_offset[s]
            species = np.where(indices >= offset, s, species)
        species = np.where(indices >= self._idx_offset[-1], -1, species)
        assert np.all(species >= 0)
        return species

    ##################################
    #      Initial Distribution      #
    ##################################
    @staticmethod
    def maxwellian(velocities,
                   temperature,
                   mass=1,
                   number_density=1,
                   mean_velocity=0):
        dim = velocities.shape[-1]
        # compute exponential with matching mean velocity and temperature
        exponential = np.exp(
            -0.5 * mass / temperature
            * np.sum((velocities - mean_velocity)**2, axis=-1))
        divisor = np.sqrt(2*np.pi * temperature / mass) ** (dim / 2)
        # multiply to get desired number density
        result = (number_density / divisor) * exponential
        return result

    def _init_error(self, moment_parameters, wanted_moments, s=None):
        dim = self.ndim
        mass = self.get_array(self.masses) if s is None else self.masses[s]
        velocities = self.vels[self.idx_range(s)]

        # compute values of maxwellian on given velocities
        state = self.maxwellian(velocities=velocities,
                                temperature=moment_parameters[dim],
                                mass=mass,
                                mean_velocity=moment_parameters[0: dim])
        # compute momenta
        mean_velocity = self.cmp_mean_velocity(state, s)
        temperature = self.cmp_temperature(state, s, mean_velocity=mean_velocity)
        # return difference from wanted_moments
        moments = np.zeros(moment_parameters.shape, dtype=float)
        moments[0: dim] = mean_velocity
        moments[dim] = temperature
        return moments - wanted_moments

    def cmp_initial_state(self,
                          number_densities,
                          mean_velocities,
                          temperatures):
        # number densities can be multiplied on the distribution at the end
        number_densities = np.array(number_densities)
        assert number_densities.shape == (self.nspc,)

        # mean velocites and temperatures are the targets for the newton scheme
        wanted_moments = np.zeros((self.nspc, self.ndim + 1), dtype=float)
        wanted_moments[:, : self.ndim] = mean_velocities
        wanted_moments[:, self.ndim] = temperatures

        # initialization parameters for the maxwellians
        init_params = np.zeros((self.nspc, self.ndim + 1), dtype=float)

        # if all specimen have equal mean_velocities and temperatures
        # then create a maxwellian in equilibrium (invariant under collisions)
        is_in_equilibrium = np.allclose(wanted_moments, wanted_moments[0])
        if is_in_equilibrium:
            # To create an equilibrium, all specimen must have equal init_params
            # We cannot achieve each specimen to fit the wanted_values
            # Thus we fit total moments to the wanted_values
            init_params[:] = sp_newton(self._init_error,
                                       wanted_moments[0],
                                       args=(wanted_moments[0], None))
        else:
            # Here we don't need an equilibrium,
            # all specimen may have different init_params
            # Thus we can fit each specimen to its wanted_values
            for s in self.species:
                init_params[s] = sp_newton(self._init_error,
                                           wanted_moments[s],
                                           args=(wanted_moments[s], s))

        # set up initial state, compute maxwellians with the init_params
        initial_state = np.zeros(self.nvels, dtype=float)
        for s in self.species:
            idx_range = self.idx_range(s)
            state = self.maxwellian(velocities=self.vels[idx_range],
                                    mass=self.masses[s],
                                    temperature=init_params[s, self.ndim],
                                    mean_velocity=init_params[s, 0:self.ndim])
            # normalize meaxwellian to number density == 1
            state /= self.cmp_number_density(state, s)
            # multiply to get the wanted number density
            state *= number_densities[s]
            initial_state[idx_range] = state
        return initial_state

    ##################################
    #        Moment functions        #
    ##################################
    @staticmethod
    def is_orthogonal(direction_1, direction_2):
        return np.allclose(np.dot(direction_1, direction_2), 0)

    @staticmethod
    def is_parallel(direction_1, direction_2):
        norms = np.linalg.norm(direction_1) * np.linalg.norm(direction_2)
        return np.allclose(np.dot(direction_1, direction_2), norms)

    # TODO if directions = None, return tensor, check if this is easy to do
    def mf_stress(self, mean_velocity, direction_1, direction_2, s=None):
        mean_velocity = np.array(mean_velocity)
        direction_1 = np.array(direction_1)
        direction_2 = np.array(direction_2)
        assert direction_1.shape == (self.ndim,)
        assert direction_2.shape == (self.ndim,)
        assert (self.is_parallel(direction_1, direction_2)
                or self.is_orthogonal(direction_1, direction_2))

        mass = self.get_array(self.masses) if s is None else self.masses[s]
        c_vels = self.c_vels(mean_velocity, s)
        p_vels_1 = self.p_vels(c_vels, direction_1)
        p_vels_2 = self.p_vels(c_vels, direction_2)
        return mass * p_vels_1 * p_vels_2

    def mf_pressure(self, mean_velocity, s=None):
        mean_velocity = np.array(mean_velocity)
        dim = self.ndim
        mass = (self.get_array(self.masses)[np.newaxis, :]
                if s is None else self.masses[s])
        velocities = self.vels[self.idx_range(s), :]
        # reshape mean_velocity (may have higher dimension)
        shape = mean_velocity.shape[:-1]
        size = np.prod(shape, dtype=int)
        mean_velocity = mean_velocity.reshape((size, dim))
        c_vels = self.c_vels(mean_velocity, s)
        c_vels = c_vels.reshape((size,) + velocities.shape)
        # compute pressure moment function
        mf_pressure = mass / dim * np.sum(c_vels**2, axis=-1)
        return mf_pressure.reshape(shape + (velocities.shape[0],))

    def mf_orthogonal_stress(self, mean_velocity, direction_1, direction_2, s=None):
        mean_velocity = np.array(mean_velocity)
        direction_1 = np.array(direction_1)
        direction_2 = np.array(direction_2)
        result = self.mf_stress(mean_velocity, direction_1, direction_2, s)
        if self.is_orthogonal(direction_1, direction_2):
            return result
        elif self.is_parallel(direction_1, direction_2):
            return result - self.mf_pressure(mean_velocity, s)
        else:
            raise ValueError

    def mf_heat_flow(self, mean_velocity, direction, s=None):
        mean_velocity = np.array(mean_velocity)
        direction = np.array(direction)
        assert direction.shape == (self.ndim,)
        mass = self.get_array(self.masses) if s is None else self.masses[s]
        c_vels = self.c_vels(mean_velocity, s)
        p_vels = self.p_vels(c_vels, direction)
        squared_sum = np.sum(c_vels ** 2, axis=-1)
        return mass * p_vels * squared_sum

    def mf_orthogonal_heat_flow(self, state, direction, s=None):
        direction = np.array(direction)
        direction = direction / np.sum(direction**2)
        # compute mean velocity
        mean_velocity = self.cmp_mean_velocity(state, s)
        # compute non-orthogonal moment function
        mf = self.mf_heat_flow(mean_velocity, direction, s)
        # subtract non-orthogonal part
        # in continuum this is (d+2)*T * c_vels
        # however this is not precise enough in grids
        # thus subtract correction term based on state
        p_vels = self.c_vels(mean_velocity) @ direction
        ct = (self.cmp_momentum(mf * state) @ direction
              / (self.cmp_momentum(p_vels*state) @ direction)
              * p_vels)
        result = mf - ct
        return result

    ##################################
    #            Moments             #
    ##################################
    def _get_state_of_species(self, state, s):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        """
        if state.shape[-1] == self.nvels:
            state = state[..., self.idx_range(s)]
        else:
            assert s is not None
            grid_size = self._idx_offset[s+1] - self._idx_offset[s]
            assert state.shape[-1] == grid_size
        return state

    def cmp_number_density(self, state, s=None, separately=False):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        separately : :obj:`bool`, optional
        """
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))
        # prepare paramters
        dim = self.ndim
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])
        if separately:
            assert s is None
            state = state[..., np.newaxis] * self.spc_matrix[np.newaxis, ...]
            shape = shape + (self.nspc,)
            dv = dv[..., np.newaxis]
            axis = -2
        else:
            axis = -1
        # compute number density
        result = np.sum(dv**dim * state, axis=axis)
        return result.reshape(shape)

    def cmp_mass_density(self, state, s=None):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        """
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))
        # prepare paramters
        dim = self.ndim
        mass = (self.get_array(self.masses)[np.newaxis, :]
                if s is None else self.masses[s])
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])
        # compute mass density
        result = np.sum(dv**dim * mass * state, axis=-1)
        return result.reshape(shape)

    def cmp_momentum(self, state, s=None):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        """
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))
        # prepare paramters
        dim = self.ndim
        mass = (self.get_array(self.masses)[np.newaxis, :]
                if s is None else self.masses[s])
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])
        velocities = self.vels[self.idx_range(s)]
        # compute momentum
        result = np.dot(dv**dim * mass * state, velocities)
        return result.reshape(shape + (dim,))

    def cmp_mean_velocity(self,
                          state=None,
                          s=None,
                          momentum=None,
                          mass_density=None):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        momentum : :obj:`~numpy.ndarray` [:obj:`float`], optional
        mass_density : :obj:`~numpy.ndarray` [:obj:`float`], optional
        """
        if state is None:
            assert momentum is not None and mass_density is not None
        if momentum is None:
            momentum = self.cmp_momentum(state, s)
        if mass_density is None:
            mass_density = self.cmp_mass_density(state, s)
        if np.any(np.isclose(mass_density, 0)):
            raise ValueError
        return momentum / mass_density[..., np.newaxis]

    def cmp_energy_density(self, state, s=None):
        r"""

        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`int`, optional
        """
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))
        # prepare paramters
        dim = self.ndim
        mass = self.get_array(self.masses) if s is None else self.masses[s]
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])
        velocities = self.vels[self.idx_range(s)]
        # compute energy density
        energy = 0.5 * mass * np.sum(velocities ** 2, axis=-1)
        result = np.dot(dv**dim * state, energy)
        # return result, reshaped into old shape
        return result.reshape(shape)

    def cmp_pressure(self, state, s=None, mean_velocity=None):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        mean_velocity : :obj:`~numpy.ndarray` [:obj:`float`], optional
        """
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))
        if mean_velocity is None:
            mean_velocity = self.cmp_mean_velocity(state, s)
        # prepare paramters
        dim = self.ndim
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])
        # compute pressure
        mf_pressure = self.mf_pressure(mean_velocity, s)
        # mf_pressure.shape[0] might be 1 or state.shape[0]
        mf_pressure = mf_pressure.reshape((-1, state.shape[-1]))
        result = np.sum(dv**dim * mf_pressure * state, axis=-1)
        return result.reshape(shape)

    def cmp_temperature(self,
                        state=None,
                        s=None,
                        pressure=None,
                        number_density=None,
                        mean_velocity=None):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        pressure : :obj:`~numpy.ndarray` [:obj:`float`], optional
        number_density : :obj:`~numpy.ndarray` [:obj:`float`], optional
        mean_velocity : :obj:`~numpy.ndarray` [:obj:`float`], optional
        """
        if state is None:
            assert pressure is not None and number_density is not None
        if pressure is None:
            pressure = self.cmp_pressure(state, s, mean_velocity)
        if number_density is None:
            number_density = self.cmp_number_density(state, s)
        if np.any(np.isclose(number_density, 0)):
            raise ValueError
        return pressure / number_density

    def cmp_momentum_flow(self, state, s=None):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        """
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))
        # prepare paramters
        dim = self.ndim
        mass = (self.get_array(self.masses)[np.newaxis, :]
                if s is None else self.masses[s])
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])
        velocities = self.vels[self.idx_range(s)]
        # compute momentum flow
        result = np.dot(dv**dim * mass * state, velocities ** 2)
        return result.reshape(shape + (dim,))

    def cmp_energy_flow(self, state, s=None):
        r"""
        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional
        """
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))
        # prepare paramters
        dim = self.ndim
        mass = (self.get_array(self.masses)[np.newaxis, :]
                if s is None else self.masses[s])
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])
        velocities = self.vels[self.idx_range(s)]
        # compute energy flow
        energies = 0.5 * mass * np.sum(velocities ** 2, axis=1)[:, np.newaxis]
        result = np.dot(dv**dim * state, energies * velocities)
        return result.reshape(shape + (dim,))

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check."""
        assert isinstance(self.ndim, int)
        assert self.ndim in {2, 3}

        assert isinstance(self.max_vel, float)
        assert self.max_vel > 0

        assert isinstance(self.base_delta, float)
        assert self.base_delta > 0

        assert isinstance(self.shapes, np.ndarray)
        assert self.shapes.dtype == int
        assert self.shapes.shape == (self.nspc, self.ndim)
        assert np.array_equal([G.shape for G in self.subgrids()],
                              self.shapes)

        assert isinstance(self.spacings, np.ndarray)
        assert self.spacings.dtype == int
        assert self.spacings.shape == (self.nspc,)
        assert np.array_equal([G.spacing for G in self.subgrids()],
                              self.spacings,)

        assert isinstance(self.subgrids(), np.ndarray)
        assert self.subgrids().shape == (self.nspc,)
        assert self.subgrids().ndim == 1
        assert self.subgrids().dtype == 'object'
        for G in self.subgrids():
            isinstance(G, bp.Grid)
            G.check_integrity()

        assert isinstance(self._idx_offset, np.ndarray)
        assert self._idx_offset.dtype == int
        assert self._idx_offset.shape == (self.nspc + 1,)
        assert np.all(self._idx_offset >= 0)
        assert np.all(self._idx_offset[1:] > self._idx_offset[:-1])
        assert np.array_equal(self._idx_offset[1:] - self._idx_offset[:-1],
                              [G.size for G in self.subgrids()])
        assert self._idx_offset[-1] == self.nvels

        assert isinstance(self.i_vels, np.ndarray)
        assert self.i_vels.dtype == int
        assert self.i_vels.shape == (self.nvels, self.ndim)
        assert self.i_vels.ndim == 2
        return

    def __str__(self,
                write_physical_grid=False):
        """:obj:`str` :
        A human readable string which describes all attributes of the instance.
        """
        description = ''
        description += "Dimension = {}\n".format(self.ndim)
        description += "Total Size = {}\n".format(self.nvels)

        for (idx_G, G) in enumerate(self.subgrids()):
            description += 'Specimen_{idx}:\n\t'.format(idx=idx_G)
            grid_str = G.__str__(write_physical_grid)
            description += grid_str.replace('\n', '\n\t')
            description += '\n'
        return description[:-1]
