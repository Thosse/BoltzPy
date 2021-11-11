import numpy as np
from scipy.optimize import newton as sp_newton
from itertools import product as iter_prod
from itertools import permutations as iter_perm
import boltzpy as bp


class BaseModel(bp.BaseClass):
    r"""Manages the Velocity Grids of all specimen.

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
    dv: :obj:`float` :
        The physical step size of the velocity grid for each specimen.
    i_vels : :obj:`~numpy.array` [:obj:`int`]
        The array of all integer velocities.
        These are the concatenated integer velocities of each specimen
    vels : :obj:`~numpy.array` [:obj:`float`]
        The array of all velocities.
        These are the concatenated velocity grids for each specimen
    _idx_offset : :obj:`~numpy.array` [:obj:`int`]
        Denotes the beginning of the respective velocity grid
        in the multi grid :attr:`iMG`.
    spc_matrix : :obj:`~numpy.array` [:obj:`int`]
        Alllows easy vectorization of species parameters
        like reflection rated.
        Multiply with this matrix to get an array of size nvels.
    """

    def __init__(self,
                 masses,
                 shapes,
                 base_delta=1.0,
                 spacings=None,
                 **kwargs):
        self.masses = np.array(masses, dtype=int)
        self.shapes = np.array(shapes, dtype=int)
        self.base_delta = float(base_delta)
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
        BaseModel.check_integrity(self)
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
                      "nspc",
                      "nvels",
                      "dv",
                      "i_vels",
                      "vels",
                      "_idx_offset",
                      "spc_matrix",
                      "species",
                      "max_vel"})
        return attrs

    @staticmethod
    def shared_attributes():
        """Set of class attributes that should share memory.

        These attributes must point to the same object
        as the attributes of the simulations CollisionModel
        This saves memory"""
        return {"i_vels", "vels", "_idx_offset", "spc_matrix"}

    #####################################
    #           Properties              #
    #####################################
    @property
    def species(self):
        """:obj:`~numpy.array` [:obj:`int`]
        The array of all specimen indices."""
        return np.arange(self.nspc)

    def subgrids(self, s=None):
        """Generate the Velocity Grids of all given specimen

        Parameters
        ----------
        s : :obj:`int`, optional

        Returns
        -------
        grids : :class:`~boltzpy.Grid`] or :obj:`~numpy.array` [:class:`~boltzpy.Grid`]
            Velocity Grids of the specimen
        """
        if np.issubdtype(type(s), np.integer):
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
    def max_i_vels(self):
        """:obj:`~numpy.array` [:obj:`float`]
        Array of maximum physical velocities of each subgrid."""
        return self.i_vels[self._idx_offset[1:] - 1]

    @property
    def max_vel(self):
        """:obj:`float`
        Maximum physical velocity of all sub grids."""
        return np.max(self.max_i_vels) * self.base_delta

    @property
    def is_cubic_grid(self):
        """:obj:`bool`
        Returns True, iff all grids are square/cubic,
        i.e. invariant under permutations"""
        return np.all(self.shapes - self.shapes[..., [0]] == 0)

    @property
    def permutation_matrices(self):
        """:obj:`~numpy.array` [:obj:`int`]
        Array of all permutation matrices for the velocities.

        .. note::
            1. The matrices are in a specific order,
               used in :func:`~boltzpy.Grid.key_partitioned_distance`.
            2. The matrices are orthogonal.
               The transpose is the inverse matrix.
        """
        # DO NOT CHANGE THIS ORDER!
        # These matrices are used for the fast collision generation.
        # This order is specially chosen for this!
        # when reshaping perm_matrices to a (3,2,3,3) array (in 3d)
        #   first index  : determines first column (int)
        #   second index : determines if remaining values are switched (bool)
        perm_list = list(iter_perm(range(self.ndim)))
        perm_array = np.array(perm_list, dtype=int)
        eye = np.eye(self.ndim, dtype=int)
        perm_matrices = eye[perm_array]
        # transpose/invert matrices. They are used most of the time.
        perm_matrices = perm_matrices.transpose((0, 2, 1))
        return perm_matrices

    @property
    def reflection_matrices(self):
        """:obj:`~numpy.array` [:obj:`int`]
        Array of all reflection matrices for the velocities.

         .. note::
            1. The matrices are in a specific order,
               used in :func:`~boltzpy.Grid.key_partitioned_distance`.
            2. The matrices are orthogonal.
               The transpose is the inverse matrix.
        """
        # DO NOT CHANGE THIS ORDER!
        # These matrices are used for the fast collision generation.
        # This order is specially chosen for this!
        refl_list = list(iter_prod(*([[1, -1]] * self.ndim)))
        refl_array = np.array(refl_list, dtype=int)[..., np.newaxis]
        eye = np.eye(self.ndim, dtype=int)[np.newaxis, ...]
        refl_matrices = eye * refl_array
        return refl_matrices

    @property
    def symmetry_matrices(self):
        """:obj:`~numpy.array` [:obj:`int`]
        Array of all symmetry operators / matrices,
        that are a homeomorphism.

         .. note::
            1. The matrices are in a specific order,
               used in :func:`~boltzpy.Grid.key_partitioned_distance`.
            2. The matrices are orthogonal.
               The transpose is the inverse matrix.
        """
        # DO NOT CHANGE THIS ORDER!
        # These matrices are used for the fast collision generation.
        # This order is specially chosen for this!
        sym_list = [r.dot(p)
                    for r in self.reflection_matrices
                    for p in self.permutation_matrices]
        sym_matrices = np.array(sym_list)
        return sym_matrices

    def c_vels(self, mean_velocity, s=None):
        """Returns the difference to the mean_velocity for each velocity


        This is simply vels - mean_velocity.
        However mean_velocity might be an array of arbitrary dimension.

        Parameters
        ----------
        mean_velocity : :obj:`~numpy.array` [:obj:`float`]
        s : :obj:`int`, optional
            specimen index, chooses the velocity grid.
            If None, the shared grid space is used.

        Returns
        -------
        c_vels : :obj:`~numpy.array` [:obj:`float`]
            Centered Velocities (around mean_velocity).
        """
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
        """Project the velocities onto a direction

        This is simply vels @ direction, but allows higher dimensions

        Parameters
        ----------
        velocities : :obj:`~numpy.array` [:obj:`float`]
        direction : :obj:`~numpy.array` [:obj:`float`]

        Returns
        -------
        p_vels : :obj:`~numpy.array` [:obj:`float`]
            Projected velocities. one dimension less (last) than velocities.
        """
        direction = np.array(direction, copy=False)
        assert direction.shape == (velocities.shape[-1],)
        assert np.linalg.norm(direction) > 1e-8
        shape = velocities.shape[:-1]
        size = np.prod(shape, dtype=int)
        velocities = velocities.reshape((size, direction.size))
        angle = direction / np.linalg.norm(direction)
        result = np.dot(velocities, angle)
        return result.reshape(shape)

    def get_array(self, parameter):
        """Generate a (nvels,) shaped array
        Parameters
        ----------
        parameter : :obj:`~numpy.array` [:obj:`float`]

        Returns
        -------
        p_vels : :obj:`~numpy.array` [:obj:`float`]
            Projected velocities. one dimension less (last) than velocities.
        """
        parameter = np.array(parameter, copy=False)
        assert parameter.shape[-1] == self.nspc
        shape = parameter.shape[:-1] + (self.nvels,)
        parameter = parameter.reshape((-1, 1, self.nspc))
        spc_matrix = self.spc_matrix[:, np.newaxis, :]
        result = np.sum(spc_matrix * parameter, axis=-1)
        return result.reshape(shape)

    def temperature_range(self,
                          mean_velocity=0.0,
                          s=None,
                          atol=None,
                          metric="temperature",
                          intervals=100,
                          return_error=False):
        """Determine a range of temperatures for which a good initial state can be generated.
        Parameters
        ----------
        mean_velocity : :obj:`~numpy.array` [:obj:`float`]
            Mean velocity parameter of the maxwellians
        s :  :obj:`~numpy.array` [:obj:`int`], optional
            If None is given, a shared range for all specimen is given.
        atol : :obj:`float`, optional
            Absolute tolerance for the given error metric.
            If None is given, then the heuristic results are returned.
        intervals : :obj:`int`
            Determines the number of test points for which the error metric is computed.
        return_error : :obj:`bool`
            Return the error metric array, instead of the temperature range.
            This is for Debugging.


        Returns
        -------
        t_range : :obj:`~numpy.array` [:obj:`float`]
            The estimated minimum and maximum temperatures,
            for which the newton scheme finds a common solution
            for the mixture.
        """
        # compute a rough estimate with the heuristic
        heuristic_range = self._temperature_range_mixture_heuristic(mean_velocity, s)
        if atol is None:
            return heuristic_range
        else:
            assert isinstance(atol, float)
            assert atol > 0
            assert isinstance(metric, str)

        # by default: check all species
        species = self.species if s is None else [s]
        # test metric error for a set of points
        test_points = np.linspace(*heuristic_range, intervals)
        # use ones, in case any cell is not written it is counted as a large error
        error = np.ones((len(species), intervals))
        # store temperature ranges for each species in here
        t_ranges = np.zeros((len(species), 2))
        for i_s, s in enumerate(species):
            # compute the error for each temperature
            for i_t, t in enumerate(test_points):
                # compute discrete maxwellian
                maxwellian = self.maxwellian(self.vels[self.idx_range(s)],
                                             temperature=t,
                                             mass=self.masses[s],
                                             mean_velocity=mean_velocity)
                # enforce number density = 1
                number_dens = self.cmp_number_density(maxwellian, s)
                maxwellian /= number_dens
                # compute error, depending on metric
                if metric == "temperature":
                    error[i_s, i_t] = t - self.cmp_temperature(maxwellian, s)
                elif metric == "mean_velocity":
                    vec_error = mean_velocity - self.cmp_mean_velocity(maxwellian, s)
                    error[i_s, i_t] = np.linalg.norm(vec_error)
                else:
                    raise NotImplementedError

        if return_error:
            return error
        # else: compute the temperature ranges of each species
        for i_s, s in enumerate(species):
            # find max temperature as last value, with error below atol
            hits, = np.where(np.abs(error[i_s]) < atol)
            if len(hits) == 0:
                raise ValueError("No Acceptable temperature found for species %1d" % s)
            i_max = hits[-1]
            # find min temperature right after the last False, before i_max
            hits, = np.where(np.abs(error[i_s, :i_max]) >= atol)
            i_min = hits[-1] + 1 if len(hits) > 0 else 0
            # get temperatures from test_points array
            t_ranges[i_s] = test_points[np.array([i_min, i_max])]

        # determine temperature range as maxmin and minmax of t_ranges
        t_range = np.array([np.max(t_ranges[:, 0]),
                            np.min(t_ranges[:, 1])])
        return t_range

    def _temperature_range_mixture_heuristic(self, mean_velocity=0.0, s=None):
        """Estimate a range of temperatures that can be initialized on this model.
        Parameters
        ----------
        mean_velocity : :obj:`~numpy.array` [:obj:`float`]
            Mean velocity parameter of the maxwellians
        s :  :obj:`~numpy.array` [:obj:`int`], optional
            If None is given, a shared range for all specimen is given.

        Returns
        -------
        t_range : :obj:`~numpy.array` [:obj:`float`]
            The estimated minimum and maximum temperatures,
            for which the newton scheme finds a common solution
            for the mixture.
        """
        mean_velocity = np.full((self.nspc, self.ndim), mean_velocity)
        if s is None:
            s = self.species
        elif type(s) in  [list, np.ndarray]:
            pass
        else:
            s = [s]

        # compute a shared range for possible temperatures for all specimen
        assert np.all(0 <= spc < self.nspc for spc in s)
        # compute temperature ranges for each specimen, store in spc_temp_range
        spc_temp_range = [self._temperature_range_single_species_heuristic(mean_velocity, s=spc)
                          for spc in s]
        spc_temp_range = np.array(spc_temp_range)
        # take max(min) and min(max), thus each specimen can be initialized
        temp_range = np.array([np.max(spc_temp_range[:, 0]),
                               np.min(spc_temp_range[:, 1])])
        return temp_range

    def _temperature_range_single_species_heuristic(self, mean_velocity, s):
        """Estimate a range of temperatures that can be initialized on this model.
        Parameters
        ----------
        mean_velocity : :obj:`~numpy.array` [:obj:`float`]

        s :  :obj:`int`

        Returns
        -------
        t_range : :obj:`~numpy.array` [:obj:`float`]
            The estimated minimum and maximum temperatures,
            for which the newton scheme finds a  solution.
        """
        assert s in list(self.species)
        assert np.all(self.shapes[s] > 1)
        # compute directly, if mean_velocity is 0
        if np.allclose(mean_velocity, 0):
            idx = self.idx_range(s)
            # set up spacing-indepentent maxwellians
            # as slowly/quickly exponentially decaying functions
            # to determine minimal maximal temperature
            # MINIMAL TEMPERATURE:
            # use a small base and normalize by spacing
            base = 0.05
            norm = self.spacings[s]
            dist = np.sum((self.i_vels[idx] / norm)**2, axis=-1)
            t_min = self.cmp_temperature(base ** dist, s=s)
            # MAXIMAL TEMPERATURE:
            # use a large base and normalize by maximum velocity
            base = 0.95
            norm = self.max_i_vels[s] * self.base_delta
            dist = np.sum((self.i_vels[idx] / norm)**2, axis=-1)
            t_max = self.cmp_temperature(base ** dist, s=s)
            return np.array([t_min, t_max])
        # if a mean_velocity is given,
        # compute two ranges for reduced models
        # and interpolate between them
        else:
            # use relative difference, for model reduction
            # Note: the model must be reduced in all dimensions
            rel_diff = np.full(self.ndim, np.max(np.abs(mean_velocity[s]) / self.dv[s]))
            lower_reduction = np.array(rel_diff, dtype=int)     # rounds down
            upper_reduction = lower_reduction + 1
            new_shapes = np.array([self.shapes[s] - lower_reduction,
                                   self.shapes[s] - upper_reduction])
            # compute interpolation weight, use distance along diagonal in velocity space
            ip_weight = np.max(np.abs(rel_diff - lower_reduction))
            ip_weight = np.array([1 - ip_weight, ip_weight])
            interpolation = np.zeros(2)
            # compute ranges for small/large reduced grids and interpolate
            for weight, shape in zip(ip_weight, new_shapes):
                # don't compute if not necessary, avoids assertion errors, for 3x3 models
                if np.isclose(weight, 0):
                    continue
                assert np.all(shape >= 2), "The mean_velocities are too large!"
                model = bp.BaseModel([self.masses[s]],
                                     [shape],
                                     self.base_delta,
                                     [self.spacings[s]])
                # model.species = [0], by construction
                centered_range = model._temperature_range_single_species_heuristic(0, s=0)
                interpolation += weight * centered_range
            return interpolation

    #####################################
    #               Indexing            #
    #####################################
    def idx_range(self, s=None):
        """A slice to access the specimens part of the shared velocity grid
         (vels, i_vels,...)

        Parameters
        ----------
        s : :obj:`~numpy.array` [:obj:`float`]

        Returns
        -------
        slice : :obj:`slice`
        """
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
        species = np.array(species, ndmin=1, dtype=int, copy=False)
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
        indices = np.array(indices, dtype=int, copy=False)
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
    # todo undo staticmethod, use cmp_numberdensitiy for t=0?
    @staticmethod
    def maxwellian(velocities,
                   temperature,
                   mass=1,
                   number_density=1,
                   mean_velocity=0):
        """Computes the continuous maxwellian velocity distribution.

        Parameters
        ----------
        velocities : :obj:`~numpy.array` [ :obj:`float` ]
        temperature : :obj:`float`
        mass : :obj:`int` or :obj:`~numpy.array` [ :obj:`int` ]
        number_density : :obj:`float`
        mean_velocity : :obj:`float`

        Returns
        -------
        distribution : :obj:`~numpy.array` [ :obj:`float` ]
        """
        dim = velocities.shape[-1]
        # compute exponential with matching mean velocity and temperature
        distance = np.sum((velocities - mean_velocity)**2,  axis=-1)
        exponential = np.exp(-0.5 * mass / temperature * distance)
        divisor = np.sqrt(2*np.pi * temperature / mass) ** dim
        # multiply to get desired number density
        result = (number_density / divisor) * exponential
        assert not np.any(np.isnan(result))
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
        """Computes a discrete velocity distribution.

        If all mean_velocities and temperatures are equal, respectively,
        then an equilibrium will be computed and the given moments
        are used to set the moments of the mixture.
        In this case the moments of each specimen itself may differ from that.

        Parameters
        ----------
        number_densities : :obj:`float`
        mean_velocities : :obj:`float`
        temperatures : :obj:`float`

        Returns
        -------
        distribution : :obj:`~numpy.array` [ :obj:`float` ]
        """
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
        # Todo assert np.all(np.abs(init_params[0:self.ndim]) < 1 * self.max_vel)
        #  This gives more reasonable initial states, but requires another heuristic
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

        # Todo assert that the correct moments are computed
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

    def mf_stress(self, mean_velocity, directions, s=None, orthogonalize=False):
        """Compute the stress moment function for given directions
        and either a single specimen or the mixture.

        Parameters
        ----------
        mean_velocity : :obj:`~numpy.array` [ :obj:`float` ]
        directions : :obj:`~numpy.array` [ :obj:`float` ]
            Must be of shape=(2,ndim).
        s : :obj:`int`, optional
        orthogonalize : :obj:`bool`. optional
            If True, the moment components of the moment function are subtracted.

        Returns
        -------
        mf_Stress : :obj:`~numpy.array` [ :obj:`float` ]
        """
        directions = np.array(directions, dtype=float, copy=False)
        mean_velocity = np.array(mean_velocity, dtype=float, copy=False)
        assert directions.shape == (2, self.ndim,)
        is_parallel = self.is_parallel(directions[0], directions[1])
        is_orthogonal = self.is_orthogonal(directions[0], directions[1])
        if not (is_parallel or is_orthogonal):
            raise ValueError("directions must be either parallel or orthogonal")
        # normalize directions
        norm = np.linalg.norm(directions, axis=1).reshape(2, 1)
        directions /= norm
        # compute stress
        mass = self.get_array(self.masses) if s is None else self.masses[s]
        c_vels = self.c_vels(mean_velocity, s)
        p_vels_1 = self.p_vels(c_vels, directions[0])
        p_vels_2 = self.p_vels(c_vels, directions[1])
        moment_function = mass * p_vels_1 * p_vels_2
        # orthogonalize, if necessary
        if orthogonalize and is_parallel:
            return moment_function - self.mf_pressure(mean_velocity, s)
        else:
            return moment_function

    def mf_pressure(self, mean_velocity, s=None):
        """Compute the pressure moment function
        for either a single specimen or the mixture.

        Parameters
        ----------
        mean_velocity : :obj:`~numpy.array` [ :obj:`float` ]
        s : :obj:`int`, optional

        Returns
        -------
        mf_pressure : :obj:`~numpy.array` [ :obj:`float` ]
        """
        mean_velocity = np.array(mean_velocity, dtype=float, copy=False)
        if s is None:
            mass = self.get_array(self.masses)[np.newaxis, :]
        else:
            mass = self.masses[s]
        dim = self.ndim
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

    def mf_heat_flow(self, mean_velocity, direction, s=None, orthogonalize_state=None):
        """Compute the heat flow moment function for a direction
        and either a single specimen or the mixture.

        Parameters
        ----------
        mean_velocity : :obj:`~numpy.array` [ :obj:`float` ]
        direction : :obj:`~numpy.array` [ :obj:`float` ]
        s : :obj:`int`, optional
        orthogonalize_state : :obj:`~numpy.array` [ :obj:`float` ], optional
            If not None, the moment components of the moment function are subtracted.
            For this an state/distribution must be given. It should be be a maxwellian.

        Returns
        -------
        mf_heaf_flow : :obj:`~numpy.array` [ :obj:`float` ]
        """
        mean_velocity = np.array(mean_velocity, dtype=float, copy=False)
        direction = np.array(direction, dtype=float, copy=False)
        assert direction.shape == (self.ndim,)
        mass = self.get_array(self.masses) if s is None else self.masses[s]
        c_vels = self.c_vels(mean_velocity, s)
        p_vels = self.p_vels(c_vels, direction)
        squared_sum = np.sum(c_vels ** 2, axis=-1)
        moment_function = mass * p_vels * squared_sum
        # orthogonalize, if necessary
        if orthogonalize_state is None:
            return moment_function
        else:
            # Subtract non-orthogonal part to orthogonalize.
            # In continuum this is (d+2)*T*c_vels,
            # this is not precise in grids!
            # thus subtract correction term based on state
            p_vels = self.c_vels(mean_velocity) @ direction
            momentum = self.cmp_momentum(moment_function * orthogonalize_state) @ direction
            norm = (self.cmp_momentum(p_vels * orthogonalize_state) @ direction)
            correction_term = momentum / norm * p_vels
            return moment_function - correction_term

    ##################################
    #            Moments             #
    ##################################
    def _get_state_of_species(self, state, s):
        r"""This reduces the state to the relevant parts, if necessary.
        It also asserts that the shape is correct.

        Parameters
        ----------
        state : :obj:`~numpy.ndarray` [:obj:`float`]
        s : :obj:`float`, optional

        Returns
        -------
        state : :obj:`~numpy.array` [ :obj:`float` ]
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
        If separately == True the number density of each specimen
        is computed separately.
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

    def cmp_stress(self, state, s=None, mean_velocity=None, directions=None):
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))

        # compute default mean_velocity, from state
        if mean_velocity is None:
            mean_velocity = self.cmp_mean_velocity(state, s)

        # choose default directions as unit vectors e1, e2
        if directions is None:
            directions = np.eye(self.ndim, dtype=float)[0:2]

        # prepare integration paramters
        dim = self.ndim
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])

        # get stress moment function
        mf_stress = self.mf_stress(mean_velocity, directions, s)
        # moment function must be 2d array
        mf_stress = mf_stress.reshape((-1, state.shape[-1]))

        # numerical integration
        result = np.sum(dv**dim * mf_stress * state, axis=-1)
        # reshape result into intitial shape
        result.reshape(shape)
        return result

    def cmp_heat_flow(self, state, s=None, mean_velocity=None, direction=None):
        state = self._get_state_of_species(state, s)
        # Reshape state into 2d
        shape = state.shape[:-1]
        size = np.prod(shape, dtype=int)
        state = state.reshape((size, state.shape[-1]))

        # compute default mean_velocity, from state
        if mean_velocity is None:
            mean_velocity = self.cmp_mean_velocity(state, s)

        # choose default directions as unit vector e1
        if direction is None:
            direction = np.eye(self.ndim, dtype=float)[0]

        # prepare integration paramters
        dim = self.ndim
        dv = (self.get_array(self.dv)[np.newaxis, :]
              if s is None else self.dv[s])

        # get stress moment function
        mf_heat_flow = self.mf_heat_flow(mean_velocity, direction, s)
        # moment function must be 2d array
        mf_heat_flow = mf_heat_flow.reshape((-1, state.shape[-1]))

        # numerical integration
        result = np.sum(dv**dim * mf_heat_flow * state, axis=-1)
        # reshape result into intitial shape
        result.reshape(shape)
        return result

    #####################################
    #           Visualization           #
    #####################################
    def plot_state(self, state, file_name=None):
        # use shape[0] to differ between animations and simple plots
        state = state.reshape((-1, self.nvels))
        # basic asserts
        assert state.ndim == 2
        assert state.shape[-1] == self.nvels
        # create animation/plot, depending on shape[0]
        fig = bp.Plot.AnimatedFigure(state.shape[0])
        for s in self.species:
            ax = fig.add_subplot((1, self.nspc, s+1), 3)
            idx_range = self.idx_range(s)
            vels = self.vels[idx_range]
            ax.plot(vels[..., 0], vels[..., 1], state[:, idx_range])
        if file_name is None:
            fig.show()
        else:
            assert type(file_name) is str
            fig.save(file_name)
        return

    #####################################
    #           Verification            #
    #####################################
    def check_integrity(self):
        """Sanity Check."""
        bp.BaseClass.check_integrity(self)
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

    # Todo this is actually useful to have , to print the subgrids
    #  but define a baseclass.__str__
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
