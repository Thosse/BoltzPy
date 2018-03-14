import numpy as np
# import math

from boltzmann.configuration import species as b_spc
from boltzmann.configuration import grid as b_grd


# def np_gcd(array_of_ints):
#     if array_of_ints.shape == (0,):
#         return 1
#     gcd = array_of_ints[0]
#     for new_number in array_of_ints[1:]:
#         gcd = math.gcd(gcd, new_number)
#     return gcd


class SVGrid:
    """Class of the concatenated Velocity Grid of all Specimen.

    .. todo::
        - add property for physical velocities (offset + G*d)
        - Ask Hans, should the Grid always contain the center/zero?
        - Todo Add unit tests
        - replace n with index_skips?
        - is it useful to implement different construction schemes?
          (grid boundaries/step_size can be chosen to fit in grids,
          based on the least common multiple,...)
        - For each S, self.n[S, :] should all be equal values. Reduce this?
        - implement base-class for multi-grid, where grid is multiGrid of dim=1

    Attributes
    ----------
    dim : int
        Dimensionality of all Grids.  Applies to all Specimen.
    d : array(float)
        Denotes the Grid step size for each specimen.
        Array of shape=(s.n, dim).
    n : array(int)
        n[S, DIM] denotes the number of Velocity-Grid points in dimension DIM
        for specimen S.
        Array of shape=(s.n, dim)
    size : array(int)
        n[S] denotes the total number of Velocity-Grid points
        for specimen S.
        Array of shape=(s.n,).
    index : array(int)
        index[i] denotes the beginning of the i-th velocity grid.
        By Definition index[0] is 0 and index[-1] is n[:,-1].sum().
        Array of shape=(s.n+1) and dType=int.
    G : array(int)
        Concatenated physical Velocity-Grids of all specimen.
        G[index[i]:index[i+1]] is the Velocity Grid of specimen i.
        G[j] denotes the physical coordinates of the j-th grid point.
        Note that some V-Grid-points occur several times,
        for different specimen.
        Array of shape(index[-1], dim).
    form : str
        Geometric shape of all Velocity-Grids. Equal to all Specimen.
        Can only be rectangular so far.
    offset : array(float)
        Offset for the physical velocities.
        The physical value of any Velocity-Grid point v_i of Specimen S
        is offset + d[S]*G[i].
        Array of shape=(dim,).
    multi : int
        Multiplicator, that determines the ratio of
        'actual step size' / :attr:`~boltzmann.configuration.Grid.d`.
        In several cases the Entries in G are multiplied by
        :attr:`~boltzmann.configuration.Grid.multi`
        and :attr:`~boltzmann.configuration.Grid.d` are divided by
        :attr:`~boltzmann.configuration.Grid.multi`.
        This does not change the physical Values, but enables a variety
         of features( Currently: several calculation steps per write,
         simple centralizing ov velocity grids)
    is_centered : bool
        True if Grid has been centered, False otherwise.

    """
    def __init__(self):
        self.dim = 0
        self.form = ''
        self.d = np.zeros(shape=(0,), dtype=float)
        self.n = np.zeros(shape=(0, 0), dtype=int)
        self.size = np.zeros(shape=(0,), dtype=int)
        self.index = np.zeros(shape=(0,), dtype=int)
        self.G = np.zeros((0, self.dim), dtype=int)
        self.multi = 1
        self.is_centered = False
        self.offset = np.zeros((self.dim,), dtype=float)
        return

    @property
    def boundaries(self):
        boundaries = np.zeros(shape=(self.size.size,
                                     2,
                                     self.dim),
                              dtype=float)
        for s in range(self.size.size):
            [beg, end] = self.index[[s, s+1]]
            G = self.G[beg:end]
            min_val = np.min(G, axis=0)
            max_val = np.max(G, axis=0)
            bound = np.array([min_val, max_val]) * self.d[s]
            if self.dim is 2:
                boundaries[s] = bound.transpose()
            elif self.dim is not 1:
                assert False, 'This is not tested for 3d'
                # Todo figure out transpose order
        return boundaries

    def setup(self,
              species,
              velocity_grid,
              offset=None,
              force_contain_center=False,
              check_integrity=True,
              create_grid=True):
        assert type(species) is b_spc.Species
        assert type(velocity_grid) is b_grd.Grid
        self.dim = velocity_grid.dim
        self.form = velocity_grid.form
        # set offset
        # Todo put into seperate method?
        if offset is None:
            self.offset = np.zeros((self.dim,), dtype=float)
        else:
            assert type(offset) in [int, list, np.ndarray]
            if type(offset) is int:
                offset = [offset]
            self.offset = np.array(offset, dtype=float)
            assert self.offset.shape is (self.dim,)

        m = species.mass
        m_min = np.amin(m)
        # The range of the Velocity-Grid of Specimen i is almost constant:
        # v_max[i]-v_min[i] = n[i] * d[i] ~ CONST
        # with CONST = velocity_grid.n * velocity_grid.d
        # If necessary, the range may decrease by at most 1 d[i]
        # => n may be rounded down

        # The lighter specimen are less inert and move further
        # => d[i]/d[j] = m[j]/m[i]
        # i.e. d[i] is inversely proportional to m[i]
        self.d = np.array(velocity_grid.d * m_min/m,
                          dtype=float)

        # Smaller step sizes lead to larger grids (n[i]*d[i] = CONST).
        # => n[i]/n[j] ~ d[j]/d[i] ~ m[i]/m[j]
        # i.e. n[i] is proportional to m[i]
        self.n = np.zeros((species.n, self.dim), dtype=int)
        for _s in range(species.n):
            # -/+1 are necessary to switch between points and length
            self.n[_s] = (m[_s]/m_min)*(velocity_grid.n-1) + 1
            # The center can be forced to be a grid point,
            # by allowing only uneven numbers per dimension
            if force_contain_center:
                for _d in range(0, self.dim):
                    if self.n[_s, _d] % 2 == 0:
                        self.n[_s, _d] -= 1
            if self.form is 'rectangular':
                self.size = self.n.prod(axis=1)
            else:
                print("ERROR - Unspecified Grid Shape")
                assert False

        # index[i] = n[0] + ... n[i-1]
        self.index = np.zeros((species.n+1,), dtype=int)
        for _s in range(species.n):
            self.index[_s+1] = self.index[_s] + self.size[_s]

        if create_grid:
            self.create_grid()
        else:
            self.G = np.zeros((0,), dtype=int)
        if check_integrity:
            self.check_integrity()
        return

    def create_grid(self):
        self.G = np.zeros(shape=(self.index[-1], self.dim),
                          dtype=int)
        for _s in range(self.n.shape[0]):
            _v = b_grd.Grid()
            _v.setup(self.dim,
                     self.n[_s],
                     self.d[_s],
                     form=self.form)
            _v.center()
            self.d[_s] = _v.d
            self.G[self.index[_s]:self.index[_s+1], :] = _v.G[:]
        self.multi = 2
        self.is_centered = True
        return

    #####################################
    #               Indexing            #
    #####################################
    def get_index(self,
                  specimen_index,
                  grid_entry):
        """Returns position of given grid_entry in :attr:`SVGrid.G`

        Firstly, we assume the given grid_entry is an element
        of the given Specimens Velocity-Grid.
        Under this condition we generate its index in attr:`SVGrid.G`.
        Secondly, we counter-check if the indexed value matches the given
        value.
        If this is true, we return the index,
        otherwise we return None.

        Parameters
        ----------
        specimen_index : int
        grid_entry : array(int)
            Array of shape=(self.dim,).

        Returns
        -------
        int
            Index/Position of grid_entry in :attr:`~SVGrid.G`.
        """
        # Todo Throw exception if not a grid entry (instead of None)
        i_flat = 0
        # get vector-index, by reversing Grid.center() - method
        i_vec = np.array((grid_entry+self.n[specimen_index]-1) // 2,
                         dtype=int)
        for _d in range(self.dim):
            n_loc = self.n[specimen_index, _d]
            i_flat *= n_loc
            i_flat += i_vec[_d]
        i_flat += self.index[specimen_index]
        if all(np.array(self.G[i_flat] == grid_entry)):
            return i_flat
        else:
            return None

    def get_specimen(self, velocity_index):
        """Returns Specimen (index) of given velocity index`

        Parameters
        ----------
        velocity_index : int

        Returns
        -------
        int
            Index of Specimen.
        """
        for i in range(self.index.size):
            if self.index[i] <= velocity_index < self.index[i+1]:
                return i
        # Todo Throw exception
        return None

    #####################################
    #           Verification            #
    #####################################

    def check_integrity(self):
        """Sanity Check"""
        s_n = self.d.shape[0]
        assert type(self.dim) is int
        assert self.dim in [1, 2, 3]
        assert type(self.d) is np.ndarray
        assert self.d.shape == (s_n,)
        assert self.d.dtype == float
        assert all(self.d > 0)
        assert type(self.n) is np.ndarray
        assert self.n.shape == (s_n, self.dim)
        assert self.n.dtype == int
        assert all(self.n.flatten() >= 2)
        assert type(self.size) is np.ndarray
        assert self.size.shape == (s_n,)
        assert self.size.dtype == int
        assert all(self.size >= 2)
        assert self.index.shape == (s_n+1,)
        assert np.array_equal(self.index[0:-1] + self.size,
                              self.index[1:])
        assert self.form in b_grd.Grid().supported_forms
        if self.form is 'rectangular':
            for _s in range(s_n):
                assert self.n[_s].prod() == self.size[_s]
        assert self.G.dtype == int
        assert self.G.shape == (self.index[-1], self.dim)
        assert self.offset.dtype == float
        assert self.offset.shape == (self.dim,)
        assert type(self.multi) is int
        assert self.multi >= 1
        assert (self.G % self.multi == 0).all
        assert type(self.is_centered) is bool
        if self.is_centered:
            assert self.multi % 2 is 0
        return

    def print(self,
              physical_grid=False):
        """Prints all Properties for Debugging Purposes"""
        print("Dimension = {}".format(self.dim))
        print("Index-Array = {}".format(self.index))
        print("Shape = {}".format(self.form))
        print("Multiplicator = {}".format(self.multi))
        print('Is centered Grid = {}'.format(self.is_centered))
        print("Offset = {}".format(self.offset))
        print('')
        for s in range(self.n.shape[0]):
            print('Specimen_{}'.format(s))
            print("Number of Total Grid Points = "
                  "{}".format(self.n[s, -1]))
            print("Grid Points per Dimension = "
                  "{}".format(self.n[s, 0:self.dim]))
            print("Step Size = {}".format(self.d[s]*self.multi))
            print("Boundaries:\n"
                  "{}".format(self.boundaries[s]))
            if physical_grid:
                print('Physical Grid :')
                beg = self.index[s]
                end = self.index[s + 1]
                print(self.G[beg:end]*self.d[s])
            print('')
