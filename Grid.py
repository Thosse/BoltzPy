import numpy as np


#############################################################################
#                 Creation of Space and Velocity Grids/Arrays               #
#############################################################################
class Grid:
    """A simple class for Positional-, Time- or Velocity-Grids.

    Attributes:
        dim (int):
            Grid dimensionality
        boundaries (:obj:'np.ndarray'):
            Describes (physical) size and position of the grid.
            Array of shape=(3,) and dtype=float
        n (:obj:'np.ndarray'):
            Describes number of grid points.
            Array of shape=(4,) and dtype=int
            x[0:2] denote number of points in x/y/z direction
            x[3] denotes total number of grid points
        d (float):
            Step size of the grid.
        shape (str):
            Shape of Grid can only be rectangular so far
        isInitialized (bool):
            Was the __init__ run successfully?
        isConstructed(bool):
            Was a make_grid-method run successfully?
    """
    # Add unit tests
    # Todo Add "check for consistency" method
    # Todo Add Circular shape
    GRID_SHAPES = ['rectangular']
    # Todo Add adaptive Grids
    isInitialized = False
    isConstructed = False

    def __calculate_number(self):
        if self.shape is 'rectangular':
            # Calculate number of grid points per dimension and total number
            _length = self.boundaries[:, 1] - self.boundaries[:, 0]
            _n_dim = np.array(np.ceil(_length / self.d) + 1,
                              dtype=int)
            self.n = _n_dim.prod()
            # TODO readjust self.d

    def __init__(self, dim, boundaries, d=None, n=None, shape='rectangular'):
        # check submitted dimensionality
        assert type(dim) is int
        assert dim in [1, 2, 3]
        # check submitted shape
        assert shape in ['rectangular']
        # check submitted boundary values
        assert type(boundaries) is list
        assert len(boundaries) is dim
        assert all(type(_) is list
                   and len(_) is 2
                   for _ in boundaries)
        assert all((type(element) in [int, float] for element in sublist)
                   for sublist in boundaries)
        assert all(sublist[1] > sublist[0] for sublist in boundaries)
        # check submitted step size
        if d is not None:
            assert n is None, \
                "ERROR - Either'd' or 'n' must be set, not Both!"
            assert type(d) is float
            assert d > 0
        # check submitted grid size
        if n is not None:
            assert d is None, \
                "ERROR - Either'd' or 'n' must be set, not Both!"
            assert type(n) is int
            assert n > 1

        self.isInitialized = True
        self.isConstructed = False
        self.dim = dim
        self.shape = shape
        self.boundaries = np.zeros((3, 2), dtype=float)
        self.boundaries[0:dim, :] = np.array(boundaries)
        if d is not None and n is None:
            self.d = d
            self.__calculate_number()
        else:
            assert False, "This is to be done!"
            # self.d = d
            # self.n = n

    def __make_rectangular_grid(self):
        assert self.shape == 'rectangular'
        # Calculate number of grid points per dimension and total number
        _length = self.boundaries[:, 1] - self.boundaries[:, 0]
        _n_dim = np.array(np.ceil(_length / self.d) + 1,
                          dtype=int)
        _grid_dimension = (_n_dim.prod(), self.dim)
        self.n = _n_dim.prod()

        # Create list of 1D grids for each dimension
        _list_of_1D_grids = [np.linspace(self.boundaries[i_d, 0],
                                         self.boundaries[i_d, 1],
                                         _n_dim[i_d])
                             for i_d in range(self.dim)]
        # Create mesh grid from 1D grids
        _mesh_list = np.meshgrid(*_list_of_1D_grids)    # *[a,b,c] = a,b,c
        grid = np.array(_mesh_list)
        # bring meshgrid into desired order/structure
        if self.dim is 2:
            grid = np.array(grid.transpose((2, 1, 0)))
        elif self.dim is 3:
            grid = np.array(grid.transpose((2, 1, 3, 0)))
        elif self.dim is not 1:
            print("Error")
        print(grid.shape)
        grid = grid.reshape(_grid_dimension)
        self.isConstructed = True
        return grid

    def make_grid(self):
        if self.shape is 'rectangular':
            return self.__make_rectangular_grid()

    def print(self):
        print("Dimension = {}".format(self.dim))
        print("Boundaries = {}".format(self.boundaries))
        print("Number of Grid Points = {}".format(self.n))
        print("Step Size = {}".format(self.d))
        print("Shape = {}".format(self.shape))
        print("Grid was initialized = {}".format(self.isInitialized))
        print("Grid was Constructed = {}".format(self.isConstructed))
        print("")


#############################################################################
#                           Species-Class                                   #
#############################################################################
class Species:
    """A simple class encapsulating data about species to be simulated.

    Attributes:
        n (int):
            Denotes total number of specimen.
        mass (:obj:'np.ndarray'):
            Describes relative (not physical!) mass of each specimen.
            Array of shape=(self.n,) and dtype=int.
        alpha (:obj:'np.ndarray'):
            Describes Probabilities of collision between 2 specimen..
            Array of shape=(self.n, self.n) and dtype=float.
        names (:obj:'np.ndarray'):
            = np.zeros(shape=(0,), dtype=str)
        colors (:obj:'np.ndarray'):
            = np.zeros(shape=(0,), dtype=str)
    """
    COLOR_LIST = ['blue',       'red',      'green',
                  'yellow',     'black',    'brown',
                  'orange',     'pink',     'gray']

    def __init__(self):
        self.n = 0
        self.mass = np.zeros(shape=(0,), dtype=int)
        self.alpha = np.zeros(shape=(0, 0), dtype=float)
        self.names = []
        self.colors = []

    def add_specimen(self,
                     mass=1,
                     alpha_list=None,
                     name=None,
                     color=None):
        # check submitted mass
        assert type(mass) is int and mass > 0
        # check submitted alpha_list
        if alpha_list is None:
            alpha_list = [1] * (self.n+1)
        else:
            assert type(alpha_list) is list
            assert all([type(_) in [int, float] for _ in alpha_list])
            assert len(alpha_list) is (self.n+1)
        # check submitted name
        if name is None:
            name = 'Specimen_' + str(self.n)
        else:
            assert type(name) is str
        # check submitted color
        if color is None:
            _free_colors = [c for c in Species.COLOR_LIST
                            if c not in self.colors]
            assert _free_colors is not [], "All Colors are used, add more."
            color = _free_colors[0]
        else:
            assert color in Species.COLOR_LIST, "Unsupported Color"

        self.n += 1
        self.mass.resize(self.n)
        self.mass[-1] = mass
        # Add a row and a col to alpha
        _alpha = np.zeros((self.n, self.n), dtype=float)
        _alpha[0:-1, 0:-1] = self.alpha
        self.alpha = _alpha
        self.alpha[-1, :] = np.array(alpha_list)
        self.alpha[:, -1] = np.array(alpha_list)
        self.names.append(name)
        self.colors.append(color)

    def print(self):
        print("Number of Specimen = {}".format(self.n))
        print("Masses of Specimen = {}".format(self.mass))
        print("Collision-Factor-Matrix = {}".format(self.alpha))
        print("Names of Specimen = {}".format(self.names))
        print("Colors of Specimen = {}".format(self.colors))
        print("")

# class PSVGrid:
#
#
# t = Grid(3, [[-1, 1], [-1, 1],[-1, 1]], d=1.0)
# print(t.dim)
# print(t.boundaries)
# print(t.d)
# print(t.n)
# print(t.shape)
# print(t.make_rectangular_grid())
#
# s = Species()
# s.add_specimen()
# s.add_specimen(name='test')
# s.add_specimen(5, [1, 2, 3], 'test2')
#
# s.add_specimen()
# s.add_specimen()
# print(s.n)
# print(s.names[:])
# print(s.mass)
# print(s.colors)
# print(s.alpha)
