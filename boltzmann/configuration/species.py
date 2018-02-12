import numpy as np


class Species:
    """A simple class encapsulating data about species to be simulated.

    Attributes
    ----------
    n : int
        Denotes total number of specimen.
    mass : np.ndarray
        Describes relative (not physical!) mass of each specimen.
        Array of shape (n,) and dtype int.
    alpha : np.ndarray
        Describes Probabilities of collision between 2 specimen..
        Array of shape=(n, n) and dtype float.
    names : np.ndarray
        Denotes Names of Specimen.
        Used in Animation
        Array of shape (n,) and dtype str.
    colors : np.ndarray
        Denotes Color of Specimen.
        Used in Animation
        Array of shape (n,) and dtype str.

    """
    COLOR_LIST = ['blue',       'red',      'green',
                  'yellow',     'black',    'brown',
                  'orange',     'pink',     'gray']

    def __init__(self,
                 float_data_type=np.float32,
                 integer_data_type=np.int32):
        self.n = 0
        self.mass = np.zeros(shape=(0,), dtype=integer_data_type)
        self.alpha = np.zeros(shape=(0, 0), dtype=float_data_type)
        self.names = []
        self.colors = []
        self.check_integrity()

    def add_specimen(self,
                     mass=1,
                     alpha_list=None,
                     name=None,
                     color=None):
        # assign default values and/or sanity check
        # sanity check for mass
        assert type(mass) is int and mass > 0
        # Give default value to alpha list and sanity check
        if alpha_list is None:
            # Todo Check if this is a good default parameter
            alpha_list = [1] * (self.n+1)
        else:
            assert type(alpha_list) is list
            assert all([type(_) in [int, float] for _ in alpha_list])
            assert len(alpha_list) is (self.n+1)
        # Give default value to name and sanity check
        if name is None:
            name = 'Specimen_' + str(self.n)
        else:
            assert type(name) is str
        # Give default value to color and sanity check
        if color is None:
            _free_colors = [c for c in Species.COLOR_LIST
                            if c not in self.colors]
            assert _free_colors is not [], "All Colors are used, add more."
            color = _free_colors[0]
        else:
            assert color in Species.COLOR_LIST, "Unsupported Color"
        # Actual Assignment of new values
        self.n += 1
        self.mass.resize(self.n)
        self.mass[-1] = mass
        # Add a row and a column to alpha
        _alpha = np.zeros(shape=(self.n, self.n),
                          dtype=self.alpha.dtype)
        _alpha[0:-1, 0:-1] = self.alpha
        self.alpha = _alpha
        self.alpha[-1, :] = np.array(alpha_list)
        self.alpha[:, -1] = np.array(alpha_list)
        self.names.append(name)
        self.colors.append(color)
        self.check_integrity()

    def check_integrity(self):
        assert type(self.n) is int
        assert self.n >= 0
        assert type(self.mass) is np.ndarray
        assert all(self.mass > 0)
        assert type(self.alpha) is np.ndarray
        assert self.alpha.shape == (self.n, self.n)
        assert type(self.names) is list
        assert len(self.names) is self.n
        assert all([type(name) is str for name in self.names])
        assert len(self.colors) is self.n
        assert type(self.colors) is list
        assert all([color in Species.COLOR_LIST for color in self.colors])

    def print(self):
        print("Number of Specimen = {}".format(self.n))
        print("Masses of Specimen = {}".format(self.mass))
        print("Collision-Factor-Matrix = \n{}".format(self.alpha))
        print("Names of Specimen  = {}".format(self.names))
        print("Colors of Specimen = {}".format(self.colors))
