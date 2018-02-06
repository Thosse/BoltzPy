import numpy as np


class Species:
    """A simple class encapsulating data about species to be simulated.

    Attributes:
    -----------

        n (int):
            Denotes total number of specimen.
        mass (:obj:'np.ndarray'):
            Describes relative (not physical!) mass of each specimen.
            Array of shape=(self.n,) and dtype=int.
        alpha (:obj:'np.ndarray'):
            Describes Probabilities of collision between 2 specimen..
            Array of shape=(self.n, self.n) and dtype=float.
        names (:obj:'np.ndarray'):
            Denotes Names of Specimen.
            Used in Animation
            Array of shape=(self.n,) and dtype=str.
        colors (:obj:'np.ndarray'):
            Denotes Color of Specimen.
            Used in Animation
            Array of shape=(self.n,) and dtype=str.
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
