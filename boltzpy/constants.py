import os

#: :obj:`str` : Default directory for simulation files.
#: If no full path is given,
#: then the file will be located in this directory.
DEFAULT_DIRECTORY = __file__[:-20] + 'Simulations/'

#: :obj:`str` : Default directory for all test files.
#: All data used for testing purposes is stored in this directory.
TEST_DIRECTORY = __file__[:-20] + 'test_data/'

#: :obj:`str` : Default file (address) for temporary test results.
TEST_TMP_FILE = TEST_DIRECTORY + '_tmp_.hdf5'

#: :obj:`list` [:obj:`str`] : Contains all available test cases
#: in the :const:`TEST_DIRECTORY`.
TEST_CASES = [os.path.join(TEST_DIRECTORY, file)
              for file in os.listdir(TEST_DIRECTORY)
              if os.path.isfile(os.path.join(TEST_DIRECTORY, file))
              and file[-5:] == '.hdf5'
              and file != '_tmp_.hdf5']

#: :obj:`str` : Default file root for simulation files.
#: If no file name is given at all, then the file root will be this.
DEFAULT_FILE_ROOT = "_unnamed_"

#: :obj:`tuple` [:obj:`int`] : Default aspect ratio for plots in
#: :class:`~boltzpy.animation.Animation`.
DEFAULT_FIGSIZE = (16, 9)

#: :obj:`int` : Default Resolution for plots in
#: :class:`~boltzpy.animation.Animation`.
DEFAULT_DPI = 300

#: :obj:`set` [:obj:`str`] :
#: Set of all currently supported moments.
# A set is faster than lists for the __contains__/in operation.
SUPP_OUTPUT = {'Mass',
               'Momentum_X',
               'Momentum_Y',
               'Momentum_Z',
               'Momentum_Flow_X',
               'Momentum_Flow_Y',
               'Momentum_Flow_Z',
               'Energy',
               'Energy_Flow_X',
               'Energy_Flow_Y',
               'Energy_Flow_Z',
               "Complete_Distribution"
               }

#: :obj:`list` [:obj:`str`]:
#: List of all currently supported Colors.
SUPP_COLORS = ['blue', 'red', 'green',
               'yellow', 'brown', 'gray',
               'olive', 'purple', 'cyan',
               'orange', 'pink', 'lime',
               'black']

#: :obj:`set` [:obj:`str`] :
#: Set of all characters, that are forbidden in any file addresses.
INVALID_CHARACTERS = {'.', '"', "'", '/', '§', '$', '&',
                      '+', '#', ',', ';', '\\', '`', '´'}

#: :obj:`set` [:obj:`int`] :
#: Set of all currently supported
#: for :class:`~boltzpy.Grid`
#: dimensions.
SUPP_GRID_DIMENSIONS = {1, 2}

#: :obj:`list` [:obj:`str`] :
#: List of all currently supported categories
#: for :class:`Position-Space-Grid <boltzpy.Grid>` points
#: (e.g. inner points, boundary points,...).
#:
#: Each P-Grid point fits into exactly one of the following categories:
#:      * **Inner Point (Default)**:
#:          * both transport and collisions are applied normally
# TODO add types:
#      * **Boundary Point**:
#          * no collision step
#          * additional reflection step after every transport step
#      * **Ghost Boundary Point**:
#          * for higher order transport
#          * so far undetermined behaviour
#      * **Constant Input/Output Point**:
#          * no collision-step
#          * no transport-step
#          * Distribution is constant over the whole simulation.
#      * **Time Variant Input/Output Point**:
#          * no collision-step,
#          * no transport-step,
#          * Distribution is freshly initialized in every time step
SUPP_BEHAVIOUR_TYPES = ['Inner Point',
                        'Constant Point',
                        'Boundary Point',
                        # 'Ghost Boundary_Point',
                        # 'Time_Variant_Point'
                        ]
