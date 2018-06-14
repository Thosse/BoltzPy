

#: :obj:`str` : Default path for simulation files.
#: If only a file name is given, then the file will be located in this folder.
DEFAULT_SIMULATION_PATH = __file__[:-22] + 'Simulations/'

# TODO SUPPORTED_SHAPE_OF_ANIMATION_GRID

#: :obj:`set` of :obj:`str` :
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
               'Energy_Flow_Z'
               }

#: :obj:`set` of :obj:`str` :
#: Set of all currently supported schemes
#: for the selection of the collision partners.
SUPP_COLL_SELECTION_SCHEMES = {'Complete'}

#: :obj:`set` of :obj:`int` :
#: Set of all currently supported Convergence Orders
#: Quadrature Formula for Approximation of the Collision Operator.
SUPP_ORDERS_COLL = {1, 2, 3, 4}

#: :obj:`set` of :obj:`int` :
#: Set of all currently supported Convergence Orders
#: of the Operator Splitting.
SUPP_ORDERS_OS = {1, 2}

#: :obj:`set` of :obj:`int` :
#: Set of all currently supported Convergence Orders
#: Quadrature Formula for Approximation of the Collision Operator.
SUPP_ORDERS_TRANSP = {1, 2}

#: :obj:`set` of :obj:`str` :
#: Set of all characters, that are forbidden in any file addresses.
INVALID_CHARACTERS = {'.', '"', "'", '/', '§', '$', '&',
                      '+', '#', ',', ';', '\\', '`', '´'}
