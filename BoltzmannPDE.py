import Initialization as bI
# import Setup as bS
# import Calculation as bC
# import Animation as bA


class BoltzmannPDE:
    """Main class, Encapsulates all Functionalities

    Attributes:
        flag_GPU (bool):
            Decides whether to compute on GPU or CPU
        single_v_grid (bool):
            Decides whether to use equal velocity grids for all specimen
        initialize (:obj:Initialization)
            Custom Class, encapsulates Input of:
                Time, Position and Velocity Data
                Specimen Data
    """
    def __init__(self,
                 use_gpu=False,
                 use_single_grid=False):
        self.__flag_GPU = use_gpu
        self.__single_v_grid = use_single_grid
        # Todo set fType depending on use_gpu, and submit to initialize
        self.initialize = bI.Initialization()

    def print(self):
        print("Todo")
