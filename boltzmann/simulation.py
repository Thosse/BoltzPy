
import boltzmann.configuration as b_cnf
import boltzmann.initialization as b_ini
import boltzmann.collisions.collision_relations as b_rel
import boltzmann.calculation as b_cal
import boltzmann.animation as b_ani
import boltzmann.constants as b_const

import os


class Simulation:
    """Handles all processes and data of a single simulation.

    Each instance correlates to a single file
    in which all parameters and results are  stored.
    An instance can be completely restored from its file.

    The more complex parts are delegated
    to its attributes.

    Parameters
    ----------
    file_name : :obj:`str`, optional
        Denotes the the file in which all simulation data will be stored.
        This can either be a file root or base name,
        in this case the file is placed in the
        :attr:`boltzmann.constants.DEFAULT_SIMULATION_PATH`,
        or a full file path.

    Attributes
    ----------
    configuration : :class:`boltzmann.configuration.Configuration`
    initialization : :class:`boltzmann.initialization.Initialization`
    collision_relations : :class:`boltzmann.collisions.CollisionRelations`
    calculation : :class:`boltzmann.calculation.Calculation`
    animation : :class:`boltzmann.animation.Animation`
    """
    def __init__(self, file_name=None):
        # Setup HDF5 File, this is necessary for the following constructors
        # Todo for GUI: check if file exists
        # Todo  if yes -> ask if edit (or choose other name?)
        # Todo  if no -> create uninitialized version

        # Todo assert proper file name, user has writing rights
        # Todo check_parameters (allow ".temp" name!)
        if file_name is None:
            # choose hidden filename in default directory
            self._file_directory = b_const.DEFAULT_DIRECTORY
            file_rood_idx = 1
            # find INDEX, s.t no file named "_unnamed_INDEX" exists
            while os.path.exists(self._file_directory
                                 + b_const.DEFAULT_FILE_ROOT
                                 + str(file_rood_idx)
                                 + ".hdf5"):
                file_rood_idx += 1
            self._file_root = b_const.DEFAULT_FILE_ROOT + str(file_rood_idx)
        else:
            # separate file directory and file root
            pos_of_file_root = file_name.rfind("/") + 1
            self._file_root = file_name[pos_of_file_root:]
            # if no directory given -> put it in the default directory
            if pos_of_file_root == 0:
                self._file_directory = b_const.DEFAULT_DIRECTORY
            else:
                self._file_directory = file_name[0:pos_of_file_root]

        # Submodules
        # Todo self.description = ... String that describes the simulation
        self.configuration = b_cnf.Configuration(self)
        self.initialization = b_ini.Initialization(self)
        # Todo self.schemes = ... Enum should be enough -> extra file
        self.collision_relations = b_rel.CollisionRelations(self)
        self.calculation = b_cal.Calculation(self)
        self.animation = b_ani.Animation(self)
        # Todo check_integrity (basic version)
        return

    @property
    def file_address(self):
        """:obj:`str` :
        Full path of the :class:`Simulation` file.
        """
        return self._file_directory + self._file_root + '.hdf5'
