
import boltzpy.constants as b_const
import os


def split_address(file_address=None):
    if file_address is None:
        # choose default directory
        file_directory = b_const.DEFAULT_DIRECTORY
        # choose default file_root
        # find unused index to attach to file_root
        file_root_idx = 1
        while os.path.exists(file_directory
                             + b_const.DEFAULT_FILE_ROOT
                             + str(file_root_idx)
                             + ".hdf5"):
            file_root_idx += 1
        file_root = b_const.DEFAULT_FILE_ROOT + str(file_root_idx)
    else:
        # separate file directory and file root
        pos_of_file_root = file_address.rfind("/") + 1
        file_root = file_address[pos_of_file_root:]
        if pos_of_file_root == 0:
            # if no directory given -> put it in the default directory
            file_directory = b_const.DEFAULT_DIRECTORY
        else:
            file_directory = file_address[0:pos_of_file_root]

    # remove hdf5 ending, if any
    if file_root[-5:] == '.hdf5':
        file_root = file_root[:-5]

    return[file_directory, file_root]
