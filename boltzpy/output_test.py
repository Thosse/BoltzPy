import boltzpy.constants as b_const
import boltzpy.simulation as b_sim
import os
import h5py
import numpy as np

directory = b_const.TEST_DIRECTORY
test_cases = [os.path.join(directory, file)
              for file in os.listdir(directory)
              if os.path.isfile(os.path.join(directory, file))
              and file[-5:] == ".hdf5"]


def test_computation():
    for case in test_cases:
        sim = b_sim.Simulation(case)
        sim.run_computation("Test")
        hdf5_file = h5py.File(case, mode='r')
        for output in sim.output_parameters.flatten():
            print(output)
            results_old = hdf5_file["Computation"][output].value
            results_new = hdf5_file["Test"][output].value
            assert np.array_equal(results_old, results_new)
    return
