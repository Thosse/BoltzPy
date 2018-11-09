import os
import h5py
import numpy as np
import pytest

import boltzpy.constants as bp_c
import boltzpy.simulation as bp


@pytest.mark.parametrize("test_case", bp_c.TEST_CASES)
def test_computation(test_case):
    # Compute Output in temporary file
    sim = bp.Simulation(test_case)
    sim.save(bp_c.TEST_TMP_FILE)
    sim.run_computation("Test")
    # Open old and new file, to compare results
    old_file = h5py.File(test_case, mode='r')
    new_file = h5py.File(bp_c.TEST_TMP_FILE, mode='r')
    # compare results
    try:
        for output in sim.output_parameters.flatten():
            results_old = old_file["Computation"][output].value
            results_new = new_file["Test"][output].value
            assert results_old.shape == results_new.shape
            assert np.array_equal(results_old, results_new)
    finally:
        os.remove(bp_c.TEST_TMP_FILE)
    return
