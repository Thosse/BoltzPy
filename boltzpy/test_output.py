import os
import h5py
import numpy as np
import pytest

import boltzpy.constants as bp_c
import boltzpy as bp


@pytest.mark.parametrize("test_case", bp_c.TEST_CASES)
def test_computation(test_case):
    # Compute Output in temporary file
    sim = bp.Simulation(test_case)
    sim.save(bp_c.TEST_TMP_FILE)
    sim.compute()
    # Open old and new file, to compare results
    old_file = h5py.File(test_case, mode='r')
    new_file = h5py.File(bp_c.TEST_TMP_FILE, mode='r')
    # compare results
    try:
        for species_name in new_file["results"].keys():
            for output in new_file["results"][species_name].keys():
                key = "results/{}/{}".format(species_name, output)
                old_results = old_file[key][()]
                new_results = new_file[key][()]
                assert old_results.shape == new_results.shape, (
                    "Results differ in shape:\t {} != {}".format(
                        old_results.shape, new_results.shape)
                )
                assert np.array_equal(old_results, new_results), (
                    "Results differ by {} (sum over absolutes)".format(
                        np.sum(abs(old_results - new_results)))
                )
    finally:
        os.remove(bp_c.TEST_TMP_FILE)
    return
