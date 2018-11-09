import boltzpy.constants as b_const
import boltzpy.simulation as b_sim
import h5py
import numpy as np
import pytest


@pytest.mark.parametrize("test_case", b_const.TEST_CASES)
def test_collision_relations(test_case):
    # Compute Output in temporary file
    sim = b_sim.Simulation(test_case)
    sim.collision_relations = np.zeros((0, 4), dtype=int)
    sim.collision_weights = np.zeros((0,), dtype=float)
    sim.save(b_const.TEST_TMP_FILE)
    # Todo will change over the next commits
    sim.run_computation("Test")
    # Open old and new file, to compare results
    old_file = h5py.File(test_case, mode='r')
    new_file = h5py.File(b_const.TEST_TMP_FILE, mode='r')
    # compare results
    for key in ['Relations', 'Weights']:
        results_old = old_file["Collisions"][key].value
        results_new = new_file["Collisions"][key].value
        assert results_old.shape == results_new.shape
        assert np.array_equal(results_old, results_new)
    return
