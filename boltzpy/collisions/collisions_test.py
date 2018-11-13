import h5py
import numpy as np
import pytest

import boltzpy.constants as bp_c
import boltzpy as bp


@pytest.mark.parametrize("test_case", bp_c.TEST_CASES)
def test_collisions(test_case):
    # Compute Output in temporary file
    sim = bp.Simulation(test_case)
    coll = bp.Collisions()
    coll.setup(sim.scheme, sim.sv, sim.s)
    # compare results
    assert sim.coll.size == coll.size
    assert np.array_equal(sim.coll.relations, coll.relations)
    assert np.array_equal(sim.coll.weights, coll.weights)
    return
