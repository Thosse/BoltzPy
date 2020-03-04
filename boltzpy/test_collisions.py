import h5py
import numpy as np
import pytest

import boltzpy.testcase as bp_t
import boltzpy as bp


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_collisions(tf):
    # Compute Output in temporary file
    sim = bp.Simulation.load(file_address=tf)
    coll = bp.Collisions()
    coll.setup(sim.scheme, sim.sv, sim.s)
    # compare results
    assert sim.coll.size == coll.size
    assert np.array_equal(sim.coll.relations, coll.relations)
    assert np.array_equal(sim.coll.weights, coll.weights)
    return
