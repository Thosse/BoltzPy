import h5py
import numpy as np
import pytest

import boltzpy.testcase as bp_t
import boltzpy as bp


@pytest.mark.parametrize("tc", bp_t.CASES)
def test_collisions(tc):
    # Compute Output in temporary file
    sim = bp.Simulation.load(file_address=tc.file_address)
    old_coll = sim.coll
    # new collisions are generated in the testcases already
    new_coll = tc.coll
    # sort collisions, to ignore different orders
    old_coll.sort()
    new_coll.sort()
    # compare results
    assert old_coll.size == new_coll.size
    assert np.array_equal(old_coll.relations, new_coll.relations)
    assert np.array_equal(old_coll.weights, new_coll.weights)
    assert old_coll == new_coll
    return


def test_issubset():
    tc1 = bp_t.CASES[2]
    tc2 = bp_t.CASES[3]
    assert tc2.coll.issubset(tc1.coll)
    assert not tc1.coll.issubset(tc2.coll)
    return
