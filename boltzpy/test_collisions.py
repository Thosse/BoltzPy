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
    for (c, coll) in enumerate(old_coll.relations):
        assert sorted(coll) == sorted(new_coll.relations[c])
        colliding_species = {tc.sv.get_specimen(idx) for idx in coll}
        if len(colliding_species) == 1:
            permutations = bp.Collisions.INTRASPECIES_PERMUTATION
        else:
            permutations = bp.Collisions.INTERSPECIES_PERMUTATION
        assert any(np.all(coll[p] == new_coll.relations[c])
                   for p in permutations)
        assert old_coll.weights[c] == new_coll.weights[c]
    return

# Todo Test that complete >= convergent
# def test_issubset():
#     tc1 = bp_t.CASES[2]
#     tc2 = bp_t.CASES[3]
#     assert tc2.coll.issubset(tc1.coll)
#     assert not tc1.coll.issubset(tc2.coll)
#     return
