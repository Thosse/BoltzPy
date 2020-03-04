import pytest

import boltzpy.testcase as bp_t
import boltzpy as bp


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_eval_of_repr_is_equal(tf):
    sim = bp.Simulation.load(file_address=tf)
    scheme_old = sim.scheme
    scheme_new = eval(scheme_old.__repr__('bp'))
    assert isinstance(scheme_old, bp.Scheme)
    assert isinstance(scheme_new, bp.Scheme)
    assert scheme_new == scheme_old
    return
