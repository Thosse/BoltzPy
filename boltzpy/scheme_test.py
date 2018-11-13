import pytest

import boltzpy.constants as bp_c
import boltzpy as bp


@pytest.mark.parametrize("test_case", bp_c.TEST_CASES)
def test_eval_or_repr_is_equal(test_case):
    sim = bp.Simulation(test_case)
    scheme_old = sim.scheme
    scheme_new = eval(scheme_old.__repr__('bp'))
    assert scheme_new.keys() == scheme_old.keys()
    assert scheme_old.items() == scheme_new.items()
    assert scheme_new == scheme_old
    return
