
import boltzpy as bp
import boltzpy.testcase as bp_t
import pytest
import numpy as np


@pytest.mark.parametrize("test_case", bp_t.CASES)
def test_computation(test_case):
    print("TestCase = ", test_case["file_name"])
    assert isinstance(test_case, bp_t.TestCase)
    test_case.compare_results()
    return

@pytest.mark.parametrize("test_case", bp_t.CASES)
def test_instances(test_case):
    print("TestCase = ", test_case["file_name"])
    old_sim = bp.Simulation(test_case.address(test_case["file_name"]))
    assert test_case["s"] == old_sim.s
    assert test_case["t"] == old_sim.t
    assert test_case["p"] == old_sim.p
    assert test_case["sv"] == old_sim.sv
    assert test_case["geometry"] == old_sim.geometry
    assert test_case["scheme"] == old_sim.scheme
    assert np.all(test_case["output_parameters"] == old_sim.output_parameters)
    return
