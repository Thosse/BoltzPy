
import boltzpy.testcase as bp_t
import pytest


@pytest.mark.parametrize("test_case", bp_t.CASES)
def test_computation(test_case):
    print("tc = ", test_case["file_name"])
    assert isinstance(test_case, bp_t.TestCase)
    test_case.compare_results()
    return

