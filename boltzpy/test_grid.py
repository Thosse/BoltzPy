import numpy as np
import pytest

import boltzpy.testcase as bp_t
import boltzpy as bp


@pytest.mark.parametrize("tc", bp_t.CASES)
def test_get_idx(tc):
    # define alternative function
    def alt_func(self, values):
        assert isinstance(self, bp.Grid)
        assert isinstance(values, np.ndarray)
        assert values.ndim in [1, 2]
        if values.ndim == 1:
            values = [values]
        results = []
        for value in values:
            grid_iterator = (idx for (idx, val) in enumerate(self.iG)
                             if np.all(val == value))
            local_index = next(grid_iterator, -1)
            results.append(local_index)
        return np.array(results)

    for grid_1 in tc.sv.vGrids:
        for grid_2 in tc.sv.vGrids:
            result = grid_1.get_idx(grid_2.iG)
            alternative = alt_func(grid_1, grid_2.iG)
            assert np.all(result == alternative)
    return
