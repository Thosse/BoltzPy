import os
import h5py
import numpy as np
import pytest

import boltzpy.constants as bp_c
import boltzpy as bp

@pytest.mark.parametrize("test_case", bp_c.TEST_CASES)
def test_reflected_indices_inverse(test_case):
    # Compute Output in temporary file
    sim = bp.Simulation(test_case)
    for r in sim.geometry.rules:
        if not isinstance(r, bp.BoundaryPointRule):
            continue
        refl = r.reflected_indices_inverse
        assert np.all(refl[refl] == np.arange(refl.size))
        for (idx_v, v) in sim.sv.iMG:
            v_refl = sim.sv.iMG[refl[idx_v]]
            assert np.all(v == -v_refl)
