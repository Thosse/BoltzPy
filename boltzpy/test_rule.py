import os
import h5py
import numpy as np
import pytest

import boltzpy.testcase as bp_t
import boltzpy as bp


@pytest.mark.parametrize("tf", bp_t.FILES)
def test_reflected_indices_inverse(tf):
    # Compute Output in temporary file
    sim = bp.Simulation.load(file_address=tf)
    for r in sim.geometry.rules:
        if not isinstance(r, bp.BoundaryPointRule):
            continue
        refl = r.reflected_indices_inverse
        assert np.all(refl[refl] == np.arange(refl.size))
        for (idx_v, v) in enumerate(sim.sv.iMG):
            v_refl = sim.sv.iMG[refl[idx_v]]
            assert np.all(v == -v_refl)
