import pytest
import os
import h5py
import numpy as np

import boltzpy.helpers.tests as test_helper
import boltzpy as bp
from boltzpy.test_grid import GRIDS
from boltzpy.test_model import MODELS
from boltzpy.test_geometry import GEOMETRIES
from boltzpy.test_BaseClass import DIRECTORY, TMP_FILE


###################################
#           Setup Cases           #
###################################
def file_path(name):
    assert isinstance(name, str)
    return DIRECTORY + "Simulation/" + name + ".hdf5"


SIM_PARAMS = [
    (file_path("2D_small"),
     {"timing": GRIDS["Grid/2D_small"],
      "geometry": GEOMETRIES["Geometry/2D_small"],
      "model": MODELS["Model/2D_small"],
      "log_state": True}),
    (file_path("equalMass"),
     {"timing": GRIDS["Grid/equalMass"],
      "geometry": GEOMETRIES["Geometry/equalMass"],
      "model": MODELS["Model/equalMass"],
      "log_state": True})]

SAVED_MOMENTS = [
    "number_density",
    "momentum",
    "mean_velocity",
    "energy_density",
    "temperature",
    "momentum_flow",
    "energy_flow"]


def setup_files():
    reply = input("You are about to reset the simulations test files. "
                  "Are you Sure? (yes, no)\n")
    if reply != "yes":
        print("ABORTED")
        return

    for (path, params) in SIM_PARAMS:
        file = h5py.File(path, mode="w")
        sim = bp.Simulation(**params)
        assert isinstance(sim, bp.Simulation)
        sim.compute(file)
    print("Sucessfully set up all Test files")
    input("Press any key to continue!")
    return


@pytest.fixture(params=SIM_PARAMS, scope='class')
def simulation(request):
    (path, params) = request.param
    assert os.path.exists(path)
    with h5py.File(path, mode="r") as old_file:
        with h5py.File(TMP_FILE, mode="w") as new_file:
            sim = bp.Simulation(**params)
            sim.compute(hdf_group=new_file)
            # inject class variables
            request.cls.sim = sim
            request.cls.new_file = new_file
            request.cls.old_file = old_file
            # pass request.cls as self to the test class
            yield
    # teardown
    os.remove(TMP_FILE)


@pytest.mark.usefixtures("simulation")
class TestSimulation:
    def test_new_file_equals_old_file(self):
        test_helper.assert_hdf_groups_are_equal(self.new_file.file,
                                                self.old_file.file)
        return

    def test_load_from_file(self):
        old_sim = bp.Simulation.load(self.new_file)
        new_sim = self.sim
        assert isinstance(old_sim, bp.Simulation)
        assert isinstance(new_sim, bp.Simulation)
        assert old_sim == new_sim

    def test_computed_same_output_moments(self):
        old_keys = self.old_file["results"].keys()
        new_keys = self.new_file["results"].keys()
        assert old_keys == new_keys

    def test_computed_states_are_equal(self):
        old_state = self.old_file["results"]["state"][()]
        new_state = self.new_file["results"]["state"][()]
        assert np.allclose(old_state, new_state)

    @pytest.mark.parametrize("moment", SAVED_MOMENTS)
    def test_computed_moments_are_equal(self, moment):
        old_moment = self.old_file["results"][moment][()]
        new_moment = self.new_file["results"][moment][()]
        assert np.allclose(old_moment, new_moment)

    @pytest.mark.parametrize("moment", SAVED_MOMENTS)
    def test_computing_moments_on_old_state_gives_old_results(self, moment):
        old_moment = self.old_file["results"][moment][()]
        # compute moment from old_state
        old_state = self.old_file["results"]["state"][()]
        model = self.sim.model
        cmp_moment = model.__getattribute__("cmp_" + moment)
        for s in model.species:
            new_moment = cmp_moment(old_state, s)
            assert np.allclose(old_moment[:, :, s], new_moment)
