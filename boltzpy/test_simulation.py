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

FILE = DIRECTORY + 'Simulations.hdf5'
_open_file = h5py.File(FILE, mode="r")
SIMULATIONS = dict()
SIMULATIONS["2D_small/Simulation"] = bp.Simulation(
    timing=GRIDS["Grid/2D_small"],
    geometry=GEOMETRIES["Geometry/2D_small"],
    model=MODELS["Model/2D_small"],
    file=_open_file["2D_small/Simulation"],
    log_state=True)
SIMULATIONS["equalMass/Simulation"] = bp.Simulation(
    timing=GRIDS["Grid/equalMass"],
    geometry=GEOMETRIES["Geometry/equalMass"],
    model=MODELS["Model/equalMass"],
    file=_open_file["equalMass/Simulation"],
    log_state=True)

SIM_PARAMS = [
    ("2D_small/Simulation", {"timing": GRIDS["Grid/2D_small"],
                             "geometry": GEOMETRIES["Geometry/2D_small"],
                             "model": MODELS["Model/2D_small"],
                             "log_state": True}),
    ("equalMass/Simulation", {"timing": GRIDS["Grid/equalMass"],
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


# Todo splitte file in separate dateien, pro simulation eins
def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the simulations test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return
        else:
            _open_file.close()

    with h5py.File(file_address, mode="w") as file:
        for (key, item) in SIMULATIONS.items():
            assert isinstance(item, bp.Simulation)
            file.create_group(key)
            item.compute(file[key])
            item.file = file[key]
    return


def test_setup_creates_same_file():
    setup_file(TMP_FILE)
    test_helper.assert_files_are_equal([FILE, TMP_FILE])
    return


@pytest.fixture(params=SIM_PARAMS, scope='class')
def simulation(request):
    (key, params) = request.param
    with h5py.File(TMP_FILE, mode="w") as new_file:
        with h5py.File(FILE, mode="r") as old_file:
            sim = bp.Simulation(file=new_file, **params)
            sim.compute(hdf_group=new_file)
            # inject class variables
            request.cls.sim = sim
            request.cls.new_file = new_file
            request.cls.old_file = old_file[key]
            # pass request.cls as self to the test class
            yield
    # teardown
    os.remove(TMP_FILE)


@pytest.mark.usefixtures("simulation")
class TestSimulation:
    def test_old_file_exists(self):
        assert os.path.exists(FILE), (
            "The test file {} is missing.".format(FILE))

    # def test_hdf5_groups_exist(self):
    #     with h5py.File(FILE, mode="r") as file:
    #         assert key in file.keys(), (
    #             "The group {} is missing in the test file-".format(key))

    def test_load_from_file(self):
        with h5py.File(FILE, mode="r") as file:
            old_sim = bp.Simulation.load(self.old_file)
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
