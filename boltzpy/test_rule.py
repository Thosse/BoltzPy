import pytest
import os
import h5py
import numpy as np

import boltzpy.helpers.tests as test_helper
import boltzpy as bp


###################################
#           Setup Cases           #
###################################
DIRECTORY = __file__[:-20] + 'test_data/'
FILE = DIRECTORY + 'Rules.hdf5'
MODELS = {"2D_2Spc/Model": bp.SVGrid(
    masses=[2, 3],
    shapes=[[5, 5], [7, 7]],
    delta=1/8,
    spacings=[6, 4],
    collision_factors=[[50, 50], [50, 50]])}
RULES = dict()
RULES["2D_2Spc/ConstantPoint"] = bp.ConstantPointRule(
    initial_rho=[2, 2],
    initial_drift=[[0, 0], [0, 0]],
    initial_temp=[1, 1],
    affected_points=[0],
    velocity_grids=MODELS["2D_2Spc/Model"])
RULES["2D_2Spc/InnerPoint"] = bp.InnerPointRule(
    initial_rho=[1, 1],
    initial_drift=[[0, 0], [0, 0]],
    initial_temp=[1, 1],
    affected_points=np.arange(1, 19),
    velocity_grids=MODELS["2D_2Spc/Model"])
RULES["2D_2Spc/BoundaryPoint"] = bp.BoundaryPointRule(
    initial_rho=[1, 1],
    initial_drift=[[0, 0], [0, 0]],
    initial_temp=[1, 1],
    affected_points=[19],
    velocity_grids=MODELS["2D_2Spc/Model"],
    reflection_rate_inverse=[0.25, 0.25, 0.25, 0.25],
    reflection_rate_elastic=[0.25, 0.25, 0.25, 0.25],
    reflection_rate_thermal=[0.25, 0.25, 0.25, 0.25],
    absorption_rate=[0.25, 0.25, 0.25, 0.25],
    surface_normal=np.array([1, 0], dtype=int))


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the rules test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    file = h5py.File(file_address, mode="w")
    for (key, item) in RULES.items():
        assert isinstance(item, bp.Rule)
        file.create_group(key)
        item.save(file[key])

    # save models
    for group in file.keys():
        key_model = group + "/Model"
        for rule in file[group].keys():
            file[group][rule].attrs["Model"] = key_model
    return


#############################
#           Tests           #
#############################
def test_file_exists():
    assert os.path.exists(FILE), (
        "The test file {} is missing.".format(FILE))


def test_setup_creates_same_file():
    temp_file_address = DIRECTORY + '_tmp_.hdf5'
    setup_file(temp_file_address)
    test_helper.assert_files_are_equal([FILE, temp_file_address])
    os.remove(temp_file_address)
    return


@pytest.mark.parametrize("key", RULES.keys())
def test_hdf5_groups_exist(key):
    file = h5py.File(FILE, mode="r")
    assert key in file.keys(), (
        "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", RULES.keys())
def test_load_from_file(key):
    file = h5py.File(FILE, mode="r")
    hdf_group = file[key]
    old = bp.Rule.load(hdf_group)
    new = RULES[key]
    assert isinstance(old, bp.Rule)
    assert isinstance(new, bp.Rule)
    assert old == new, (
        "\n{}\nis not equal to\n\n{}".format(old, new)
    )


@pytest.mark.parametrize("key", RULES.keys())
def test_reflected_indices_inverse(key):
    rule = RULES[key]
    # get model
    model = MODELS[h5py.File(FILE, mode='r')[key].attrs["Model"]]
    if not isinstance(rule, bp.BoundaryPointRule):
        return
    refl = rule.reflected_indices_inverse
    assert np.all(refl[refl] == np.arange(refl.size))
    for (idx_v, v) in enumerate(model.iMG):
        v_refl = model.iMG[refl[idx_v]]
        assert np.all(v == -v_refl)


# Todo 2D_small from test_models does not work. What are proper criteria?
#  if possible: what parameter needs to change?
# Todo why doesn't this initialize?
# model = bp.SVGrid(
#     masses=[2, 3],
#     shapes=[[5, 5], [7, 7]],
#     delta=1/8,
#     spacings=[6, 4],
#     collision_factors=[[50, 50], [50, 50]])
# rule = bp.ConstantPointRule(
#     initial_rho=[2, 2],
#     initial_drift=[[0, 0], [0, 0]],
#     initial_temp=[1, 1],
#     affected_points=[0],
#     velocity_grids=model)

# Todo Allow Model=None?
#  or as class attribute?, probably a bad idea
#  attribute: normalized_state for boundary points
