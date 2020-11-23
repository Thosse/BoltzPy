import pytest
import os
import h5py
import numpy as np

import boltzpy.helpers.tests as test_helper
import boltzpy as bp
from boltzpy.test_model import MODELS


###################################
#           Setup Cases           #
###################################
FILE = test_helper.DIRECTORY + 'Rules.hdf5'
RULES = dict()
# Test rules allow up to 4 specimen
RULES["2D_small/LeftConstant"] = bp.ConstantPointRule(
    particle_number=[2, 2],
    mean_velocity=[[0, 0], [0, 0]],
    temperature=[1, 1],
    affected_points=[0],
    model=MODELS["2D_small/Model"])
RULES["2D_small/Interior"] = bp.InnerPointRule(
    particle_number=[1, 1],
    mean_velocity=[[0, 0], [0, 0]],
    temperature=[1, 1],
    affected_points=np.arange(1, 9),
    model=MODELS["2D_small/Model"])
RULES["2D_small/RightBoundary"] = bp.BoundaryPointRule(
    particle_number=[1, 1],
    mean_velocity=[[0, 0], [0, 0]],
    temperature=[1, 1],
    affected_points=[9],
    model=MODELS["2D_small/Model"],
    reflection_rate_inverse=[0.25, 0.25, 0.25, 0.25],
    reflection_rate_elastic=[0.25, 0.25, 0.25, 0.25],
    reflection_rate_thermal=[0.25, 0.25, 0.25, 0.25],
    absorption_rate=[0.25, 0.25, 0.25, 0.25],
    surface_normal=np.array([1, 0], dtype=int))
RULES["equalMass/LeftBoundary"] = bp.BoundaryPointRule(
    particle_number=[2, 2],
    mean_velocity=[[0, 0], [0, 0]],
    temperature=[1, 1],
    affected_points=[0],
    model=MODELS["equalMass/Model"],
    reflection_rate_inverse=[0.45, 0.45, 0.45, 0.45],
    reflection_rate_elastic=[0.45, 0.45, 0.45, 0.45],
    reflection_rate_thermal=[0.1, 0.1, 0.1, 0.1],
    absorption_rate=[0, 0, 0, 0],
    surface_normal=np.array([-1, 0], dtype=int))
RULES["equalMass/LeftInterior"] = bp.InnerPointRule(
    particle_number=[2, 2],
    mean_velocity=[[0, 0], [0, 0]],
    temperature=[1, 1],
    affected_points=np.arange(1, 5),
    model=MODELS["equalMass/Model"])
RULES["equalMass/RightInterior"] = bp.InnerPointRule(
    particle_number=[1, 1],
    mean_velocity=[[0, 0], [0, 0]],
    temperature=[1, 1],
    affected_points=np.arange(5, 9),
    model=MODELS["equalMass/Model"])
RULES["equalMass/RightBoundary"] = bp.BoundaryPointRule(
    particle_number=[1, 1],
    mean_velocity=[[0, 0], [0, 0]],
    temperature=[1, 1],
    affected_points=[9],
    model=MODELS["equalMass/Model"],
    reflection_rate_inverse=[0.15, 0.15, 0.15, 0.15],
    reflection_rate_elastic=[0.15, 0.15, 0.15, 0.15],
    reflection_rate_thermal=[0.15, 0.15, 0.15, 0.15],
    absorption_rate=[0.55, 0.55, 0.55, 0.55],
    surface_normal=np.array([1, 0], dtype=int))

# Sub dictionaries for specific attribute tests
CLASSES = {name: {key: item for (key, item) in RULES.items()
                  if isinstance(item, subclass)}
           for (name, subclass) in bp.Rule.subclasses().items()}


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the rules test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    with h5py.File(file_address, mode="w") as file:
        for (key, item) in RULES.items():
            assert isinstance(item, bp.Rule)
            file.create_group(key)
            item.save(file[key], True)

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
    setup_file(test_helper.TMP_FILE)
    test_helper.assert_files_are_equal([FILE, test_helper.TMP_FILE])
    os.remove(test_helper.TMP_FILE)
    return


@pytest.mark.parametrize("key", RULES.keys())
def test_hdf5_groups_exist(key):
    with h5py.File(FILE, mode="r") as file:
        assert key in file.keys(), (
            "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", RULES.keys())
def test_load_from_file(key):
    with h5py.File(FILE, mode="r") as file:
        hdf_group = file[key]
        old = bp.Rule.load(hdf_group)
        new = RULES[key]
        assert isinstance(old, bp.Rule)
        assert isinstance(new, bp.Rule)
        assert old == new


@pytest.mark.parametrize("key", CLASSES["InnerPointRule"].keys())
@pytest.mark.parametrize("attribute", bp.InnerPointRule.attributes())
def test_attributes_of_inner_points(attribute, key):
    with h5py.File(FILE, mode="r") as file:
        old = file[key][attribute][()]
        new = RULES[key].__getattribute__(attribute)
        if isinstance(new, np.ndarray) and (new.dtype == float):
            assert np.allclose(old, new)
        else:
            assert np.all(old == new)


@pytest.mark.parametrize("key", CLASSES["ConstantPointRule"].keys())
@pytest.mark.parametrize("attribute", bp.ConstantPointRule.attributes())
def test_attributes_of_constant_points(attribute, key):
    with h5py.File(FILE, mode="r") as file:
        old = file[key][attribute][()]
        new = RULES[key].__getattribute__(attribute)
        if isinstance(new, np.ndarray) and (new.dtype == float):
            assert np.allclose(old, new)
        else:
            assert np.all(old == new)


@pytest.mark.parametrize("key", CLASSES["BoundaryPointRule"].keys())
@pytest.mark.parametrize("attribute", bp.BoundaryPointRule.attributes())
def test_attributes_of_boundary_points(attribute, key):
    with h5py.File(FILE, mode="r") as file:
        old = file[key][attribute][()]
        new = RULES[key].__getattribute__(attribute)
        if isinstance(new, np.ndarray) and (new.dtype == float):
            assert np.allclose(old, new)
        else:
            assert np.all(old == new)


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


@pytest.mark.parametrize("key", CLASSES["BoundaryPointRule"].keys())
def test_reflection_keeps_total_mass(key):
    rule = RULES[key]
    # get model
    model_key = h5py.File(FILE, mode='r')[key].attrs["Model"]
    model = MODELS[model_key]
    for _ in range(100):
        inflow = np.zeros((1, model.size))
        n_incoming_vels = rule.incoming_velocities.size
        rand_vals = np.random.random(n_incoming_vels)
        inflow[..., rule.incoming_velocities] = rand_vals
        reflected_inflow = rule.reflection(inflow, model)
        for s in model.species:
            mass_in = model.mass_density(inflow, s)
            mass_refl = model.mass_density(reflected_inflow, s)
            absorption = rule.absorption_rate[s]
            assert np.isclose((1 - absorption) * mass_in, mass_refl)


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
#     particle_number=[2, 2],
#     mean_velocity=[[0, 0], [0, 0]],
#     temperature=[1, 1],
#     affected_points=[0],
#     model=model)

# Todo Allow Model=None?
#  or as class attribute?, probably a bad idea
#  attribute: normalized_state for boundary points
