import pytest
import numpy as np
import boltzpy as bp
from tests.test_model import MODELS


###################################
#           Setup Cases           #
###################################
RULES = dict()
RULES["2D_small/LeftConstant"] = bp.ConstantPointRule(
    number_densities=[2, 2],
    mean_velocities=[[0, 0], [0, 0]],
    temperatures=[1, 1],
    affected_points=[0],
    **MODELS["Model/2D_small"].__dict__)
RULES["2D_small/Interior"] = bp.InnerPointRule(
    number_densities=[1, 1],
    mean_velocities=[[0, 0], [0, 0]],
    temperatures=[1, 1],
    affected_points=np.arange(1, 9),
    **MODELS["Model/2D_small"].__dict__)
RULES["2D_small/RightBoundary"] = bp.BoundaryPointRule(
    number_densities=[1, 1],
    mean_velocities=[[0, 0], [0, 0]],
    temperatures=[1, 1],
    affected_points=[9],
    refl_inverse=[0.25, 0.25],
    refl_elastic=[0.25, 0.25],
    refl_thermal=[0.25, 0.25],
    refl_absorbs=[0.25, 0.25],
    surface_normal=np.array([1, 0], dtype=int),
    **MODELS["Model/2D_small"].__dict__)
RULES["equalMass/LeftBoundary"] = bp.BoundaryPointRule(
    number_densities=[2, 2],
    mean_velocities=[[0, 0], [0, 0]],
    temperatures=[1, 1],
    affected_points=[0],
    refl_inverse=[0.45, 0.45],
    refl_elastic=[0.45, 0.45],
    refl_thermal=[0.1, 0.1],
    refl_absorbs=[0, 0],
    surface_normal=np.array([-1, 0], dtype=int),
    **MODELS["Model/equalMass"].__dict__)
RULES["equalMass/LeftInterior"] = bp.InnerPointRule(
    number_densities=[2, 2],
    mean_velocities=[[0, 0], [0, 0]],
    temperatures=[1, 1],
    affected_points=np.arange(1, 5),
    **MODELS["Model/equalMass"].__dict__)
RULES["equalMass/RightInterior"] = bp.InnerPointRule(
    number_densities=[1, 1],
    mean_velocities=[[0, 0], [0, 0]],
    temperatures=[1, 1],
    affected_points=np.arange(5, 9),
    **MODELS["Model/equalMass"].__dict__)
RULES["equalMass/RightBoundary"] = bp.BoundaryPointRule(
    number_densities=[1, 1],
    mean_velocities=[[0, 0], [0, 0]],
    temperatures=[1, 1],
    affected_points=[9],
    refl_inverse=[0.15, 0.15],
    refl_elastic=[0.15, 0.15],
    refl_thermal=[0.15, 0.15],
    refl_absorbs=[0.55, 0.55],
    surface_normal=np.array([1, 0], dtype=int),
    **MODELS["Model/equalMass"].__dict__)


#############################
#           Tests           #
#############################
@pytest.mark.parametrize("key", RULES.keys())
def test_reflected_indices_inverse(key):
    rule = RULES[key]
    if not isinstance(rule, bp.BoundaryPointRule):
        return
    refl = rule.refl_idx_inverse
    assert np.all(refl[refl] == np.arange(refl.size))
    for (idx_v, v) in enumerate(rule.i_vels):
        v_refl = rule.i_vels[refl[idx_v]]
        assert np.all(v == -v_refl)


@pytest.mark.parametrize("key", RULES)
def test_reflection_keeps_total_mass(key):
    rule = RULES[key]
    # only check BoundaryPoints
    if not isinstance(rule, bp.BoundaryPointRule):
        return
    assert isinstance(rule, bp.BoundaryPointRule)
    for _ in range(100):
        inflow = np.zeros((1, rule.nvels))
        n_incoming_vels = rule.vels_in.size
        rand_vals = np.random.random(n_incoming_vels)
        inflow[..., rule.vels_in] = rand_vals
        reflected_inflow = rule.reflection(inflow)
        for s in rule.species:
            mass_in = rule.cmp_mass_density(inflow, s)
            mass_refl = rule.cmp_mass_density(reflected_inflow, s)
            absorption = rule.refl_absorbs[s]
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
