import sys
import os
mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mod_path)

import boltzpy as b_sim
from boltzpy.constants import TEST_DIRECTORY as PATH
import numpy as np

# this is a dict of functions
# each function creates a corresponding test_case
generate_test_case = dict()

test_cases = ["shock_1spc",
              "shock_2spc",
              "shock_2spc_complete"
              ]


# Single Species simulation
def shock_1spc(file_name="shock_1spc", compute=True, animate=False):
    # remove old file, if necessary
    if os.path.exists(PATH + file_name + ".hdf5"):
        os.remove(PATH + file_name + ".hdf5")
    sim = b_sim.Simulation(PATH + file_name)
    sim.add_specimen(mass=2, collision_rate=[50])
    sim.setup_time_grid(max_time=0.1,
                        number_time_steps=11,
                        calculations_per_time_step=10)
    sim.setup_position_grid(grid_dimension=1,
                            grid_shape=[21],
                            grid_spacing=0.5)
    sim.set_velocity_grids(grid_dimension=2,
                           min_points_per_axis=4,
                           max_velocity=1.5,
                           velocity_offset=[-0.2, 0])
    sim.add_rule('Inner Point',
                 np.array([1.0]),
                 np.array([[0.0, 0.0]]),
                 np.array([1.0]),
                 name='High Pressure')
    sim.choose_rule(np.arange(0, 10), 0)
    sim.add_rule('Inner Point',
                 np.array([1.0]),
                 np.array([[0.0, 0.0]]),
                 np.array([1.0]),
                 name='Low Pressure')
    sim.choose_rule(np.arange(10, 21), 1)
    sim.scheme["Approach"] = "DiscreteVelocityModels"
    sim.scheme["OperatorSplitting_Order"] = 1
    sim.scheme["Transport_Scheme"] = "FiniteDifferences"
    sim.scheme["Transport_Order"] = 1
    sim.scheme["Collisions_RelationsScheme"] = "UniformComplete"
    sim.scheme["Collisions_ComputationScheme"] = "EulerScheme"
    sim.save()
    if compute:
        sim.run_computation()
    if animate:
        sim.create_animation()
    return
generate_test_case["shock_1spc"] = shock_1spc


def shock_2spc(file_name="shock_2spc", compute=True, animate=False):
    # remove old file, if necessary
    if os.path.exists(PATH + file_name + ".hdf5"):
        os.remove(PATH + file_name + ".hdf5")
    sim = b_sim.Simulation(PATH + file_name)
    sim.add_specimen(mass=2, collision_rate=[50])
    sim.add_specimen(mass=3, collision_rate=[50, 50])
    sim.setup_time_grid(max_time=0.1,
                        number_time_steps=11,
                        calculations_per_time_step=50)
    sim.setup_position_grid(grid_dimension=1,
                            grid_shape=[21],
                            grid_spacing=0.5)
    sim.set_velocity_grids(grid_dimension=2,
                           min_points_per_axis=4,
                           max_velocity=1.5,
                           velocity_offset=[-0.2, 0])
    sim.add_rule('Inner Point',
                 np.array([2.0, 1.0]),
                 np.array([[0.0, 0.0], [0.0, 0.0]]),
                 np.array([1.0, 1.0]),
                 name='High Pressure')
    sim.choose_rule(np.arange(0, 10), 0)
    sim.add_rule('Inner Point',
                 np.array([1.0, 1.0]),
                 np.array([[0.0, 0.0], [0.0, 0.0]]),
                 np.array([1.0, 1.0]),
                 name='Low Pressure')
    sim.choose_rule(np.arange(10, 21), 1)
    sim.scheme["Approach"] = "DiscreteVelocityModels"
    sim.scheme["OperatorSplitting_Order"] = 1
    sim.scheme["Transport_Scheme"] = "FiniteDifferences"
    sim.scheme["Transport_Order"] = 1
    sim.scheme["Collisions_RelationsScheme"] = "UniformComplete"
    sim.scheme["Collisions_ComputationScheme"] = "EulerScheme"
    sim.save()
    if compute:
        sim.run_computation()
    if animate:
        sim.create_animation()
    return
generate_test_case["shock_2spc"] = shock_2spc


def shock_2spc_complete(file_name="shock_2spc_complete",
                        compute=True,
                        animate=False):
    # remove old file, if necessary
    if os.path.exists(PATH + file_name + ".hdf5"):
        os.remove(PATH + file_name + ".hdf5")
    sim = b_sim.Simulation(PATH + file_name)
    sim.add_specimen(mass=2, collision_rate=[50])
    sim.add_specimen(mass=3, collision_rate=[50, 50])
    sim.output_parameters = np.array([["Complete_Distribution"]])
    sim.setup_time_grid(max_time=0.1,
                        number_time_steps=11,
                        calculations_per_time_step=50)
    sim.setup_position_grid(grid_dimension=1,
                            grid_shape=[21],
                            grid_spacing=0.5)
    sim.set_velocity_grids(grid_dimension=2,
                           min_points_per_axis=4,
                           max_velocity=1.5,
                           velocity_offset=[-0.2, 0])
    sim.add_rule('Inner Point',
                 np.array([2.0, 1.0]),
                 np.array([[0.0, 0.0], [0.0, 0.0]]),
                 np.array([1.0, 1.0]),
                 name='High Pressure')
    sim.choose_rule(np.arange(0, 10), 0)
    sim.add_rule('Inner Point',
                 np.array([1.0, 1.0]),
                 np.array([[0.0, 0.0], [0.0, 0.0]]),
                 np.array([1.0, 1.0]),
                 name='Low Pressure')
    sim.choose_rule(np.arange(10, 21), 1)
    sim.scheme["Approach"] = "DiscreteVelocityModels"
    sim.scheme["OperatorSplitting_Order"] = 1
    sim.scheme["Transport_Scheme"] = "FiniteDifferences"
    sim.scheme["Transport_Order"] = 1
    sim.scheme["Collisions_RelationsScheme"] = "UniformComplete"
    sim.scheme["Collisions_ComputationScheme"] = "EulerScheme"
    sim.save()
    if compute:
        sim.run_computation()
    if animate:
        sim.create_animation()
    return
generate_test_case["shock_2spc_complete"] = shock_2spc_complete


def shock_2spc_equalmass(file_name="shock_2spc_equalmass",
                         compute=True,
                         animate=False):
    # remove old file, if necessary
    if os.path.exists(PATH + file_name + ".hdf5"):
        os.remove(PATH + file_name + ".hdf5")
    sim = b_sim.Simulation(PATH + file_name)
    sim.add_specimen(mass=2, collision_rate=[50])
    sim.add_specimen(mass=2, collision_rate=[50, 5])
    sim.setup_time_grid(max_time=0.1,
                        number_time_steps=11,
                        calculations_per_time_step=10)
    sim.setup_position_grid(grid_dimension=1,
                            grid_shape=[21],
                            grid_spacing=0.5)
    sim.set_velocity_grids(grid_dimension=2,
                           min_points_per_axis=4,
                           max_velocity=1.5,
                           velocity_offset=[-0.2, 0])
    sim.add_rule('Inner Point',
                 np.array([2.0, 1.0]),
                 np.array([[0.0, 0.0], [0.0, 0.0]]),
                 np.array([1.0, 1.0]),
                 name='High Pressure')
    sim.choose_rule(np.arange(0, 10), 0)
    sim.add_rule('Inner Point',
                 np.array([1.0, 1.0]),
                 np.array([[0.0, 0.0], [0.0, 0.0]]),
                 np.array([1.0, 1.0]),
                 name='Low Pressure')
    sim.choose_rule(np.arange(10, 21), 1)
    sim.scheme["Approach"] = "DiscreteVelocityModels"
    sim.scheme["OperatorSplitting_Order"] = 1
    sim.scheme["Transport_Scheme"] = "FiniteDifferences"
    sim.scheme["Transport_Order"] = 1
    sim.scheme["Collisions_RelationsScheme"] = "UniformComplete"
    sim.scheme["Collisions_ComputationScheme"] = "EulerScheme"
    sim.save()
    if compute:
        sim.run_computation()
    if animate:
        sim.create_animation()
    return
generate_test_case["shock_2spc_equalmass"] = shock_2spc_equalmass


if __name__ == "__main__":
    for key in generate_test_case.keys():
        print()
        case_str = "Generating {key}.hdf5".format(key=key)
        print(case_str)
        print(len(case_str)*'-')
        generate_test_case[key](key)
