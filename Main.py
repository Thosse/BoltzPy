
# Desired Command / Road Map
import boltzpy as bp
import numpy as np

exisiting_simulation_file = None
sim = bp.Simulation(exisiting_simulation_file)
if exisiting_simulation_file is None:
    sim.add_specimen(mass=2, collision_rate=[50])
    sim.add_specimen(mass=3, collision_rate=[50, 50])
    sim.setup_time_grid(max_time=1,
                        number_time_steps=101,
                        calculations_per_time_step=10)
    sim.setup_position_grid(grid_dimension=1,
                            grid_shape=[21],
                            grid_spacing=0.5)
    sim.set_velocity_grids(grid_dimension=2,
                           min_points_per_axis=4,
                           max_velocity=1.5)
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
    sim.scheme.OperatorSplitting = "FirstOrder"
    sim.scheme.Transport = "FiniteDifferences_FirstOrder"
    sim.scheme.Transport_VelocityOffset = np.array([-0.2, 0.0])
    sim.scheme.Collisions_Generation = "UniformComplete"
    sim.scheme.Collisions_Computation = "EulerScheme"
print(sim.__str__(write_physical_grids=True))
sim.save()

sim.run_computation()

sim.create_animation()
# sim.create_animation(np.array([['Mass']]),
#                      sim.s.specimen_array[0:1])
