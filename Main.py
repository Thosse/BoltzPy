
# Desired Command / Road Map
import boltzpy as b_sim
import numpy as np

sim = b_sim.Simulation()
new_simulation = True
if new_simulation:
    sim.add_specimen(mass=2, collision_rate=[50])
    sim.add_specimen(mass=3, collision_rate=[50, 50])
    sim.coll_substeps = 5
    sim.setup_time_grid(max_time=1,
                        number_time_steps=101,
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
print(sim.__str__(write_physical_grids=True))
sim.save()

sim.run_computation()

sim.create_animation()
# sim.create_animation(np.array([['Mass']]),
#                      sim.s.specimen_array[0:1])
