# Desired Command / Road Map
import boltzmann as b_sim
from datetime import datetime
import numpy as np

print('Starting Time:\n' + str(datetime.now()) + '\n')
sim = b_sim.Simulation()
if sim.configuration.s.n == 0:
    sim.configuration.add_specimen(mass=2, collision_rate=[50])
    sim.configuration.add_specimen(mass=3, collision_rate=[50, 50])
    sim.configuration.coll_substeps = 5
    sim.configuration.set_time_grid(max_time=1,
                                    number_time_steps=101,
                                    calculations_per_time_step=10)
    sim.configuration.set_position_grid(grid_dimension=1,
                                        grid_shape=[21],  # 200
                                        grid_spacing=0.5)  # 0.1
    sim.configuration.set_velocity_grids(grid_dimension=2,
                                         min_points_per_axis=4,
                                         max_velocity=1.5,
                                         velocity_offset=[-0.2, 0])
print(sim.configuration.__str__(write_physical_grids=True))
sim.configuration.save()

# ini.load(cnf.file_address)
sim.initialization.add_rule('Inner Point',
                            np.array([2.0, 1.0]),
                            np.array([[0.0, 0.0], [0.0, 0.0]]),
                            np.array([1.0, 1.0]),
                            name='High Pressure')
sim.initialization.apply_rule(np.arange(0, 10), 0)
sim.initialization.add_rule('Inner Point',
                            np.array([1.0, 1.0]),
                            np.array([[0.0, 0.0], [0.0, 0.0]]),
                            np.array([1.0, 1.0]),
                            name='Low Pressure')
sim.initialization.apply_rule(np.arange(10, 21), 1)
sim.initialization.print()
sim.initialization.save()

sim.calculation.run()

sim.animation.animate(np.array([['Mass']]),
                      sim.configuration.s.specimen_array[0:1])
