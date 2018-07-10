# Desired Command / Road Map
import boltzmann.configuration as b_cnf
import boltzmann.initialization as b_ini
import boltzmann.calculation as b_cal
import boltzmann.animation as b_ani
from datetime import datetime
import numpy as np


print('Starting Time:\n' + str(datetime.now()) + '\n')
cnf = b_cnf.Configuration()
if cnf.s.n == 0:
    cnf.add_specimen(mass=2, collision_rate=[50])
    cnf.add_specimen(mass=3, collision_rate=[50, 50])
    cnf.coll_substeps = 5
    cnf.set_time_grid(max_time=1,
                      number_time_steps=101,
                      calculations_per_time_step=10)      # 20 1001 10
    cnf.set_position_grid(grid_dimension=1,
                          grid_shape=[21],  # 200
                          grid_spacing=0.5)   # 0.1
    cnf.set_velocity_grids(grid_dimension=2,
                           min_points_per_axis=4,
                           max_velocity=1.5,
                           velocity_offset=[-0.2, 0])
print(cnf.__str__(write_physical_grids=True))
cnf.save()

ini = b_ini.Initialization(cnf)
ini.add_rule('Inner Point',
             [2.0, 1.0],
             [[0, 0], [0, 0]],
             [1, 1],
             name='High Pressure')
ini.apply_rule(np.arange(0, 10), 0)
ini.add_rule('Inner Point',
             [1, 1],
             [[0, 0], [0, 0]],
             [1, 1],
             name='Low Pressure')
ini.apply_rule(np.arange(10, 21), 1)
ini.print(True)

cal = b_cal.Calculation(cnf, ini)
cal.run()

ani = b_ani.Animation(cnf)

ani.run()
