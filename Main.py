# Desired Command / Road Map
import boltzmann.configuration as b_cnf
import boltzmann.initialization as b_ini
import boltzmann.calculation as b_cal
import boltzmann.animation as b_ani
from datetime import datetime


print('Starting Time:\n' + str(datetime.now()) + '\n')
cnf = b_cnf.Configuration("default")
cnf.add_specimen(mass=2)               # 2
cnf.add_specimen(mass=3)               # 3
cnf.s.collision_rate_matrix *= 50                 # 2
cnf.set_time_grid(0.1, 11, 10)      # 20 1001 10
cnf.set_position_grid(1,
                      [21],  # 200
                      grid_spacing=0.1)   # 0.1
cnf.set_velocity_grids(grid_dimension=2,
                       min_points_per_axis=4,
                       max_velocity=1.5)
cnf.collision_steps_per_time_step = 1     # 50
print(cnf.__str__(write_physical_grids=True))
cnf.save()

ini = b_ini.Initialization(cnf)
ini.add_rule('Inner Point',
             [2, 1, 1],
             [[0, 0], [0, 0], [0, 0]],
             [1, 1, 1],
             name='foo')
ini.apply_rule(0, [0], [10])
ini.add_rule('Inner Point',
             [4, 2, 1],
             [[0, 0], [0, 0], [0, 0]],
             [1, 1, 1],
             name='bar')
ini.apply_rule(1, [10], [21])
ini.print(True)

cal = b_cal.Calculation(cnf, ini)
cal.run()

ani = b_ani.Animation(cnf)

ani.run()
