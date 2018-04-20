# Desired Command / Road Map
import boltzmann.configuration as b_cnf
import boltzmann.initialization as b_ini
import boltzmann.calculation as b_cal
import boltzmann.animation as b_ani
from datetime import datetime


print('Starting Time:\n' + str(datetime.now()))
cnf = b_cnf.Configuration()
cnf.add_specimen(mass=1)               # 2
cnf.add_specimen(mass=2)               # 3
cnf.s.collision_rate_matrix *= 50                 # 2
cnf.configure_time(0.01, 10, 1)      # 20 1001 10
cnf.configure_position_space(1,
                             [20],       # 200
                             step_size=0.1)   # 0.1
cnf.configure_velocity_space(dimension=2,
                             grid_points_x_axis=4,    # 8
                             max_v=1.5)       # 1.5
cnf.collision_steps_per_time_step = 1     # 50
cnf.print()

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
ini.apply_rule(1, [10], [20])
ini.print()

cal = b_cal.Calculation(cnf, ini)
cal.run()

ani = b_ani.Animation(cnf)

ani.run()