# Desired Command / Road Map
import boltzmann.configuration as b_cnf
import boltzmann.initialization as b_ini
import boltzmann.calculation as b_cal
import boltzmann.animation as b_ani
from datetime import datetime


print('Starting Time:\n' + str(datetime.now()))
cnf = b_cnf.Configuration()
cnf.add_specimen(1)               # 2
cnf.add_specimen(2)               # 3
cnf.s._alpha *= 50                 # 2
cnf.configure_time(10, 2510, 10)      # 20 1001 10
cnf.configure_position_space(1,
                             [500],       # 200
                             step_size=0.1)   # 0.1
cnf.configure_velocity_space(dimension=2,
                             grid_points_x_axis=40,    # 8
                             max_v=1.5)       # 1.5
cnf.collision_steps_per_time_step = 0     # 50
cnf.print()

ini = b_ini.Initialization(cnf)
ini.add_rule('Inner Point',
             [2, 1, 1],
             [[0, 0], [0, 0], [0, 0]],
             [1, 1, 1],
             name='foo')
ini.apply_rule(0, [0], [250])
ini.add_rule('Inner Point',
             [4, 2, 1],
             [[0, 0], [0, 0], [0, 0]],
             [1, 1, 1],
             name='bar')
ini.apply_rule(1, [250], [500])
ini.print()

cal = b_cal.Calculation(cnf, ini)
cal.run()

ani = b_ani.Animation(cnf)

ani.run()