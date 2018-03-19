# Desired Command / Road Map
import boltzmann as b_pde
import boltzmann.calculation.calculation as b_cal
import boltzmann.animation.animation as b_ani

B = b_pde.BoltzmannPDE()
B.cnf.add_specimen(2)               # 2
B.cnf.add_specimen(3)               # 3
B.cnf.s._alpha *= 2                 # 2
B.cnf.configure_time(1, 501, 1)      # 20 1001 10
B.cnf.configure_position_space(1,
                               [200],       # 200
                               step_size=0.1)   # 0.1
B.cnf.configure_velocity_space(dimension=2,
                               grid_points_x_axis=8,    # 8
                               max_v=1.5)       # 1.5
B.cnf.collision_steps_per_time_step = 2000     # 50
B.cnf.print()

B.begin_initialization()
B.ini.add_rule('Inner Point',
               [2, 1, 1],
               [[0, 0], [0, 0], [0, 0]],
               [1, 1, 1],
               name='foo')
B.ini.apply_rule(0, [0], [100])
B.ini.add_rule('Inner Point',
               [4, 2, 1],
               [[0, 0], [0, 0], [0, 0]],
               [1, 1, 1],
               name='bar')
B.ini.apply_rule(1, [100], [200])
B.ini.print()

cal = b_cal.Calculation(B.cnf, B.ini)
cal.run()

ani = b_ani.Animation(B.cnf)
ani.run()
