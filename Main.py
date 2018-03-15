# Desired Command / Road Map
import boltzmann as b_pde
import boltzmann.calculation.calculation as b_clc
import boltzmann.animation.animation as b_ani

B = b_pde.BoltzmannPDE()
B.cnf.add_specimen(2)               # 2
B.cnf.add_specimen(3)               # 3
B.cnf.s._alpha *= 2                 # 2
B.cnf.configure_time(2, 11, 10)      # 20 1001 10
B.cnf.configure_position_space(1,
                               [20],       # 200
                               step_size=0.1)   # 0.1
B.cnf.configure_velocity_space(dimension=2,
                               grid_points_x_axis=4,    # 8
                               max_v=1.5)       # 1.5
B.cnf.collision_steps_per_time_step = 0     # 50
B.cnf.print()
B.cnf.setup()

B.begin_initialization()
B.ini.add_rule('Inner Point',
               [2, 1, 1],
               [[0, 0], [0, 0], [0, 0]],
               [1, 1, 1],
               name='foo')
B.ini.apply_rule(0, [0], [10])
B.ini.add_rule('Inner Point',
               [4, 2, 1],
               [[0, 0], [0, 0], [0, 0]],
               [1, 1, 1],
               name='bar')
B.ini.apply_rule(1, [10], [20])
B.ini.print(True)

clc = b_clc.Calculation(B.cnf, B.ini)
clc.run()

ani = b_ani.Animation(B.cnf, False)
ani.run()
