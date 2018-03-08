# Desired Command / Road Map
import boltzmann as b_pde
import boltzmann.calculation.output_function as b_opf
import boltzmann.calculation.calculation as b_clc

B = b_pde.BoltzmannPDE()
B.cnf.add_specimen(2)
B.cnf.add_specimen(3)
B.cnf.configure_time(5, 10, 10)

B.cnf.configure_position_space(1,
                               [10],
                               step_size=1)
B.cnf.configure_velocity_space(dimension=2,
                               grid_points_x_axis=5,
                               max_v=2.0)
B.cnf.print(True)
B.cnf.setup_collisions()
B.begin_initialization()
B.ini.add_rule('Inner_Point',
               [1, 2],
               [[0, 0], [0, 0]],
               [1, 1],
               name='foo')
B.ini.apply_rule(0, [0], [5])
B.ini.add_rule('Inner_Point',
               [2, 4],
               [[0, 0], [0, 0]],
               [1, 1],
               name='bar')
B.ini.apply_rule(1, [5], [10])
B.ini.print(True)
moments = ['Mass']
clc = b_clc.Calculation(B.cnf, B.ini, moments)
clc.run()
print(clc.write_mass[-1, ...])

# B.animate()
#
# B.save_animation()
