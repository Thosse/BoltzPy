# Desired Command / Road Map
import boltzmann as b_pde

B = b_pde.BoltzmannPDE()
B.config.add_specimen(2)
B.config.add_specimen(3)
B.config.configure_time(1.0, 5)
B.config.configure_position_space(2,
                                  [2, 11],
                                  step_size=0.1)
B.config.configure_velocity_space(dimension=2,
                                  grid_points_x_axis=5,
                                  max_v=6.0)
B.config.print(True)
B.config.configure_collisions('complete')
B.begin_initialization()
B.init.add_rule('Inner_Point',
                [1, 2],
                [[0, 0], [0, 0]],
                [1, 1],
                name='foo')
B.init.apply_rule(0, [0, 0], [1, 10])
B.init.add_rule('Inner_Point',
                [2, 4],
                [[0, 0], [0, 0]],
                [1, 1],
                name='bar')
B.init.apply_rule(1, [0, 0], [1, 1])
B.init.print(True)
print(B.init.create_psv_grid().sum(axis=1).reshape(tuple(B.config.p.n)))

# B.calc.setup(animated_Moments)
# B.calc.run()
#
# B.animate()
#
# B.save_animation()
