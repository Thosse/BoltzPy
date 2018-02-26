# Desired Command / Road Map
import boltzmann as b_pde

B = b_pde.BoltzmannPDE()
B.config.add_specimen(2)
B.config.add_specimen(3)
B.config.configure_time(1.0, 5)
B.config.configure_position_space(2,
                           [2, 10],
                           step_size=0.1)
# Todo - still depends on max_v => rounding errors
B.config.configure_velocity_space(dimension=2,
                           grid_points_x_axis=5,
                           max_v=6.0)
# B.configure_collisions('complete')
B.begin_initialization()
B.init.add_rule('Inner_Point',
                [1, 2],
                [[0,0], [0,0]],
                [1, 1],
                name='foo')
B.init.add_rule('Inner_Point',
                [10, 20],
                [[0,1], [0,0]],
                [1, 1],
                name='bar')
B.init.print(True)

# B.init.
#     # Make PSV Grid
#     # Initialize PSV Grid based on RHO, DRIFT AND TEMP
#
# B.initialize.print()

#
# B.calc.setup(animated_Moments)
# B.calc.run()
#
# B.animate()
#
# B.save_animation()
