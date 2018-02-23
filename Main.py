# Desired Command / Road Map
import boltzmann as b_pde

B = b_pde.BoltzmannPDE()
B.add_specimen(2)
B.add_specimen(3)
B.configure_time(1.0, 5)
B.configure_position_space(2,
                           [2, 10],
                           step_size=0.1)
# Todo - still depends on max_v => rounding errors
B.configure_velocity_space(dimension=2,
                           grid_points_x_axis=5,
                           max_v=6.0)
B.configure_collisions('complete')
B.print_configuration(True)

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
