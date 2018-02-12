# Desired Command / Road Map
import boltzmann as bP

B = bP.BoltzmannPDE()
B.add_specimen()
B.add_specimen(2)
B.setup_time(1, 5)
B.setup_position_space(2,
                       [2, 10],
                       step_size=0.1)
B.setup_velocity_space(dimension=2,
                       grid_points_x_axis=3,
                       max_v=1.0)
B.print()
# B.run_configuration()

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
