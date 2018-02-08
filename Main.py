# Desired Command / Road Map
import boltzmann as bP

B = bP.BoltzmannPDE()
B.add_specimen()
B.add_specimen(mass=2, name='Bernd')

B.setup_time(5, 1)
B.config.setup_pSpace(3, [[0, 5], [0.0, 0.1], [0.0, 9.9]], step_size=0.1)
B.config.setup_vSpace(2, 2.0, 0.1)
B.print()
B.run_configuration()

# B.init.
#     # Make PSV Grid
#     # Initialize PSV Grid based on RHO, DRIFT AND TEMP
#
# B.initialize.print()

#
# B.setup()
# B.setup.print() - prints grids as plot
#
# B.calc(animated_Moments)
#
# B.animate()
#
# B.save_animation()
