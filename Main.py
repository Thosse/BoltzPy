# Desired Command / Road Map
import boltzmann as bP

x = bP.BoltzmannPDE()
x.config.add_specimen()
x.config.add_specimen(mass=2, name='Bernd')

x.config.set_t(5, 1)
x.config.set_p(3, [[0, 5], [0.0, 0.1], [0.0, 9.9]], step_size=0.1)
x.config.set_v(2, 2.0, 0.1)
x.print()
# x.config.setup()

# x.init.
#     # Make PSV Grid
#     # Initialize PSV Grid based on RHO, DRIFT AND TEMP
#
# x.initialize.print()

#
# x.setup()
# x.setup.print() - prints grids as plot
#
# x.calc(animated_Moments)
#
# x.animate()
#
# x.save_animation()
