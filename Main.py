# Desired Command / Road Map
import BoltzmannPDE as bB

x = bB.BoltzmannPDE()
x.initialize.add_specimen()
x.initialize.add_specimen(mass=2, name='Bernd')
x.initialize.time(5, 1)
x.initialize.position_space(3, [[0,5],[0.0, 0.1], [0.0, 9.9]], step_size=0.1)
x.initialize.velocity_space(2, 2.0, 0.1)
x.initialize.print()

#
# x.setup()
# x.setup.print() - prints grids as plot
#
# x.calc(animated_Moments)
#
# x.animate()
#
# x.save_animation()
