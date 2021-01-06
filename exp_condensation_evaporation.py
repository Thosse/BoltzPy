
# Desired Command / Road Map
import boltzpy as bp
import numpy as np

timing = bp.Grid((2001,), 1/1000, 500)
model = bp.CollisionModel([2, 3],
                          [(5, 5), (7, 7)],
                          0.25,
                          [6, 4],
                          np.array([[500, 500], [500, 500]]))
# Initial state
number_densities = [0.1, 0.9]
mean_velocities = [[0.0, 0.0], [0.0, 0.0]]
temperatures = [.5, .5]
geometry = bp.Geometry(
    (1001, ),
    0.25,
    [bp.ConstantPointRule(
        number_densities=number_densities,
        mean_velocities=mean_velocities,
        temperatures=temperatures,
        affected_points=[0],
        **model.__dict__),
     bp.InnerPointRule(
        number_densities=number_densities,
        mean_velocities=mean_velocities,
        temperatures=temperatures,
        affected_points=np.arange(1, 1000),
        **model.__dict__),
     bp.BoundaryPointRule(
        number_densities=number_densities,
        mean_velocities=mean_velocities,
        temperatures=temperatures,
        affected_points=[1000],
        refl_inverse=[0.9, 1],
        refl_elastic=[0, 0],
        refl_thermal=[0, 0],
        refl_absorbs=[0.1, 0],
        surface_normal=np.array([1, 0], dtype=int),
        **model.__dict__)
     ]
)
sim = bp.Simulation(timing, geometry, model)
sim.compute()

sim.animate()
