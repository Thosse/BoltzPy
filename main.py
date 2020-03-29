
# Desired Command / Road Map
import boltzpy as bp
import numpy as np

exisiting_simulation_file = None
sim = bp.Simulation(exisiting_simulation_file)
if exisiting_simulation_file is None:
    sim.add_specimen(mass=2, collision_rate=[50])
    sim.add_specimen(mass=3, collision_rate=[50, 50])
    sim.setup_time_grid(max_time=1,
                        number_time_steps=201,
                        calculations_per_time_step=5)
    sim.setup_position_grid(grid_dimension=1,
                            grid_shape=(31, ),
                            grid_spacing=0.5)
    sim.set_velocity_grids(grid_dimension=2,
                           maximum_velocity=1.5,
                           shapes=[(7, 7),
                                   (9, 9)])     # shapes were before :(5, 5),(7, 7)
    sim.geometry = bp.Geometry(
        sim.p.shape,
        [bp.ConstantPointRule(
            initial_rho=[1.0, 1.0],
            initial_drift=[[0.0, 0.0], [0.0, 0.0]],
            initial_temp=[.50, .50],
            velocity_grids=sim.sv,
            affected_points=[0],
            species=sim.s),
         bp.InnerPointRule(
            initial_rho=[1.0, 1.0],
            initial_drift=[[0.0, 0.0], [0.0, 0.0]],
            initial_temp=[.50, .50],
            affected_points=np.arange(1, 30),
            velocity_grids=sim.sv,
            species=sim.s),
         bp.BoundaryPointRule(
            initial_rho=[1.0, 1.0],
            initial_drift=[[0.0, 0.0], [0.0, 0.0]],
            initial_temp=[.6, .6],
            affected_points=[30],
            reflection_rate_inverse=[.3, .3],
            reflection_rate_elastic=[.3, .3],
            reflection_rate_thermal=[0.3, .3],
            absorption_rate=[0.1, .1],
            surface_normal=np.array([1, 0], dtype=int),
            velocity_grids=sim.sv,
            species=sim.s)
         ]
    )
    # sim.geometry.rules[0].plot(sim.sv, sim.s, 0)
    sim.scheme.OperatorSplitting = "FirstOrder"
    sim.scheme.Transport = "FiniteDifferences_FirstOrder"
    sim.scheme.Transport_VelocityOffset = np.array([0.0, 0.0])
    sim.scheme.Collisions_Generation = "Simple"     # "UniformComplete"
    sim.scheme.Collisions_Computation = "EulerScheme"
    # print(sim.__str__(write_physical_grids=True))
    sim.coll.setup(sim.scheme, sim.sv, sim.s)
    bp.collisions.plot(sim.sv, sim.coll.relations)
    sim.save()
    sim.compute()

sim.animate()
# sim.create_animation(np.array([['Mass']]),
#                      sim.s.specimen_array[0:1])
