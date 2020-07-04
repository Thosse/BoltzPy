
# Desired Command / Road Map
import boltzpy as bp
import numpy as np

exisiting_simulation_file = None
if exisiting_simulation_file is not None:
    sim = bp.Simulation.load(exisiting_simulation_file)
else:
    sim = bp.Simulation(exisiting_simulation_file)
    sim.setup_time_grid(max_time=1,
                        number_time_steps=201,
                        calculations_per_time_step=5)
    sim.sv = bp.SVGrid([2, 3],
                       [(5, 5), (7, 7)],
                       0.25,
                       [6, 4],
                       np.array([[50, 50], [50, 50]]))
    sim.geometry = bp.Geometry(
        (31, ),
        0.5,
        [bp.ConstantPointRule(
            initial_rho=[1.0, 1.0],
            initial_drift=[[0.0, 0.0], [0.0, 0.0]],
            initial_temp=[.50, .50],
            affected_points=[0],
            velocity_grids=sim.sv),
         bp.InnerPointRule(
            initial_rho=[1.0, 1.0],
            initial_drift=[[0.0, 0.0], [0.0, 0.0]],
            initial_temp=[.50, .50],
            affected_points=np.arange(1, 30),
            velocity_grids=sim.sv),
         bp.BoundaryPointRule(
            initial_rho=[1.0, 1.0],
            initial_drift=[[0.0, 0.0], [0.0, 0.0]],
            initial_temp=[.5, .5],
            affected_points=[30],
            reflection_rate_inverse=[.3, .3],
            reflection_rate_elastic=[.3, .3],
            reflection_rate_thermal=[0.3, .3],
            absorption_rate=[0.1, .1],
            surface_normal=np.array([1, 0], dtype=int),
            velocity_grids=sim.sv)
         ]
    )
    sim.scheme.OperatorSplitting = "FirstOrder"
    sim.scheme.Transport = "FiniteDifferences_FirstOrder"
    sim.scheme.Transport_VelocityOffset = np.array([0.0, 0.0])
    # sim.scheme.Collisions_Generation = "UniformComplete"
    sim.scheme.Collisions_Generation = "Convergent"
    sim.scheme.Collisions_Computation = "EulerScheme"
    # print(sim.__str__(write_physical_grids=True))
    sim.coll.setup(sim.scheme, sim.sv)
    sim.save()
    # #
    # grp = sim.coll.group(sim.sv, mode="species")
    # for (key, colls) in grp.items():
    #     if key[0] == key[2]:
    #         continue
    #     print(key)
    #     print(len(colls))
    #     # import time
    #     # time.sleep(1)
    #     colls = np.array([c[0:4] for c in colls])
    #     bp.collisions.plot(sim.sv, colls, iterative=True)
    # sim.save()
    sim.compute()

sim.animate()
# sim.create_animation(np.array([['Mass']]),
#                      sim.s.specimen_array[0:1])
