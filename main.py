
# Desired Command / Road Map
import boltzpy as bp
import numpy as np

exisiting_simulation_file = None
if exisiting_simulation_file is not None:
    sim = bp.Simulation.load(exisiting_simulation_file)
else:
    t = bp.Grid((201,), 1/1000, 5)
    model = bp.Model([2, 3],
                     [(5, 5), (7, 7)],
                     0.25,
                     [6, 4],
                     np.array([[50, 50], [50, 50]]))
    geometry = bp.Geometry(
        (31, ),
        0.5,
        [bp.ConstantPointRule(
            initial_rho=[1.0, 1.0],
            initial_drift=[[0.0, 0.0], [0.0, 0.0]],
            initial_temp=[.50, .50],
            affected_points=[0],
            model=model),
         bp.InnerPointRule(
            initial_rho=[1.0, 1.0],
            initial_drift=[[0.0, 0.0], [0.0, 0.0]],
            initial_temp=[.50, .50],
            affected_points=np.arange(1, 30),
            model=model),
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
            model=model)
         ]
    )
    coll = bp.Collisions()
    coll.setup(model=model)
    sim = bp.Simulation(t, geometry, model, coll, exisiting_simulation_file)
    sim.save()
    # #
    # grp = sim.coll.group(sim.model, mode="species")
    # for (key, colls) in grp.items():
    #     if key[0] == key[2]:
    #         continue
    #     print(key)
    #     print(len(colls))
    #     # import time
    #     # time.sleep(1)
    #     colls = np.array([c[0:4] for c in colls])
    #     bp.collisions.plot(sim.model, colls, iterative=True)
    # sim.save()
    sim.compute()

sim.animate()
# sim.create_animation(np.array([['Mass']]),
#                      sim.s.specimen_array[0:1])
