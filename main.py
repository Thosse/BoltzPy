
# Desired Command / Road Map
import boltzpy as bp
import numpy as np

exisiting_simulation_file = None
if exisiting_simulation_file is not None:
    import h5py
    sim = bp.Simulation.load(h5py.File(exisiting_simulation_file, mode='r'))
else:
    timing = bp.Grid((201,), 1/1000, 5)
    model = bp.Model([2, 3],
                     [(5, 5), (7, 7)],
                     0.25,
                     [6, 4],
                     np.array([[50, 50], [50, 50]]))
    geometry = bp.Geometry(
        (31, ),
        0.5,
        [bp.ConstantPointRule(
            particle_number=[1.0, 1.0],
            mean_velocity=[[0.0, 0.0], [0.0, 0.0]],
            temperature=[.50, .50],
            affected_points=[0],
            model=model),
         bp.InnerPointRule(
            particle_number=[1.0, 1.0],
            mean_velocity=[[0.0, 0.0], [0.0, 0.0]],
            temperature=[.50, .50],
            affected_points=np.arange(1, 30),
            model=model),
         bp.BoundaryPointRule(
            particle_number=[1.0, 1.0],
            mean_velocity=[[0.0, 0.0], [0.0, 0.0]],
            temperature=[.5, .5],
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
    coll.setup(model)
    sim = bp.Simulation(timing, geometry, model, coll, exisiting_simulation_file)
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
