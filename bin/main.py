
# Desired Command / Road Map
import boltzpy as bp
import numpy as np

exisiting_simulation_file = None
if exisiting_simulation_file is not None:
    import h5py
    sim = bp.Simulation.load(h5py.File(exisiting_simulation_file, mode='r'))
else:
    timing = bp.Grid((201,), 1/1000, 5)
    model = bp.CollisionModel([2, 3],
                              [(5, 5), (7, 7)],
                              0.25,
                              [6, 4],
                              np.array([[50, 50], [50, 50]]))

    geometry = bp.Geometry(
        (31, ),
        0.5,
        [bp.ConstantPointRule(
            number_densities=[1.0, 1.0],
            mean_velocities=[[0.0, 0.0], [0.0, 0.0]],
            temperatures=[.50, .50],
            affected_points=[0],
            masses=[2, 3],
            shapes=[(5, 5), (7, 7)],
            base_delta=0.25,
            spacings=[6, 4]),
         bp.InnerPointRule(
            number_densities=[1.0, 1.0],
            mean_velocities=[[0.0, 0.0], [0.0, 0.0]],
            temperatures=[.50, .50],
            affected_points=np.arange(1, 30),
            masses=[2, 3],
            shapes=[(5, 5), (7, 7)],
            base_delta=0.25,
            spacings=[6, 4]),
         bp.BoundaryPointRule(
            number_densities=[1.0, 1.0],
            mean_velocities=[[0.0, 0.0], [0.0, 0.0]],
            temperatures=[.5, .5],
            affected_points=[30],
            refl_inverse=[.3, .3],
            refl_elastic=[.3, .3],
            refl_thermal=[0.3, .3],
            refl_absorbs=[0.1, .1],
            surface_normal=np.array([1, 0], dtype=int),
            masses=[2, 3],
            shapes=[(5, 5), (7, 7)],
            base_delta=0.25,
            spacings=[6, 4]),
         ]
    )
    sim = bp.Simulation(timing, geometry, model, exisiting_simulation_file)
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
