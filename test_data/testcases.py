
import sys
import os
mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mod_path)

import boltzpy as bp
import boltzpy.constants as bp_c
import numpy as np
import h5py

def get_test_parameters(file_name,
                        s=None,
                        t=None,
                        p=None,
                        sv=None,
                        coll=None,
                        geometry=None,
                        scheme=None,
                        output_parameters=None):
    parameters = dict()
    if os.path.exists(bp_c.TEST_DIRECTORY + file_name + ".hdf5"):
        file_name += "_new"
        # remove existing "filename_new.hdf5" files
        if os.path.exists(bp_c.TEST_DIRECTORY + file_name + ".hdf5"):
            os.remove(bp_c.TEST_DIRECTORY + file_name + ".hdf5")
    parameters["file_name"] = bp_c.TEST_DIRECTORY + file_name + ".hdf5"

    if s is None:
        s = bp.Species()
        s.add(mass=2,
              collision_rate=np.array([50], dtype=float))
        s.add(mass=3,
              collision_rate=np.array([50, 50], dtype=float))
    parameters["s"] = s

    if t is None:
        t = bp.Grid(ndim=1,
                    shape=(5,),
                    form='rectangular',
                    physical_spacing=0.01,
                    spacing=3)
    parameters["t"] = t

    if p is None:
        p = bp.Grid(ndim=1,
                    shape=(6,),  # todo make this a tuple
                    spacing=1,
                    form='rectangular',
                    physical_spacing=0.5)
    parameters["p"] = p

    if sv is None:
        sv = bp.SVGrid(ndim=2,
                       maximum_velocity=1.5,
                       shapes=[(5, 5), (7, 7)],
                       spacings=[3, 2],
                       forms=["rectangular", "rectangular"],
                       )
    parameters["sv"] = sv

    if geometry is None:
        rules = [bp.Rule(behaviour_type="Inner Point",
                         initial_rho=np.array([2.0, 1.0]),
                         initial_drift=np.array([[0.0, 0.0],
                                                 [0.0, 0.0]]),
                         initial_temp=np.array([1.0, 1.0]),
                         affected_points=[0, 1, 2],
                         name='High Pressure'),
                 bp.Rule(behaviour_type="Inner Point",
                         initial_rho=np.array([1.0, 1.0]),
                         initial_drift=np.array([[0.0, 0.0],
                                                 [0.0, 0.0]]),
                         initial_temp=np.array([1.0, 1.0]),
                         affected_points=[3, 4, 5],
                         name='Low Pressure')
                 ]
        geometry = bp.Geometry(ndim=p.ndim,
                               shape=p.shape,
                               rules=rules
                               )
    parameters["geometry"] = geometry

    if scheme is None:
        scheme = bp.Scheme(OperatorSplitting="FirstOrder",
                           Transport="FiniteDifferences_FirstOrder",
                           Transport_VelocityOffset=np.array([-0.2, 0.0]),
                           Collisions_Generation="UniformComplete",
                           Collisions_Computation="EulerScheme")
    parameters["scheme"] = scheme

    if output_parameters is None:
        output_parameters = np.array([['Mass',
                                       'Momentum_X'],
                                      ['Momentum_X',
                                       'Momentum_Flow_X'],
                                      ['Energy',
                                       'Energy_Flow_X']])
    parameters["output_parameters"] = output_parameters

    if coll is None:
        coll = bp.Collisions()
        coll.setup(scheme, sv, s)
    parameters["coll"] = coll

    return parameters


def create_testcase(file_name,
                    s=None,
                    t=None,
                    p=None,
                    sv=None,
                    coll=None,
                    geometry=None,
                    scheme=None,
                    output_parameters=None):
    params = get_test_parameters(file_name=file_name,
                                 s=s,
                                 t=t,
                                 p=p,
                                 sv=sv,
                                 coll=coll,
                                 geometry=geometry,
                                 scheme=scheme,
                                 output_parameters=output_parameters)
    sim = bp.Simulation(params["file_name"])
    sim.s = params["s"]
    sim.t = params["t"]
    sim.p = params["p"]
    sim.sv = params["sv"]
    sim.geometry = params["geometry"]
    sim.rule_arr = sim.geometry.rules
    sim.init_arr = sim.geometry.init_array
    sim.scheme = params["scheme"]
    sim.output_parameters = params["output_parameters"]
    sim.coll = params["coll"]
    print(sim.__str__(write_physical_grids=True))
    sim.save()
    sim.run_computation()
    return sim


def update_testcase(file_name,
                    s=None,
                    t=None,
                    p=None,
                    sv=None,
                    coll=None,
                    geometry=None,
                    scheme=None,
                    output_parameters=None):
    loc_old_results = bp_c.TEST_DIRECTORY + file_name + ".hdf5"
    assert os.path.exists(loc_old_results)
    sim_new = create_testcase(file_name=file_name,
                              s=s,
                              t=t,
                              p=p,
                              sv=sv,
                              coll=coll,
                              geometry=geometry,
                              scheme=scheme,
                              output_parameters=output_parameters)
    loc_new_results = sim_new.file_address + ".hdf5"
    # Open old and new file, to compare results
    old_file = h5py.File(loc_old_results, mode='r')
    new_file = h5py.File(loc_new_results, mode='r')
    # compare results
    try:
        for output in sim_new.output_parameters.flatten():
            results_old = old_file["Computation"][output][()]
            results_new = new_file["Computation"][output][()]
            assert results_old.shape == results_new.shape
            assert np.array_equal(results_old, results_new)
        os.remove(loc_old_results)
        sim_new.save(loc_old_results)
        sim_new.run_computation()
        print("Successfully updated: ", file_name)
    except AssertionError:
        print("Update failed: ", file_name)
    finally:
        os.remove(loc_new_results)
    return
