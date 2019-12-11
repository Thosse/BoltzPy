
import boltzpy as bp
import boltzpy.constants as bp_c
import numpy as np
import h5py
import os


class TestCase(dict):
    def __init__(self,
                 file_name,
                 s=None,
                 t=None,
                 p=None,
                 sv=None,
                 coll=None,
                 geometry=None,
                 scheme=None,
                 output_parameters=None):
        super().__init__(self)

        self["file_name"] = file_name

        if s is None:
            s = bp.Species()
            s.add(mass=2,
                  collision_rate=np.array([50], dtype=float))
            s.add(mass=3,
                  collision_rate=np.array([50, 50], dtype=float))
        self["s"] = s

        if t is None:
            t = bp.Grid(ndim=1,
                        shape=(5,),
                        form='rectangular',
                        physical_spacing=0.01,
                        spacing=3)
        self["t"] = t

        if p is None:
            p = bp.Grid(ndim=1,
                        shape=(6,),
                        spacing=1,
                        form='rectangular',
                        physical_spacing=0.5)
        self["p"] = p

        if sv is None:
            sv = bp.SVGrid(ndim=2,
                           maximum_velocity=1.5,
                           shapes=[(5, 5), (7, 7)],
                           spacings=[3, 2],
                           forms=["rectangular", "rectangular"],
                           )
        self["sv"] = sv

        if geometry is None:
            rules = [bp.Rule(behaviour_type="Constant Point",
                             initial_rho=np.array([2.0, 1.0]),
                             initial_drift=np.array([[0.0, 0.0],
                                                     [0.0, 0.0]]),
                             initial_temp=np.array([1.0, 1.0]),
                             affected_points=[0]),
                     bp.Rule(behaviour_type="Inner Point",
                             initial_rho=np.array([2.0, 1.0]),
                             initial_drift=np.array([[0.0, 0.0],
                                                     [0.0, 0.0]]),
                             initial_temp=np.array([1.0, 1.0]),
                             affected_points=[1, 2]),
                     bp.Rule(behaviour_type="Inner Point",
                             initial_rho=np.array([1.0, 1.0]),
                             initial_drift=np.array([[0.0, 0.0],
                                                     [0.0, 0.0]]),
                             initial_temp=np.array([1.0, 1.0]),
                             affected_points=[3, 4]),
                     bp.Rule(behaviour_type="Constant Point",
                             initial_rho=np.array([1.0, 1.0]),
                             initial_drift=np.array([[0.0, 0.0],
                                                     [0.0, 0.0]]),
                             initial_temp=np.array([1.0, 1.0]),
                             affected_points=[5])
                     ]
            geometry = bp.Geometry(ndim=p.ndim,
                                   shape=p.shape,
                                   rules=rules
                                   )
        self["geometry"] = geometry

        if scheme is None:
            scheme = bp.Scheme(OperatorSplitting="FirstOrder",
                               Transport="FiniteDifferences_FirstOrder",
                               Transport_VelocityOffset=np.array([-0.2, 0.0]),
                               Collisions_Generation="UniformComplete",
                               Collisions_Computation="EulerScheme")
        self["scheme"] = scheme

        if output_parameters is None:
            output_parameters = np.array([['Mass',
                                           'Momentum_X'],
                                          ['Momentum_X',
                                           'Momentum_Flow_X'],
                                          ['Energy',
                                           'Energy_Flow_X']])
        self["output_parameters"] = output_parameters

        if coll is None:
            coll = bp.Collisions()
            # Todo move into save_results?
            coll.setup(scheme=scheme, svgrid=sv, species=s)
        self["coll"] = coll
        return

    @staticmethod
    def address(file_name):
        return bp_c.TEST_DIRECTORY + file_name + ".hdf5"

    def save_results(self, address=None):
        if address is None:
            address = self.address(self["file_name"])
        assert not os.path.exists(address), address
        sim = bp.Simulation(address)
        sim.s = self["s"]
        sim.t = self["t"]
        sim.p = self["p"]
        sim.sv = self["sv"]
        sim.geometry = self["geometry"]
        sim.scheme = self["scheme"]
        sim.output_parameters = self["output_parameters"]
        sim.coll = self["coll"]
        sim.save()
        sim.compute()
        return sim

    def compare_results(self):
        address_old = self.address(self["file_name"])
        assert os.path.exists(address_old)
        address_new = self.address("_tmp")
        assert address_old != address_new
        output = "No comparison done so far"
        try:
            self.save_results(address_new)
            # Open old and new file
            old_file = h5py.File(address_old, mode='r')
            new_file = h5py.File(address_new, mode='r')
            # compare results
            for output in self["output_parameters"].flatten():
                results_old = old_file["Computation"][output][()]
                results_new = new_file["Computation"][output][()]
                assert results_old.shape == results_new.shape
                assert np.array_equal(results_old, results_new)
        except AssertionError:
            print("Update failed: ", self["file_name"])
            print("\tDifferences found in: ", output)
            return False
        finally:
            os.remove(address_new)
        return True

    def update_results(self):
        if self.compare_results():
            os.remove(self.address(self["file_name"]))
            self.save_results()
            print("Successfully updated: ", self["file_name"])
        else:
            assert False


################################################################################
#                           Implemented TestCases                              #
################################################################################
CASES = list()

# Mono Species, shock
tc1_s = bp.Species()
tc1_s.add(mass=1, collision_rate=np.array([50], dtype=float))
tc1_sv = bp.SVGrid(ndim=2,
                   maximum_velocity=1.5,
                   shapes=[(5, 5)],
                   spacings=[2],
                   forms=["rectangular"],
                   )
tc1_rules = [bp.Rule(behaviour_type="Constant Point",
                     initial_rho=np.array([1.0]),
                     initial_drift=np.array([[0.0, 0.0]]),
                     initial_temp=np.array([1.0]),
                     affected_points=[0]),
             bp.Rule(behaviour_type="Inner Point",
                     initial_rho=np.array([1.0]),
                     initial_drift=np.array([[0.0, 0.0]]),
                     initial_temp=np.array([1.0]),
                     affected_points=[1, 2]),
             bp.Rule(behaviour_type="Inner Point",
                     initial_rho=np.array([1.0]),
                     initial_drift=np.array([[0.0, 0.0]]),
                     initial_temp=np.array([1.0]),
                     affected_points=[3, 4]),
             bp.Rule(behaviour_type="Constant Point",
                     initial_rho=np.array([1.0]),
                     initial_drift=np.array([[0.0, 0.0]]),
                     initial_temp=np.array([1.0]),
                     affected_points=[5])
             ]
tc1_geometry = bp.Geometry(ndim=1,
                           shape=(6,),
                           rules=tc1_rules)
CASES.append(TestCase("shock_monospecies",
                      s=tc1_s,
                      sv=tc1_sv,
                      geometry=tc1_geometry))

# Two Species, eqal mass, shock
tc2_s = bp.Species()
tc2_s.add(mass=3, collision_rate=np.array([50], dtype=float))
tc2_s.add(mass=3, collision_rate=np.array([50, 50], dtype=float))
tc2_sv = bp.SVGrid(ndim=2,
                   maximum_velocity=1.5,
                   shapes=[(5, 5), (5, 5)],
                   spacings=[6, 6],
                   forms=["rectangular", "rectangular"],
                   )
CASES.append(TestCase("shock_2Species_equalMass",
                      s=tc2_s,
                      sv=tc2_sv))

# Two Species, shock
CASES.append(TestCase("shock_2Species"))

# Two Species, shock, complete distribution
CASES.append(TestCase("shock_2species_complete",
                      output_parameters=np.array([["Complete_Distribution"]])))

################################################################################
#                                   Main                                       #
################################################################################
if __name__ == "__main__":
    for tc in CASES:
        assert isinstance(tc, TestCase)
        tc.update_results()
