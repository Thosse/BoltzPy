
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
        else:
            assert isinstance(s, bp.Species)
        self["s"] = s

        if t is None:
            t = bp.Grid(ndim=1,
                        shape=(5,),
                        physical_spacing=0.01,
                        spacing=3)
        self["t"] = t

        if p is None:
            p = bp.Grid(ndim=1,
                        shape=(6,),
                        spacing=1,
                        physical_spacing=0.5)
        self["p"] = p

        if sv is None:
            spacings = bp.SVGrid.generate_spacings(s.mass)
            shapes = [(int(2*m + 1), int(2*m + 1)) for m in s.mass]
            sv = bp.SVGrid(ndim=2,
                           maximum_velocity=1.5,
                           shapes=shapes,
                           spacings=spacings,
                           )
        self["sv"] = sv

        if geometry is None:
            left_rho = 2*np.ones(s.size)
            right_rho = np.ones(s.size)
            initial_drift = np.zeros((s.size, sv.ndim))
            initial_temp = np.ones(s.size)
            rules = [
                bp.ConstantPointRule(
                    initial_rho=left_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=[0],
                    velocity_grids=sv),
                bp.InnerPointRule(
                    initial_rho=left_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=np.arange(1, p.size // 2),
                    velocity_grids=sv),
                bp.InnerPointRule(
                    initial_rho=right_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=np.arange(p.size // 2, p.size - 1),
                    velocity_grids=sv),
                bp.ConstantPointRule(
                    initial_rho=right_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=[p.size - 1],
                    velocity_grids=sv)
                ]
            geometry = bp.Geometry(shape=p.shape,
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
        if sim.coll == bp.Collisions():
            sim.coll.setup(scheme=sim.scheme, svgrid=sim.sv, species=sim.s)
        sim.save()
        sim.compute()
        return sim

    def compare_results(self):
        address_old = self.address(self["file_name"])
        assert os.path.exists(address_old)
        address_new = bp_c.TEST_TMP_FILE
        assert address_old != address_new
        # remove new_Address, if it exists already
        if os.path.exists(address_new):
            os.remove(address_new)
        self.save_results(address_new)
        output = "No comparison done so far"
        try:
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

    def replace_results(self):
        msg = input("Are you absolutely sure? "
                    "You are replacing this test case (yes/no)")
        if msg == "yes":
            os.remove(self.address(self["file_name"]))
            self.save_results()
            print("Successfully updated: ", self["file_name"])
        else:
            print("Abort replacing testcase: ", self["file_name"])


################################################################################
#                           Implemented TestCases                              #
################################################################################
CASES = list()

# Mono Species, shock
tc1_s = bp.Species()
tc1_s.add(mass=2, collision_rate=np.array([50], dtype=float))
CASES.append(TestCase("shock_monospecies",
                      s=tc1_s)
             )

# Two Species, eqal mass, shock
tc2_s = bp.Species()
tc2_s.add(mass=2, collision_rate=np.array([50], dtype=float))
tc2_s.add(mass=2, collision_rate=np.array([50, 50], dtype=float))
CASES.append(TestCase("shock_2Species_equalMass",
                      s=tc2_s))

# Two Species, shock, complete distribution
CASES.append(TestCase("shock_2species_complete",
                      output_parameters=np.array([["Complete_Distribution"]])))


################################################################################
#                                   Main                                       #
################################################################################
def update_all_tests():
    for tc in CASES:
        print("TestCase = ", tc["file_name"])
        assert isinstance(tc, TestCase)
        tc.update_results()


def replace_all_tests():
    for tc in CASES:
        print("TestCase = ", tc["file_name"])
        assert isinstance(tc, TestCase)
        tc.replace_results()
