
import boltzpy as bp
import numpy as np
import os


class TestCase(bp.Simulation):
    def __init__(self,
                 file_address,
                 t=None,
                 sv=None,
                 coll=None,
                 geometry=None,
                 scheme=None,
                 log_state=True):
        if t is None:
            t = bp.Grid(shape=(5,),
                        delta=0.01/3,
                        spacing=3)

        if sv is None:
            sv = bp.SVGrid([2, 3],
                           [[5, 5], [7, 7]],
                           1/8,
                           [6, 4],
                           [[50, 50], [50, 50]])

        if geometry is None:
            left_rho = 2*np.ones(sv.specimen)
            right_rho = np.ones(sv.specimen)
            initial_drift = np.zeros((sv.specimen, sv.ndim))
            initial_temp = np.ones(sv.specimen)
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
                    affected_points=np.arange(1, 3),
                    velocity_grids=sv),
                bp.InnerPointRule(
                    initial_rho=right_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=np.arange(3, 5),
                    velocity_grids=sv),
                bp.BoundaryPointRule(
                    initial_rho=right_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=[5],
                    velocity_grids=sv,
                    reflection_rate_inverse=np.full(sv.specimen, 0.25, dtype=float),
                    reflection_rate_elastic=np.full(sv.specimen, 0.25, dtype=float),
                    reflection_rate_thermal=np.full(sv.specimen, 0.25, dtype=float),
                    absorption_rate=np.full(sv.specimen, 0.25, dtype=float),
                    surface_normal=np.array([1, 0], dtype=int))
                ]
            geometry = bp.Geometry(shape=(6,), delta=0.5, rules=rules)

        if scheme is None:
            scheme = bp.Scheme(OperatorSplitting="FirstOrder",
                               Transport="FiniteDifferences_FirstOrder",
                               Transport_VelocityOffset=np.array([-0.2, 0.0]),
                               Collisions_Generation="UniformComplete",
                               Collisions_Computation="EulerScheme")

        if coll is None:
            coll = bp.Collisions()
            coll.setup(scheme=scheme, model=sv)

        super().__init__(t, geometry, sv, coll, scheme, file_address, log_state)
        return

    @property
    def default_directory(self):
        return __file__[:-19] + 'test_data/'

    @property
    def temporary_file(self):
        """:obj:`str` :
        Default file address for temporary test results.
        """
        return self.default_directory + '_tmp_.hdf5'

    @staticmethod
    def load(file_address):
        """Set up and return a :class:`TestCase` instance
        based on the parameters in the given HDF5 group.

        Parameters
        ----------
        file_address : :obj:`str`, optional
            The full path to the simulation (hdf5) file.

        Returns
        -------
        self : :class:`TestCase`
        """
        simulation = bp.Simulation.load(file_address)
        # ignore private attributes
        params = {key: value
                  for (key, value) in simulation.__dict__.items()
                  if key[0] != "_"}
        params["file_address"] = simulation.file_address
        self = TestCase(**params)
        self.check_integrity(complete_check=False)
        return self


################################################################################
#                           Implemented TestCases                              #
################################################################################
CASES = list()

# Mono Species, shock
tc1_sv = bp.SVGrid([2],
                   [[5, 5]],
                   1.5 / 8,
                   [4],
                   [[50]])
CASES.append(TestCase("shock_monospecies",
                      sv=tc1_sv)
             )

# Two Species, eqal mass, shock,
tc2_sv = bp.SVGrid([2, 2],
                   [[5, 5], [5,5]],
                   1.5 / 8,
                   [4, 4],
                   [[50, 50], [50, 50]])
CASES.append(TestCase("shock_2Species_equalMass",
                      sv=tc2_sv))


# Convergent Collision model
convergent_scheme = bp.Scheme(OperatorSplitting="FirstOrder",
                              Transport="FiniteDifferences_FirstOrder",
                              Transport_VelocityOffset=np.array([-0.2, 0.0]),
                              Collisions_Generation="Convergent",
                              Collisions_Computation="EulerScheme")
CASES.append(TestCase("shock_2species_convergent",
                      scheme=convergent_scheme))

FILES = [tc.file_address for tc in CASES]


# for tc in FILES:
#     import h5py
#     h5py.File(tc).attrs["log_state"] = True

################################################################################
#                                   Main                                       #
################################################################################
def replace_all_tests():
    msg = input("Are you absolutely sure? "
                "You are about to replace all test cases (yes/no)")
    if msg == "yes":
        for tc in CASES:
            print("TestCase = ", tc.file_address)
            assert isinstance(tc, TestCase)
            if os.path.isfile(tc.file_address):
                os.remove(tc.file_address)
            tc.compute()
    else:
        print("Aborted replacing testcases!")
