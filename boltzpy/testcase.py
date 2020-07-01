
import boltzpy as bp
import numpy as np
import os


class TestCase(bp.Simulation):
    def __init__(self,
                 file_address,
                 s=None,
                 t=None,
                 sv=None,
                 coll=None,
                 geometry=None,
                 scheme=None):
        super().__init__(file_address)

        if s is None:
            s = bp.Species()
            s.add(mass=2,
                  collision_rate=np.array([50], dtype=float))
            s.add(mass=3,
                  collision_rate=np.array([50, 50], dtype=float))
        else:
            assert isinstance(s, bp.Species)
        self.s = s

        if t is None:
            t = bp.Grid(shape=(5,),
                        delta=0.01/3,
                        spacing=3)
        self.t = t

        if sv is None:
            spacings = bp.SVGrid.default_spacing(s.mass)
            shapes = np.array([(int(2*m + 1), int(2*m + 1))
                               for m in s.mass])
            delta = 2 * 1.5 / np.max((shapes[:, 0] - 1) * spacings)
            sv = bp.SVGrid(s.mass,
                           shapes,
                           delta,
                           spacings)
        self.sv = sv

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
                    velocity_grids=sv,
                    species=s),
                bp.InnerPointRule(
                    initial_rho=left_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=np.arange(1, 3),
                    velocity_grids=sv,
                    species=s),
                bp.InnerPointRule(
                    initial_rho=right_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=np.arange(3, 5),
                    velocity_grids=sv,
                    species=s),
                bp.BoundaryPointRule(
                    initial_rho=right_rho,
                    initial_drift=initial_drift,
                    initial_temp=initial_temp,
                    affected_points=[5],
                    velocity_grids=sv,
                    reflection_rate_inverse=np.full(s.size, 0.25, dtype=float),
                    reflection_rate_elastic=np.full(s.size, 0.25, dtype=float),
                    reflection_rate_thermal=np.full(s.size, 0.25, dtype=float),
                    absorption_rate=np.full(s.size, 0.25, dtype=float),
                    surface_normal=np.array([1, 0], dtype=int),
                    species=s)
                ]
            geometry = bp.Geometry(shape=(6,), delta=0.5, rules=rules)
        self.geometry = geometry

        if scheme is None:
            scheme = bp.Scheme(OperatorSplitting="FirstOrder",
                               Transport="FiniteDifferences_FirstOrder",
                               Transport_VelocityOffset=np.array([-0.2, 0.0]),
                               Collisions_Generation="UniformComplete",
                               Collisions_Computation="EulerScheme")
        self.scheme = scheme

        if coll is None:
            coll = bp.Collisions()
            coll.setup(scheme=self.scheme, svgrid=self.sv, species=self.s)
        self.coll = coll
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

    @property
    def shape_of_results(self):
        shape_of_results = super().shape_of_results
        for (s, species_name) in enumerate(self.s.names):
            [beg, end] = self.sv.index_range[s]
            velocities = end - beg
            shape = (self.t.size, self.p.size, velocities)
            shape_of_results[species_name]['state'] = shape
        return shape_of_results

    def write_results(self, data, tw_idx, hdf_group):
        super().write_results(data, tw_idx, hdf_group)
        for (s, species_name) in enumerate(self.s.names):
            (beg, end) = self.sv.index_range[s]
            spc_group = hdf_group[species_name]
            # complete distribution
            spc_group["state"][tw_idx] = data.state[..., beg:end]
        # update index of current time step
        hdf_group.attrs["t"] = tw_idx + 1
        return

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
tc1_s = bp.Species()
tc1_s.add(mass=2, collision_rate=np.array([50], dtype=float))
CASES.append(TestCase("shock_monospecies",
                      s=tc1_s)
             )

# Two Species, eqal mass, shock,
tc2_s = bp.Species()
tc2_s.add(mass=2, collision_rate=np.array([50], dtype=float))
tc2_s.add(mass=2, collision_rate=np.array([50, 50], dtype=float))
CASES.append(TestCase("shock_2Species_equalMass",
                      s=tc2_s))


# Convergent Collision model
convergent_scheme = bp.Scheme(OperatorSplitting="FirstOrder",
                              Transport="FiniteDifferences_FirstOrder",
                              Transport_VelocityOffset=np.array([-0.2, 0.0]),
                              Collisions_Generation="Convergent",
                              Collisions_Computation="EulerScheme")
CASES.append(TestCase("shock_2species_convergent",
                      scheme=convergent_scheme))

FILES = [tc.file_address for tc in CASES]


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
