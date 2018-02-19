import boltzmann as bP


class BoltzmannTest(bP.BoltzmannPDE):
    r"""Test Utility

    Provides Unit tests and more complicated Tests.
    Tests are done by a second, non-optimized. and simple implementation of
    (preferably) all methods.

    .. todo::
        - create each class a second time,
          inheriting the original class and adding testing functions
          for each method and a test() function that applies all tests
        - Collisions;
          second test with max_v= lowest common multiple(masses)
          => i_ar should be the same
    """
    # TODO a lot to do here
    def __init__(self, *args, **kwargs):
        super(BoltzmannTest, self).__init__(*args, **kwargs)
