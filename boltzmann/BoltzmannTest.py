import boltzmann as bP


class BoltzmannTest(bP.BoltzmannPDE):
    r"""Test Utility

    Provides Unit tests and more complicated Tests.
    Tests are done by a second, non-optimized. and simple implementation of
    (preferably) all methods.
    """
    # TODO a lot to do here
    def __init__(self, *args, **kwargs):
        super(BoltzmannTest, self).__init__(*args, **kwargs)
