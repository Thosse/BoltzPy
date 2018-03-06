import numpy as np


class Animation:
    r"""Determines Calculation Output and handles animation or results

    .. todo::
        - implement complete output, for testing

    Attributes
    ----------
    moments : array(bool)
        Iff moments[i] is True,
        then SUPPORTED_OUTPUT[i] is calculated and animated.
        Array of shape=(len(SUPPORTED_OUTPUT),)
    save_animation : bool
        If True, save animation into video file.
        """
    def __init__(self):
        self.moments = np.full((0,),
                               True,
                               dtype=bool)
        self.save_animation = False
