import numpy as np
from .kernel import ExpKernelMatrix
from ..base import ThinningMethodSimulator, IntensityFunction, SimulationOutput

class ExpKernelModelSimulator:
    def __init__(self, delta=1):
        self._delta = delta

    def __call__(self, mu, a, b, T):
        # 1次元の場合の対応
        if isinstance(mu, (int, float)):
            mu = np.array([mu])
            a = np.array([[a]])
            b = np.array([[b]])

        kernel = ExpKernelMatrix.build(a, b)
        events = ThinningMethodSimulator(mu, kernel)(T)
        t_vals = np.arange(0, T + self._delta, self._delta)
        intensity = IntensityFunction(mu, kernel, events)(t_vals)

        return SimulationOutput(events, T, intensity, params={'mu': mu, 'a': a, 'b': b}, kernel_type='exp_kernel')
