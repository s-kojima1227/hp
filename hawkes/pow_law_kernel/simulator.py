import numpy as np
from .kernel import PowLawKernelMatrix
from ..base import ThinningMethodSimulator, IntensityFunction, SimulationOutput

class PowLawKernelModelSimulator:
    def __init__(self, delta=1):
        self._delta = delta

    def __call__(self, mu, K, p, c, T):
        # 1次元の場合の対応
        if isinstance(mu, (int, float)):
            mu = np.array([mu])
            K = np.array([[K]])
            p = np.array([[p]])
            c = np.array([[c]])

        kernel = PowLawKernelMatrix.build(K, p, c)
        events = ThinningMethodSimulator(mu, kernel)(T)
        t_vals = np.arange(0, T + self._delta, self._delta)
        intensity = IntensityFunction(mu, kernel, events)(t_vals)

        return SimulationOutput(events, T, intensity, params={'mu': mu.tolist(), 'K': K.tolist(), 'p': p.tolist(), 'c': c.tolist()}, kernel_type='pow_law_kernel')
