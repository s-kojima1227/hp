import numpy as np
from ..base import IntensityFunction

class PowLawKernel:
    def __init__(self, K: float, p: float, c: float):
        self._K = float(K)
        self._p = float(p)
        self._c = float(c)

    def __call__(self, t_s):
        return self._K * np.power(t_s + self._c, -self._p)

class PowLawKernelMatrix:
    @staticmethod
    def build(K: np.ndarray, p: np.ndarray, c: np.ndarray):
        if not (K.shape == p.shape == c.shape):
            raise ValueError('パラメーターの次元が不適切です')
        dim = K.shape[0]
        return np.array([[PowLawKernel(K[i, j], p[i, j], c[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)

class PowLawKernelIntensity(IntensityFunction):
    def __init__(self, mu, K, p, c, events):
        super().__init__(mu, PowLawKernelMatrix.build(K, p, c), events)
