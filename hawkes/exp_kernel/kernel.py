import numpy as np
from ..base import IntensityFunction

class ExpKernel:
    def __init__(self, a: float, b: float):
        self._a = a
        self._b = b

    def __call__(self, t_s):
        return self._a * self._b * np.exp(-self._b * t_s)

class ExpKernelMatrix:
    @staticmethod
    def build(a: np.ndarray, b: np.ndarray):
        if not (a.shape == b.shape):
            raise ValueError('パラメーターの次元が不適切です: a.shape={}, b.shape={}'.format(a.shape, b.shape))
        dim = a.shape[0]
        return np.array([[ExpKernel(a[i, j], b[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)

class ExpKernelIntensity(IntensityFunction):
    def __init__(self, mu, a, b, events):
        super().__init__(mu, ExpKernelMatrix.build(a, b), events)
