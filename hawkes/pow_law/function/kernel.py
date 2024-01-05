import numpy as np
from ...base import Kernels as Base

class Kernels(Base):
    def __init__(self, multipliers: np.ndarray, exponents: np.ndarray, cutoffs: np.ndarray):
        if not (multipliers.shape == exponents.shape == cutoffs.shape):
            raise ValueError('パラメーターの次元が不適切です')
        dim = multipliers.shape[0]
        kernels = np.array([[self._Kernel(multipliers[i, j], exponents[i, j], cutoffs[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)
        super().__init__(kernels)

    class _Kernel:
        def __init__(self, multiplier: float, exponent: float, cutoff: float):
            self._multiplier = float(multiplier)
            self._exponent = float(exponent)
            self._cutoff = float(cutoff)

        def __call__(self, t):
            return self._multiplier * np.power(t + self._cutoff, -self._exponent)
