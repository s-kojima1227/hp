import numpy as np
from .base import BaseKernel

class ExpKernel(BaseKernel):
    def __init__(self, a: float, b: float):
        self._a = a
        self._b = b

    def __call__(self, t_s):
        return self._a * self._b * np.exp(-self._b * t_s)
