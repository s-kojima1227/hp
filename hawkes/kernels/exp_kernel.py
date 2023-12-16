import numpy as np
from .base import BaseKernel

class ExpKernel(BaseKernel):
    def __init__(self, alpha: float, beta: float):
        self._alpha = alpha
        self._beta = beta

    def __call__(self, t_s):
        return self._alpha * self._beta * np.exp(-self._beta * t_s)
