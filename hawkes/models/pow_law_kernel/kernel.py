import numpy as np

class PowLawKernel:
    def __init__(self, K: float, p: float, c: float):
        self._K = float(K)
        self._p = float(p)
        self._c = float(c)

    def __call__(self, t_s):
        return self._K * np.power(t_s + self._c, -self._p)
