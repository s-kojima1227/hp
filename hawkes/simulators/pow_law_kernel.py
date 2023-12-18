import numpy as np
from .base import BaseSimulator
from ..kernels.pow_law_kernel import PowLawKernel

class PowLowKernelSimulator(BaseSimulator):
    def __init__(self, mus, K_s, p_s, c_s):

        # 1次元の場合の対応
        if isinstance(mus, (int, float)):
            mus = np.array([mus])
            K_s = np.array([[K_s]])
            p_s = np.array([[p_s]])
            c_s = np.array([[c_s]])

        if not (K_s.shape == p_s.shape == c_s.shape):
            raise ValueError('パラメーターの次元が不適切です')

        self._K_s = K_s
        self._p_s = p_s
        self._c_s = c_s

        super().__init__(mus)

    def _build_kernels(self):
        n_nodes = self._K_s.shape[0]
        kernels = np.empty((n_nodes, n_nodes), dtype=object)

        for i in range(n_nodes):
            for j in range(n_nodes):
                kernels[i, j] = PowLawKernel(self._K_s[i, j], self._p_s[i, j], self._c_s[i, j])

        return kernels

