import numpy as np
from .base import BaseSimulator
from ..kernels.exp_kernel import ExpKernel

class ExpKernelSimulator(BaseSimulator):
    def __init__(self, mu_s, a_s, b_s):
        """初期化メソッド

        Parameters
        ----------
        mus : np.ndarray, shape=(n,)
            ベースライン
        alphas : np.ndarray, shape=(n, n)
            イベント発生時のインパルス
        betas : np.ndarray, shape=(n, n)
        """

        # 1次元の場合の対応
        if isinstance(mu_s, (int, float)):
            mu_s = np.array([mu_s])
            a_s = np.array([[a_s]])
            b_s = np.array([[b_s]])

        if a_s.shape != b_s.shape:
            raise ValueError('パラメーターの次元が不適切です')

        self._a_s = a_s
        self._b_s = b_s

        super().__init__(mu_s)

    def _build_kernels(self):
        n_nodes = self._a_s.shape[0]
        kernels = np.empty((n_nodes, n_nodes), dtype=object)

        for i in range(n_nodes):
            for j in range(n_nodes):
                kernels[i, j] = ExpKernel(self._a_s[i, j], self._b_s[i, j])

        return kernels

