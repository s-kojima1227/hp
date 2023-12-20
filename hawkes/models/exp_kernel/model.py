import numpy as np
from .kernel import ExpKernel
from .loglik import ExpKernelLogLik
from ...simulators import ThinningSimulator
from ...optimizers import OptimizerBuilder
from ..intensity_fn import IntensityFunction
from ..dto.simutation import SimulationOutput

class ExpKernelModel:
    def __init__(self):
        self._is_fitted = False

    def fit(self, events, T, optimizer_settings=None):
        # 1次元の場合の対応
        if isinstance(events, np.ndarray):
            events = [events]
        self._is_fitted = True
        self._events = events
        self._T = T

        dim = len(events)
        self._search_space = {
            'mu': np.array([[0, 10] for _ in range(dim)]),
            'a': np.array([[[0, 10] for _ in range(dim)] for _ in range(dim)]),
            'b': np.array([[[0, 10] for _ in range(dim)] for _ in range(dim)]),
        }
        init_params = np.array([0.1, 0.1, 0.1])
        params_order = ['mu', 'a', 'b']
        optimizer = OptimizerBuilder(optimizer_settings, dim, self._search_space, init_params, params_order)()

        log_lik_fn = ExpKernelLogLik(events, T)
        return optimizer(log_lik_fn)

    def score(self, mu, a, b, events, T):
        # 1次元の場合の対応
        if isinstance(mu, (int, float)):
            mu = np.array([mu])
            a = np.array([[a]])
            b = np.array([[b]])
            events = [events]
        log_lik_fn = ExpKernelLogLik(events, T)
        return log_lik_fn(mu, a, b)

    def simulate(self, mu, a, b, T, delta=1):
        # 1次元の場合の対応
        if isinstance(mu, (int, float)):
            mu = np.array([mu])
            a = np.array([[a]])
            b = np.array([[b]])

        if a.shape != b.shape:
            raise ValueError('パラメータaとパラメータbのサイズが一致しません')

        dim = mu.shape[0]
        kernel = np.array([[ExpKernel(a[i, j], b[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)
        events = ThinningSimulator(mu, kernel)(T)
        t_vals = np.arange(0, T + delta, delta)
        intensity = IntensityFunction(mu, kernel, events)(t_vals)

        return SimulationOutput(events, T, intensity, params={'mu': mu, 'a': a, 'b': b}, kernel_type='exp_kernel')
