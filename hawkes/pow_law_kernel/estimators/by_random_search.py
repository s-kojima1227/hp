import numpy as np
from ..kernel import PowLawKernelIntensity
from ..loglik import PowLawKernelLogLik
from ...base import RandomSearchOptimizer, EstimationOutput

class PowLawKernelModelEstimatorByRandomSearch:
    def __call__(self, events, T, n_iter=1000000, search_space=None, delta=1):
        # 1次元の場合の対応
        if isinstance(events, np.ndarray):
            events = [events]

        dim = len(events)

        # パラメータの探索範囲が設定されていない場合は、デフォルトの探索範囲を設定する
        if search_space is None:
            search_space = self._default_search_space(dim)
        else:
            search_space = self._format_search_space(search_space, dim)

        # ランダムサーチオプティマイザ生成
        optimizer = RandomSearchOptimizer(n_iter, search_space)

        # パラメータ推定
        params, score = optimizer(PowLawKernelLogLik(events, T))
        mu, K, p, c = (params['mu'], params['K'], params['p'], params['c'])

        # 強度計算
        t_vals = np.arange(0, T + delta, delta)
        intensity = PowLawKernelIntensity(mu, K, p, c, events)(t_vals)

        # 推定結果インスタンス生成
        output = EstimationOutput(events, T, intensity, params={'mu': mu, 'K': K, 'p': p, 'c': c}, kernel_type='pow_law_kernel', loglik=score)

        return output

    def _format_search_space(self, search_space, dim):
        if dim == 1:
            return {
                'mu': np.array([search_space['mu']]),
                'K': np.array([[search_space['K']]]),
                'p': np.array([[search_space['p']]]),
                'c': np.array([[search_space['c']]]),
            }

        return search_space

    def _default_search_space(self, dim):
        return {
            'mu': np.array([[0, 10] for _ in range(dim)]),
            'K': np.array([[[0, 10] for _ in range(dim)] for _ in range(dim)]),
            'p': np.array([[[1, 10] for _ in range(dim)] for _ in range(dim)]),
            'c': np.array([[[1, 10] for _ in range(dim)] for _ in range(dim)]),
        }
