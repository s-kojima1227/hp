import numpy as np
from ..kernel import PowLawKernelIntensity
from ..loglik import PowLawKernelLogLik
from ...base import GradientMethodOptimizer, EstimationOutput

class PowLawKernelModelEstimatorByGradientMethod:
    def __call__(self, events, T, learning_rate=0.00001, tol=0.001, max_iter=1000000, init_params=None, delta=1):
        if isinstance(events, np.ndarray):
            events = [events]

        dim = len(events)

        if dim != 1:
            raise ValueError('多次元の場合は未対応です')

        # オプティマイザー用に初期値の整形
        init_params = self._format_params(init_params)

        # 勾配法オプティマイザー生成
        optimizer = GradientMethodOptimizer(
            learning_rate=learning_rate,
            tol=tol,
            max_iter=max_iter,
            init_params=init_params
        )

        # パラメータ推定
        params = optimizer(PowLawKernelLogLik(events, T))

        # FIXME: 1次元の場合に限定した一時的な対応
        mu, K, p, c = params
        mu, K, p, c = np.array([mu]), np.array([[K]]), np.array([[p]]), np.array([[c]])
        score = PowLawKernelLogLik(events, T)(mu, K, p, c)
        # --------------

        # 強度計算
        t_vals = np.arange(0, T + delta, delta)
        intensity = PowLawKernelIntensity(mu, K, p, c, events)(t_vals)

        # 推定結果インスタンス生成
        output = EstimationOutput(events, T, intensity, params={'mu': mu, 'K': K, 'p': p, 'c': c}, kernel_type='pow_law_kernel', loglik=score)

        return output

    def _format_params(self, params):
        # 初期値が設定されていない場合は、デフォルトの初期値を設定する
        if params is None:
            return self._default_init_params()
        # 初期値が辞書型である場合は、numpy配列に変換する
        elif isinstance(params, dict):
            return np.array([params['mu'], params['K'], params['p'], params['c']])
        else:
            raise ValueError('init_paramsは辞書型である必要があります')

    def _default_init_params(self):
        return np.array([0.1, 0.1, 2, 2])
