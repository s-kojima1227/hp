import numpy as np
from ..vo import Events, EventsFactory as EF
from ..function import Kernels

class ThinningMethod:
    def __init__(self, baselines: np.ndarray, kernels: Kernels):
        if baselines.shape[0] != kernels.value.shape[0] or baselines.shape[0] != kernels.value.shape[1]:
            raise ValueError('基底強度パラメータとカーネル関数の次元が一致しません')

        self._dim = baselines.shape[0]
        self._baselines = baselines
        self._kernels = kernels.value

    def __call__(self, end_time) -> Events:
        events = [np.empty(0) for _ in range(self._dim)]
        lambda_ast = np.sum(self._baselines)
        t = 0

        while t < end_time:
            # 平均 1/lambda_ast の指数分布に従う乱数Eを生成し、次のイベントの候補の発生時刻を t <- t + E とする
            E = np.random.exponential(1 / lambda_ast)
            t += E

            if t > end_time: break

            # 採択率rを計算する
            lambda_k_s = np.array([self._lambda_k(k, t, events) for k in range(self._dim)])
            lambda_ = np.sum(lambda_k_s)
            r = lambda_ / lambda_ast
            # 一様乱数Uを生成し、rと比較する
            U = np.random.uniform()

            # U <= r ならば、イベントを採択する
            if U <= r:
                # どのノードでイベントが発生するかを決める
                k = np.random.choice(self._dim, p=lambda_k_s/lambda_)
                events[k] = np.append(events[k], t)
                lambda_ast = lambda_ + np.sum([self._kernels[k, i](0) for i in range(self._dim)])
            else:
                lambda_ast = lambda_

        return EF.from_events_grouped_by_mark(events, end_time)

    def _lambda(self, t, events):
        return np.sum([self._lambda_k(k, t, events) for k in range(self._dim)])

    def _lambda_k(self, k, t, events):
        return self._baselines[k] + np.sum([np.sum(self._kernels[k, i](t - events[i][events[i] < t])) for i in range(self._dim)])






