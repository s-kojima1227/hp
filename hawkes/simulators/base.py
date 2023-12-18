from abc import ABCMeta, abstractmethod
import numpy as np

class BaseSimulator(metaclass=ABCMeta):
    def __init__(self, mus):
        """
        Parameters
        ----------
        mus : np.ndarray, shape=(n,)
        kernels : np.ndarray, shape=(n, n)
            カーネル関数の行列
        """
        self._n_nodes = mus.shape[0]
        self._mus = mus
        self._kernels = self._build_kernels()

    @abstractmethod
    def _build_kernels(self):
        """カーネル関数を構築する"""
        pass

    def simulate(self, T):
        """シミュレーションを実行する

        Parameters
        ----------
        T : float
            シミュレーションの終了時刻

        Returns
        -------
        events : list of np.ndarray
            各ノードのイベント時刻のリスト
        """

        events = [np.empty(0) for _ in range(self._n_nodes)]
        lambda_ast = np.sum(self._mus)
        t = 0

        while t < T:
            # 平均 1/lambda_ast の指数分布に従う乱数Eを生成し、次のイベントの候補の発生時刻を t <- t + E とする
            E = np.random.exponential(1 / lambda_ast)
            t += E

            if t > T: break

            # 採択率rを計算する
            lambda_k_s = np.array([self._lambda_k(k, t, events) for k in range(self._n_nodes)])
            lambda_ = np.sum(lambda_k_s)
            r = lambda_ / lambda_ast
            # 一様乱数Uを生成し、rと比較する
            U = np.random.uniform()

            # U <= r ならば、イベントを採択する
            if U <= r:
                # どのノードでイベントが発生するかを決める
                k = np.random.choice(self._n_nodes, p=lambda_k_s/lambda_)
                events[k] = np.append(events[k], t)
                lambda_ast = lambda_ + np.sum([self._kernels[k, i](0) for i in range(self._n_nodes)])
            else:
                lambda_ast = lambda_

        return events if self._n_nodes >= 2 else events[0]

    def _lambda(self, t, events):
        """全ノードにおける条件付き強度を計算する"""
        return np.sum([self._lambda_k(k, t, events) for k in range(self._n_nodes)])

    def _lambda_k(self, k, t, events):
        """ノードkにおける条件付き強度を計算する"""
        return self._mus[k] + np.sum([np.sum(self._kernels[k, i](t - events[i][events[i] < t])) for i in range(self._n_nodes)])



