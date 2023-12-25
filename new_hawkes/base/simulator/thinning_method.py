import numpy as np

class ThinningMethod:
    """間引き法によるHawkes過程のシミュレータ"""
    def __init__(self, mu, kernel):
        """
        Parameters
        ----------
        mus : np.ndarray, shape=(n,)
        kernels : np.ndarray, shape=(n, n)
            カーネル関数の行列
        """
        if mu.shape[0] != kernel.shape[0] or mu.shape[0] != kernel.shape[1]:
            raise ValueError('基底強度パラメータとカーネル関数の次元が一致しません')

        self._dim = mu.shape[0]
        self._mu = mu
        self._kernel = kernel

    def __call__(self, T):
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

        events = [np.empty(0) for _ in range(self._dim)]
        lambda_ast = np.sum(self._mu)
        t = 0

        while t < T:
            # 平均 1/lambda_ast の指数分布に従う乱数Eを生成し、次のイベントの候補の発生時刻を t <- t + E とする
            E = np.random.exponential(1 / lambda_ast)
            t += E

            if t > T: break

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
                lambda_ast = lambda_ + np.sum([self._kernel[k, i](0) for i in range(self._dim)])
            else:
                lambda_ast = lambda_

        return events

    def _lambda(self, t, events):
        """全ノードにおける条件付き強度を計算する"""
        return np.sum([self._lambda_k(k, t, events) for k in range(self._dim)])

    def _lambda_k(self, k, t, events):
        """ノードkにおける条件付き強度を計算する"""
        return self._mu[k] + np.sum([np.sum(self._kernel[k, i](t - events[i][events[i] < t])) for i in range(self._dim)])






