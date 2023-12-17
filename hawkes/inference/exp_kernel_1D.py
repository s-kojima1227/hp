import numpy as np

class ExpKernel1D:
    def __init__(self, delta=0.00001, epsilon=0.01):
        self._delta = delta
        self._epsilon = epsilon

    def fit(self, events, T, init_params=np.array([0.1, 0.1, 0.1])):
        self._events = events
        self._T = T
        return self._fit_grad(events, init_params)

    # 勾配法による最尤法の数値解放
    def _fit_grad(self, events, init_params):
        self._events = events
        params = init_params
        while True:
            grad = self._grad(params)
            params += self._delta * grad
            if np.linalg.norm(grad) < self._epsilon:
                break

        return params

    # 勾配を計算する
    def _grad(self, params):
        mu, a, b = params
        T = self._T
        n = len(self._events)
        G = np.zeros(n)
        dG_db = np.zeros(n)

        for i in range(n - 1):
            diff_t = self._events[i + 1] - self._events[i]
            G[i + 1] = (G[i] + a * b) * np.exp(-b * diff_t)
            dG_db[i + 1] = (dG_db[i] + a) * np.exp(-b * diff_t) - G[i + 1] * diff_t

        lambda_ = G + mu
        dlambda_da = G / a
        dlambda_db = dG_db

        dlogL_dmu = np.sum(1 / lambda_) - T
        dlogL_da = np.sum(dlambda_da / lambda_) - np.sum(1 - np.exp(-b * (T - self._events)))
        dlogL_db = np.sum(dlambda_db / lambda_) - np.sum(a * (T - self._events) * np.exp(-b * (T - self._events)))

        return np.array([dlogL_dmu, dlogL_da, dlogL_db])
