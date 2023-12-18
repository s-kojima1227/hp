import numpy as np

class ExpKernel1DLogLik:
    def __init__(self, events, T):
        self._events = events
        self._T = T

    def __call__(self, params):
        mu, a, b = params
        events = self._events
        n = len(events)
        T = self._T
        log_lik = 0
        log_lik += np.sum([np.log(mu + np.sum(a * b * np.exp(-b * (events[i] - events[:i])))) for i in range(n)])
        log_lik -= mu * T
        log_lik -= np.sum(a * (1 - np.exp(-b * (T - events))))

        return log_lik

    def grad(self, params):
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
