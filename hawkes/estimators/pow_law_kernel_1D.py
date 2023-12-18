import numpy as np
from scipy.special import gamma, digamma
from .log_likelihoods.pow_law_kernel_1D import PowLawKernel1DLogLik

class PowLawKernel1DEstimator:
    def __init__(self, delta=0.00001, epsilon=0.01):
        self._delta = delta
        self._epsilon = epsilon
        self._is_fitted = False

    def fit(self, events, T, init_params=np.array([0.1, 0.1, 0.1, 0.1])):
        self._is_fitted = True
        self._events = events
        self._T = T
        return self._fit_grad(events, init_params)

    def log_likelihood(self, params):
        if not self._is_fitted:
            raise Exception("データをフィットしてください")
        log_lik = PowLawKernel1DLogLik()
        return log_lik(params, self._events, self._T)

    def _fit_grad(self, events, init_params):
        self._events = events
        params = init_params
        while True:
            grad = self._grad(params)
            params += self._delta * grad
            if np.linalg.norm(grad) < self._epsilon:
                break

        return params

    def _grad(self, params):
        mu, K, p, c = params
        T = self._T
        events = self._events
        n = len(events)
        G = np.zeros(n)
        dG_dK = np.zeros(n)
        dG_dp = np.zeros(n)
        dG_dc = np.zeros(n)

        Delta = 1.0 / 16
        s = np.linspace(-9, 9, (1 / Delta) * 18 + 1)
        log_psi = s - np.exp(-s)
        log_dpsi = log_psi + np.log(1 + np.exp(-s))
        psi = np.exp(log_psi)
        H_G = Delta * K * np.exp(log_dpsi + (p - 1) * log_psi - c * psi) / gamma(p)
        H_dG_dp = Delta * K * np.exp(log_dpsi + (p - 1) * log_psi - c * psi) * (log_psi - digamma(p)) / gamma(p)
        H_dG_dc = -Delta * K * np.exp(log_dpsi + (p - 1) * log_psi - c * psi) * psi / gamma(p)

        G_x = np.zeros_like(s)

        for i in range(n - 1):
            G_x = (G_x + 1) * np.exp(-psi * (events[i + 1] - events[i]))
            G[i + 1] = G_x @ H_G
            dG_dK[i + 1] = G[i + 1] / K
            dG_dp[i + 1] = G_x @ H_dG_dp
            dG_dc[i + 1] = G_x @ H_dG_dc

        lambda_ = G + mu
        dlambda_dK = dG_dK
        dlambda_dp = dG_dp
        dlambda_dc = dG_dc

        dlogL_dmu = np.sum(1 / lambda_) - T
        dlogL_dK = np.sum(dlambda_dK / lambda_) \
            + 1 / (p - 1) * np.sum(np.power(T - events + c, -(p - 1)) - np.power(c, -(p - 1)))
        dLogL_dp = np.sum(dlambda_dp / lambda_) \
            - K / (p - 1)**2 * np.sum(np.power(T - events + c, -(p - 1)) - np.power(c, -(p - 1))) \
            - K / (p - 1) * np.sum(np.log(T - events + c) * np.power(T - events + c, -(p - 1)) - np.log(c) * np.power(c, -(p - 1)))
        dLogL_dc = np.sum(dlambda_dc / lambda_) - K * np.sum(np.power(T - events + c, -p) - np.power(c, -p))

        return np.array([dlogL_dmu, dlogL_dK, dLogL_dp, dLogL_dc])
