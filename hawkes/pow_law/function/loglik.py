from ...base import LogLik as Base, Events
from .intensity import Intensities
from .compensator import compensators
import numpy as np
from ..converter import ParamsConverter
from scipy.special import gamma, digamma
from .kernel import Kernels

class LogLik(Base):
    def _intensity_i(self, mark, time, events: Events, params):
        intensities = Intensities(params, events)
        return intensities[mark](time)

    def _compensators(self, time, events: Events, params):
        return compensators(time, events, params)

    def grad(self, params):
        dim = self._events.dim

        if dim != 1:
            raise NotImplementedError('勾配計算は現在1次元のみ対応しています')

        events = self._events.grouped_by_mark[0]
        T = self._events.end_time

        mu, K, p, c = params
        n = len(events)
        G = np.zeros(n)
        dG_dK = np.zeros(n)
        dG_dp = np.zeros(n)
        dG_dc = np.zeros(n)

        num_division = 16
        Delta = 1.0 / num_division
        s = np.linspace(-9, 9, num_division * 18 + 1)
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
