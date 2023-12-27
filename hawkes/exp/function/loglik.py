from hawkes.base.vo.events import Events
from ...base import LogLik as Base
from .intensity import Intensities
from .compensator import compensators
import numpy as np
from ..converter import ParamsConverter

class LogLik(Base):
    def _intensity_i(self, mark, time, events: Events, params):
        intensities = Intensities(params, events)
        return intensities[mark](time)

    def _compensators(self, time, events: Events, params):
        return compensators(time, events, params)

    def grad(self, params):
        dim = self._events.dim
        T = self._events.end_time

        if self._events.dim != 1:
            raise NotImplementedError('指数カーネルの場合の対数尤度の勾配計算は現在1次元のみ対応しています')

        events = self._events.grouped_by_mark[0]
        mu, a, b = ParamsConverter.unpack(params, dim)
        n = len(events)
        G = np.zeros(n)
        dG_db = np.zeros(n)

        for i in range(n - 1):
            diff_t = events[i + 1] - events[i]
            G[i + 1] = (G[i] + a * b) * np.exp(-b * diff_t)
            dG_db[i + 1] = (dG_db[i] + a) * np.exp(-b * diff_t) - G[i + 1] * diff_t

        lambda_ = G + mu
        dlambda_da = G / a
        dlambda_db = dG_db

        dlogL_dmu = np.sum(1 / lambda_) - T
        dlogL_da = np.sum(dlambda_da / lambda_) - np.sum(1 - np.exp(-b * (T - events)))
        dlogL_db = np.sum(dlambda_db / lambda_) - np.sum(a * (T - events) * np.exp(-b * (T - events)))

        return np.array([dlogL_dmu, dlogL_da, dlogL_db])

    # MEMO: 指数カーネルの場合には、下記の高速化ができる
    def _calc_loglik(self, params):
        dim = self._events.dim
        T = self._events.end_time
        mu = params[:dim]
        a = params[dim:dim * (dim + 1)].reshape(dim, dim)
        b = params[dim * (dim + 1):].reshape(dim, dim)
        events = self._events.grouped_by_mark

        log_lik = 0

        for i in range(dim):
            log_lik -= mu[i] * T
            g_i, sum_G_i = self._calc_weights_dim_i(i, b[i])

            for k in range(events[i].shape[0]):
                s = mu[i] + a[i] @ (b[i] * g_i[k])
                log_lik += np.log(s)

            log_lik -= a[i] @ sum_G_i

        return log_lik

    def _calc_weights_dim_i(self, i, b_i):
        dim = self._events.dim
        events = self._events.grouped_by_mark
        T = self._events.end_time

        t_i = events[i]
        n_jumps_i = t_i.shape[0]
        g_i = np.zeros((n_jumps_i, dim))
        G_i = np.zeros((n_jumps_i + 1, dim))
        sum_G_i = np.zeros(dim)

        for j in range(dim):
            t_j = events[j]
            n_jumps_j = t_j.shape[0]
            ij = 0
            for k in range(n_jumps_i + 1):
                t_i_k = t_i[k] if k < n_jumps_i else T
                if k > 0:
                    ebt = np.exp(-b_i[j] * (t_i_k - t_i[k - 1]))

                    if k < n_jumps_i:
                        g_i[k, j] = g_i[k - 1, j] * ebt
                    G_i[k, j] = g_i[k - 1, j] * (1 - ebt)
                else:
                    if k < n_jumps_i:
                        g_i[k, j] = 0
                    G_i[k, j] = 0
                    sum_G_i[j] = 0

                while ij < n_jumps_j and t_j[ij] < t_i_k:
                    ebt = np.exp(-b_i[j] * (t_i_k - t_j[ij]))
                    if k < n_jumps_i:
                        g_i[k, j] += ebt
                    G_i[k, j] += 1 - ebt
                    ij += 1

                sum_G_i[j] += G_i[k, j]

        return g_i, sum_G_i