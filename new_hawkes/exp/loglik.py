import numpy as np
from ..base import HawkesLogLik

class LogLik(HawkesLogLik):
    def __call__(self, params):
        # 最初のself._dim個はmu、次のself._dim * self._dim個はa、最後のself._dim * self._dim個はb
        mu = params[:self._dim]
        a = params[self._dim:self._dim * (self._dim + 1)].reshape(self._dim, self._dim)
        b = params[self._dim * (self._dim + 1):].reshape(self._dim, self._dim)

        log_lik = 0

        for i in range(self._dim):
            log_lik -= mu[i] * self._T
            g_i, sum_G_i = self._calc_weights_dim_i(i, b[i])

            for k in range(self._events[i].shape[0]):
                s = mu[i] + a[i] @ (b[i] * g_i[k])
                log_lik += np.log(s)

            log_lik -= a[i] @ sum_G_i

        return log_lik

    def _calc_weights_dim_i(self, i, b_i):
        t_i = self._events[i]
        n_jumps_i = t_i.shape[0]
        g_i = np.zeros((n_jumps_i, self._dim))
        G_i = np.zeros((n_jumps_i + 1, self._dim))
        sum_G_i = np.zeros(self._dim)

        for j in range(self._dim):
            t_j = self._events[j]
            n_jumps_j = t_j.shape[0]
            ij = 0
            for k in range(n_jumps_i + 1):
                t_i_k = t_i[k] if k < n_jumps_i else self._T
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

    def grad(self, params):
        if self._dim != 1:
            raise NotImplementedError('指数カーネルの場合の対数尤度の勾配計算は現在1次元のみ対応しています')

        events = self._events[0]

        mu, a, b = params
        T = self._T
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
