import numpy as np

class ExpKernelLogLik:
    def __init__(self, events, T):
        self._events = events
        self._T = T
        self._dim = events.shape[0]

    def __call__(self, mu, a, b):
        log_lik = 0
        for i in range(self._dim):
            loss -= mu[i] * self._T
            g_i, sum_G_i = self._calc_weights_dim_i(i, b[i])
            for k in range(self._events[i].shape[0]):
                s = mu[i] + a[i] @ (b[i] * g_i[k])
                if s <= 0:
                    raise ValueError('影響率の合計は負になりえません')
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
