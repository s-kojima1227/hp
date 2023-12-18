import numpy as np

class PowLawKernel1DLogLik:
    def __call__(self, params, events, T):
        mu, K, p, c = params
        n = len(events)
        log_lik = 0
        log_lik += np.sum([np.log(mu + np.sum(K * np.power(events[i] - events[:i] + c, -p))) for i in range(n)])
        log_lik -= mu * T
        log_lik += K / (p - 1) * np.sum(np.power(T - events + c, -(p - 1)) - np.power(c, p - 1))

        return log_lik
