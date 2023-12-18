import numpy as np

class ExpKernel1DLogLik:
    def __call__(self, params, events, T):
        mu, a, b = params
        n = len(events)
        log_lik = 0
        log_lik += np.sum([np.log(mu + np.sum(a * b * np.exp(-b * (events[i] - events[:i])))) for i in range(n)])
        log_lik -= mu * T
        log_lik -= np.sum(a * (1 - np.exp(-b * (T - events))))

        return log_lik
