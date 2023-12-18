import numpy as np

class ExpKernel1DLogLik:
    def __call__(self, params, events, T):
        mu, a, b = params
        n = len(events)
        logLik = 0

        for i in range(n):
            logLik += np.log(mu + np.sum(a * b * np.exp(-b * (events[i] - events[:i]))))
        logLik -= mu * T
        logLik -= np.sum(a * (1 - np.exp(-b * (T - events))))

        return logLik
