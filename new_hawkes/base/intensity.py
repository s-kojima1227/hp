from typing import Any
import numpy as np

class HawkesIntensity:
    def __init__(self, mu, kernel, events):
        self._mu = mu
        self._kernel = kernel
        self._events = events
        self._dim = len(events)
        self._intensity_dim = [HawkesIntensityDim_i(mu[i], kernel[i], events) for i in range(self._dim)]

    def __getitem__(self, i):
        if isinstance(i, int) and (0 <= i < self._dim):
            return self._intensity_dim[i]
        else:
            raise IndexError

class HawkesIntensityDim_i:
    def __init__(self, mu_i, kernel_i, events):
        self._mu_i = mu_i
        self._kernel_i = kernel_i
        self._events = events
        self._dim = len(events)

    def __call__(self, t):
        intensity_i = np.zeros(len(t))
        intensity_i += self._mu_i

        for j in range(self._dim):
            for k, t_k in enumerate(t):
                H_t_k = self._events[j][self._events[j] < t_k]
                intensity_i[k] += np.sum(self._kernel_i[j](t_k - H_t_k))

        return intensity_i
