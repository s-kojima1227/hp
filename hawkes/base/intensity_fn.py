import numpy as np
from .function_io import FunctionIO

class IntensityFunction:
    def __init__(self, mu, kernel, events):
        self._mu = mu
        self._kernel = kernel
        self._events = events
        self._dim = len(events)

    def __call__(self, t_vals):
        intensity = np.zeros((self._dim, len(t_vals)))
        for i in range(self._dim):
            intensity[i] += self._mu[i]
            for j in range(self._dim):
                for k, t in enumerate(t_vals):
                    intensity[i, k] += np.sum(self._kernel[i, j](t - self._events[j][self._events[j] < t]))

        return [FunctionIO(t_vals, intensity[i]) for i in range(self._dim)]
