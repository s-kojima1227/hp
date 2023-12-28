import numpy as np
from ..vo.events import Events

class Intensities:
    def __init__(self, mu, kernel, events: Events):
        self._mu = mu
        self._kernel = kernel
        self._events = events
        self._dim = events.dim
        self._intensity_dim = [self._Intensity(mu[i], kernel[i], events) for i in range(self._dim)]

    def __call__(self, t):
        return np.array([self._intensity_dim[i](t) for i in range(self._dim)])

    def __getitem__(self, i):
        return self._intensity_dim[i]

    class _Intensity:
        def __init__(self, mu_i, kernel_i, events: Events):
            self._mu_i = mu_i
            self._kernel_i = kernel_i
            self._events = events.grouped_by_mark
            self._dim = events.dim

        def __call__(self, t):
            is_scalar = False
            if isinstance(t, (int, float)):
                is_scalar = True
                t = np.array([t])

            intensity_i = np.zeros(len(t))

            intensity_i += self._mu_i

            for j in range(self._dim):
                for k, t_k in enumerate(t):
                    H_t_k = self._events[j][self._events[j] < t_k]
                    intensity_i[k] += np.sum(self._kernel_i[j](t_k - H_t_k))

            return intensity_i if not is_scalar else intensity_i[0]
