import numpy as np
from ..vo.events import Events

class Intensities:
    def __init__(self, baselines, kernels, events: Events):
        self._baselines = baselines
        self._kernels = kernels
        self._events = events
        self._dim = events.dim
        self._intensities = [self._Intensity(baselines[i], kernels[i], events) for i in range(self._dim)]

    def __call__(self, t):
        return np.array([self._intensities[i](t) for i in range(self._dim)])

    def __getitem__(self, i):
        return self._intensities[i]

    class _Intensity:
        def __init__(self, baselines_i, kernels_i, events: Events):
            self._baselines_i = baselines_i
            self._kernels_i = kernels_i
            self._events = events.grouped_by_mark
            self._dim = events.dim

        def __call__(self, t):
            is_scalar = False
            if isinstance(t, (int, float)):
                is_scalar = True
                t = np.array([t])

            intensities_i = np.zeros(len(t))

            intensities_i += self._baselines_i

            for j in range(self._dim):
                for k, t_k in enumerate(t):
                    H_t_k = self._events[j][self._events[j] < t_k]
                    intensities_i[k] += np.sum(self._kernels_i[j](t_k - H_t_k))

            return intensities_i if not is_scalar else intensities_i[0]
