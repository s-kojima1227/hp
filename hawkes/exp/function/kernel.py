import numpy as np
from ...base import Kernels as Base

class Kernels(Base):
    def __init__(self, adjacencies: np.ndarray, decays: np.ndarray):
        if not (adjacencies.shape == decays.shape):
            raise ValueError('パラメーターの次元が不適切です: adjacencies.shape={}, decays.shape={}'.format(adjacencies.shape, decays.shape))
        dim = adjacencies.shape[0]
        kernels = np.array([[self._Kernel(adjacencies[i, j], decays[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)
        super().__init__(kernels)

    class _Kernel:
        def __init__(self, adjacency: float, decay: float):
            self._adjacency = adjacency
            self._decay = decay

        def __call__(self, t) -> float:
            return self._adjacency * self._decay * np.exp(-self._decay * t)
