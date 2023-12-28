import numpy as np
from typing import Union

class Kernels:
    def __init__(self, adjacencies: np.ndarray, decays: np.ndarray):
        dim = adjacencies.shape[0]
        self._kernels = np.array([[self._Kernel(adjacencies[i, j], decays) for j in range(dim)] for i in range(dim)], dtype=object)

    def __getitem__(self, i):
        return self._kernels[i]

    @property
    def value(self):
        return self._kernels

    class _Kernel:
        def __init__(self, adjacencies_ij: np.ndarray, decays: np.ndarray):
            self._adjacencies_ij = adjacencies_ij
            self._decays = decays

        def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            if isinstance(t, float):
                return np.sum(self._adjacencies_ij * self._decays * np.exp(-self._decays * t))
            else:
                adjacency_ij = np.reshape(self._adjacencies_ij, (-1, 1))
                decays = np.reshape(self._decays, (-1, 1))
                t = np.reshape(t, (1, -1))

                return np.sum(
                    adjacency_ij * decays * np.exp(np.matmul(-decays, t)),
                    axis=0
                )
