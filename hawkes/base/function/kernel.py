import numpy as np
from abc import ABC

class Kernels(ABC):
    def __init__(self, kernels: np.ndarray):
        self._kernels = kernels

    def __getitem__(self, i: int) -> np.ndarray:
        return self._kernels[i]

    @property
    def value(self) -> np.ndarray:
        return self._kernels
