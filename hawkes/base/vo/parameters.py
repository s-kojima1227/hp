from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

class Parameters(ABC):
    @property
    @abstractmethod
    def unpacked(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dict(self) -> Dict:
        pass

    @property
    @abstractmethod
    def baselines(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def kernel_params(self) -> Tuple[np.ndarray, ...]:
        pass

class ParametersFactory(ABC):
    @abstractmethod
    def build_from_packed(self, baselines: np.ndarray, kernel_params: Tuple) -> 'Parameters':
        pass

    @abstractmethod
    def build_from_unpacked(self, params_unpacked: np.ndarray) -> 'Parameters':
        pass