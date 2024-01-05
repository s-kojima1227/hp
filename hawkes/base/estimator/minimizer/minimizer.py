from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np

class Minimizer(ABC):
    @abstractmethod
    def __call__(self, objective_fn: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        pass
