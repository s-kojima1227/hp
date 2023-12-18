from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def __call__(self, objective_fn, init_params, bounds=None):
        pass
