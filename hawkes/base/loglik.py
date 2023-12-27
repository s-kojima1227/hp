from abc import ABC, abstractmethod
from typing import Any
from .events import Events

class LogLik(ABC):
    def __init__(self, events: Events, intensity=None, compensator=None):
        self._dim = events.dim
        self._events = events.grouped_by_mark
        self._T = events.end_time
        self._intensity = intensity
        self._compensator = compensator

    def build_loss(self):
        return Loss(self)

    @abstractmethod
    def __call__(self, params):
        pass

    @abstractmethod
    def grad(self, params):
        pass

class Loss:
    def __init__(self, loglik: LogLik):
        self._loglik = loglik

    def __call__(self, params: Any):
        return -self._loglik(params)

    def grad(self, params: Any):
        return -self._loglik.grad(params)
