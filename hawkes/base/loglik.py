from abc import ABC, abstractmethod
from typing import Any

class LogLik(ABC):
    def __init__(self, events, T):
        self._events = events
        self._T = T
        self._dim = len(events)

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
