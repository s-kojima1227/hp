from ..vo import Events
import numpy as np
from abc import ABC, abstractmethod

class LogLik(ABC):
    def __init__(self, events: Events):
        self._events = events

    def __call__(self, params: np.ndarray) -> float:
        H_T = self._events.ordered_by_time
        T = self._events.end_time

        loglik = 0
        for (t_i, m_i) in H_T:
            intensity = self._intensity_i(int(m_i), t_i, self._events, params)
            loglik += np.log(intensity)

        compensators = self._compensators(T, self._events, params)
        loglik -= np.sum(compensators)

        return loglik

    def to_loss(self):
        return Loss(self)

    @abstractmethod
    def _intensity_i(self, mark, time, events: Events, params: np.ndarray) -> float:
        pass

    @abstractmethod
    def _compensators(self, time, events: Events, params: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def grad(self, params) -> np.ndarray:
        pass

class Loss:
    def __init__(self, loglik: LogLik):
        self._loglik = loglik

    def __call__(self, params: np.ndarray) -> float:
        return -self._loglik(params)

    def grad(self, params: np.ndarray) -> np.ndarray:
        return -self._loglik.grad(params)