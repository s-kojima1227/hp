from abc import ABC, abstractmethod

class HawkesLogLik(ABC):
    def __init__(self, events, T):
        self._events = events
        self._T = T
        self._dim = len(events)

    @abstractmethod
    def __call__(self, params):
        pass

    @abstractmethod
    def grad(self, params):
        pass
