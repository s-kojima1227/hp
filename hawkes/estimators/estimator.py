from abc import ABC, abstractmethod
from .optimizers.gradient_optimizer import GradientOptimizer

class BaseEstimator(ABC):
    def __init__(self, optimizer=GradientOptimizer()):
        self._optimizer = optimizer
        self._is_fitted = False

    def fit(self, events, T, init_params=None):
        self._is_fitted = True
        self._events = events
        self._T = T
        self._log_lik_fn = self._build_log_lik_fn(events, T)
        init_params = self._get_default_init_params() if init_params is None else init_params
        bounds = self._get_params_bounds()
        return self._optimizer(self._log_lik_fn, init_params, bounds)

    def log_lik(self, params):
        if not self._is_fitted:
            raise Exception("データをフィットしてください")
        return self._log_lik_fn(params)

    @abstractmethod
    def _build_log_lik_fn(self, events, T):
        pass

    @abstractmethod
    def _get_default_init_params(self):
        pass

    @abstractmethod
    def _get_params_bounds(self):
        pass
