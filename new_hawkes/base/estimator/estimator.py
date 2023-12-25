import numpy as np
from .optimizer import (
    RandomSearch,
    GradientMethod,
    ScipyMinimizer,
)
from abc import ABC, abstractmethod

class HawkesEstimator(ABC):
    def __init__(self):
        self._optimizer = None

        self._random_search_config = {
            'n_iter': 1000000,
            'search_space': None,
        }
        self._gradient_method_config = {
            'learning_rate': 0.00001,
            'tol': 0.001,
            'max_iter': 1000000,
        }
        self._scipy_minimizer_config = {
            'init_params': None,
            'bounds': None,
        }

    def __call__(self, events, T, type='random_search'):
        # 1次元の場合の対応
        if isinstance(events, np.ndarray):
            events = [events]

        dim = len(events)
        loglik = self._loglik(events, T)

        # FIXME: 本当はこの分岐をやめたいが、目的関数が一致しないため、一時的な対応
        # 処理を統一するには、目的関数を対数尤度から対数尤度の負に変更する必要あり
        if self._optimizer == 'random_search':
            params, score = RandomSearch(**self._random_search_config)(loglik)
        elif self._optimizer == 'gradient_method':
            params, score = GradientMethod(**self._gradient_method_config)(loglik)
        elif self._optimizer == 'scipy_minimizer':
            params, score = ScipyMinimizer(**self._scipy_minimizer_config)(lambda x: -loglik(x))
            score = -score

    @abstractmethod
    def _loglik(self, events, T):
        pass

    def _set_random_search_config(self, n_iter, search_space):
        self._random_search_config['n_iter'] = n_iter
        self._random_search_config['search_space'] = search_space
        self._optimizer = 'random_search'

    def _set_gradient_method_config(self, learning_rate, tol, max_iter, init_params):
        self._gradient_method_config['learning_rate'] = learning_rate
        self._gradient_method_config['tol'] = tol
        self._gradient_method_config['max_iter'] = max_iter
        self._gradient_method_config['init_params'] = init_params
        self._optimizer = 'gradient_method'

    def _set_scipy_minimizer_config(self, init_params, bounds):
        self._scipy_minimizer_config['init_params'] = init_params
        self._scipy_minimizer_config['bounds'] = bounds
        self._optimizer = 'scipy_minimizer'
