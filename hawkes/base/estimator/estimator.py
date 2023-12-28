import numpy as np
from .minimizer import (
    RandomSearch,
    GradientMethod,
    GridSearch,
    ScipyMinimizer,
)
from .output import Output
from abc import ABC, abstractmethod
from ..vo import Events, EventsFactory

class Estimator(ABC):
    def __init__(self, t_interval=1):
        self.t_interval = t_interval
        self._minimization_config = None
        self._dim = None

    def __call__(self, events, T):
        # 1次元の場合の対応
        if isinstance(events, np.ndarray):
            events = [events]

        events = EventsFactory.from_events_grouped_by_mark(events, T)

        minimizer = self._build_minimizer(events.dim)
        loss = self._build_loss(events)
        params, loss = minimizer(loss)
        t = np.arange(0, T + self.t_interval, self.t_interval)
        intensities = self._build_intensities(params, events)

        return Output(
            events=events,
            t=t,
            intensity=intensities(t),
            params=self._format_params(params, events.dim),
            kernel_type=self._get_kernel_type(),
            loglik=-loss,
        )

    def _build_minimizer(self, dim):
        minimization_config = self._minimization_config
        if minimization_config is None:
            minimization_config = self._get_default_minimization_config(dim)

        method = minimization_config['method']
        option = minimization_config['option']
        if method == 'gradient':
            return GradientMethod(
                learning_rate=option.get('learning_rate', 0.0001),
                tol=option.get('tol', 0.01),
                max_iter=option.get('max_iter', 1000000),
                init_params=option.get('init_params'),
            )
        elif method == 'scipy':
            return ScipyMinimizer(
                init_params=option.get('init_params'),
                bounds=option.get('bounds'),
            )
        elif method == 'random_search':
            return RandomSearch(
                option.get('n_iter', 1000000),
                option.get('bounds'),
            )
        elif method == 'grid_search':
            return GridSearch(
                option.get('grid'),
            )
        else:
            raise ValueError('Unknown minimization method: {}'.format(self._minimization_config['method']))

    def set_minimization_config(self, method, option):
        self._minimization_config = {
            'method': method,
            'option': option,
        }

    @abstractmethod
    def _build_loss(self, events: Events):
        pass

    @abstractmethod
    def _build_intensities(self, params, events: Events):
        pass

    @abstractmethod
    def _get_default_minimization_config(self, dim):
        pass

    @abstractmethod
    def _get_kernel_type(self):
        pass

    @abstractmethod
    def _format_params(self, params, dim):
        pass
