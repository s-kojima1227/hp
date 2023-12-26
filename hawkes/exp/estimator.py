import numpy as np
from .loglik import LogLik
from ..base import Estimator as BaseEstimator
from .converter import ParamsConverter, BoundsConverter
from .intensity import Intensity

class Estimator(BaseEstimator):
    def _build_loss(self, events, T):
        return LogLik(events, T).build_loss()

    def _get_default_minimization_config(self, dim):
        return {
            'method': 'scipy',
            'option': {
                'init_params': np.full(dim + dim * dim + dim * dim, 0.1),
                'bounds': [(1e-5, None)] * (dim + dim * dim + dim * dim),
            }
        }

    def set_minimization_config(self, method, option):
        if method == 'gradient' or method == 'scipy':
            init_params = option.get('init_params')
            mu = init_params.get('mu')
            a = init_params.get('a')
            b = init_params.get('b')
            option['init_params'] = \
                ParamsConverter.pack(*ParamsConverter.toTensor(mu, a, b))
        if method == 'scipy' or method == 'random_search':
            bounds = option.get('bounds')
            bounds_mu = bounds.get('mu')
            bounds_a = bounds.get('a')
            bounds_b = bounds.get('b')
            option['bounds'] = \
                BoundsConverter.pack(*BoundsConverter.toTensor(bounds_mu, bounds_a, bounds_b))
        if method == 'grid_search':
            grid = option.get('grid')
            grid_mu = grid.get('mu')
            grid_a = grid.get('a')
            grid_b = grid.get('b')
            option['grid'] = \
                BoundsConverter.pack(*BoundsConverter.toTensor(grid_mu, grid_a, grid_b))

        super().set_minimization_config(method, option)

    def _build_intensity(self, params, events, dim):
        mu, a, b = ParamsConverter.unpack(params, dim)
        return Intensity(mu, a, b, events)

    def _format_params(self, params, dim):
        return ParamsConverter.toDict(*ParamsConverter.unpack(params, dim))

    def _get_kernel_type(self):
        return 'exp'
