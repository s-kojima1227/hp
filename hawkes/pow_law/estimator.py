import numpy as np
from .loglik import LogLik
from ..base import Estimator as BaseEstimator, Events
from .converter import ParamsConverter, BoundsConverter
from .intensity import Intensity

class Estimator(BaseEstimator):
    def _build_loss(self, events: Events):
        return LogLik(events).build_loss()

    def _get_default_minimization_config(self, dim):
        return {
            'method': 'scipy',
            # FIXME: 初期値が不適当
            'option': {
                'init_params': np.hstack([np.full(dim + dim * dim, 0.1), np.full(dim * dim + dim * dim, 2)]),
                'bounds': [(1e-5, None)] * (dim + dim * dim + dim * dim + dim * dim),
            }
        }

    def set_minimization_config(self, method, option):
        if method == 'gradient' or method == 'scipy':
            init_params = option.get('init_params')
            mu = init_params.get('mu')
            K = init_params.get('K')
            p = init_params.get('p')
            c = init_params.get('c')
            option['init_params'] = \
                ParamsConverter.pack(*ParamsConverter.toTensor(mu, K, p, c))
        if method == 'scipy' or method == 'random_search':
            bounds = option.get('bounds')
            bounds_mu = bounds.get('mu')
            bounds_K = bounds.get('K')
            bounds_p = bounds.get('p')
            bounds_c = bounds.get('c')
            option['bounds'] = \
                BoundsConverter.pack(*BoundsConverter.toTensor(bounds_mu, bounds_K, bounds_p, bounds_c))
        if method == 'grid_search':
            grid = option.get('grid')
            grid_mu = grid.get('mu')
            grid_K = grid.get('K')
            grid_p = grid.get('p')
            grid_c = grid.get('c')
            option['grid'] = \
                BoundsConverter.pack(*BoundsConverter.toTensor(grid_mu, grid_K, grid_p, grid_c))

        super().set_minimization_config(method, option)

    def _build_intensity(self, params, events: Events):
        mu, K, p, c = ParamsConverter.unpack(params, events.dim)
        return Intensity(mu, K, p, c, events)

    def _format_params(self, params, dim):
        return ParamsConverter.toDict(*ParamsConverter.unpack(params, dim))

    def _get_kernel_type(self):
        return 'pow_law'
