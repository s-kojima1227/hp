import numpy as np
from .function import LogLik, Intensities
from ..base import Estimator as Base, Loss
from .converter import BoundsConverter as BC
from .vo import Parameters as Params, ParametersFactory as PF

class Estimator(Base):
    @property
    def _loss_fn(self) -> Loss:
        return LogLik(self._events).to_loss()

    @property
    def _intensities_fn(self) -> Intensities:
        return Intensities(self._params_vo, self._events)

    @property
    def _params_vo(self) -> Params:
        return PF(self._dim).build_from_unpacked(self._params)

    @property
    def _kernel_type(self):
        return 'pow_law'

    @property
    def _default_minimization_config(self):
        dim = self._dim
        init_params = PF(dim).build_from_unpacked(
            np.hstack([np.full(dim + dim * dim, 0.1), np.full(dim * dim + dim * dim, 2)])
        )
        return {
            'method': 'scipy',
            'option': {
                'init_params': init_params,
                'bounds': [(1e-5, None)] * (dim + dim * dim + dim * dim + dim * dim),
            }
        }

    def set_minimization_config(self, method, option):
        if method == 'gradient' or method == 'scipy':
            init_params = option.get('init_params')
            option['init_params'] = PF().build_from_dict(init_params)
        if method == 'scipy' or method == 'random_search':
            bounds = option.get('bounds')
            bounds_mu = bounds.get('mu')
            bounds_K = bounds.get('K')
            bounds_p = bounds.get('p')
            bounds_c = bounds.get('c')
            option['bounds'] = \
                BC.pack(*BC.to_tensor(bounds_mu, bounds_K, bounds_p, bounds_c))
        if method == 'grid_search':
            grid = option.get('grid')
            grid_mu = grid.get('mu')
            grid_K = grid.get('K')
            grid_p = grid.get('p')
            grid_c = grid.get('c')
            option['grid'] = \
                BC.pack(*BC.to_tensor(grid_mu, grid_K, grid_p, grid_c))

        super().set_minimization_config(method, option)
